#!/usr/bin/env python
"""Download and decode the VGGSound clips that experiments/media_steer.py reads (plan.md, "Data").

    uv run --with av==15.1.0 scripts/prepare_media.py                  # 16 candidates per class, about 200 MB
    uv run --with av==15.1.0 scripts/prepare_media.py --candidates 32  # when the gate leaves a class short

PyAV decodes the mp4 files here and nowhere else, so it is not a dependency of the project: `--with` adds it for this command
and uv.lock is untouched. The dataset revisions must be pinned first (`uv run scripts/pin_media.py`).

What it does, in order:
1. vggsound.csv (YouTube id, start, label, split) from VGG_CSV_REPO, and the list of clips in VGG_CLIPS_REPO's data/ (the
   first shard of VGGSound, about 10,000 clips, one mp4 each), both at their pinned revisions.
2. Each clip's class: the one class word its label names (media.label_class); clips of no class or of two are not used.
3. Per class (in media.CLASSES order), the clips sorted by name and shuffled with one generator seeded 0; the first N that
   decode are the candidates. A clip is skipped (and listed) when it has no video or no audio stream or under 9 s of video.
4. Per candidate: the mp4 (its sha256 recorded), 8 frames at media.FRAME_TIMES (the first frame at or after each time),
   centre-cropped to a square and resized to 224 x 224 (bicubic), and the sound in mono at 16 kHz, padded with silence or
   cut to exactly 10 s, into data/vggsound/clips/<id>.npz.
5. The manifest: the revisions, the label-to-class table, the counts per class in the shard, every candidate with its hashes
   (the mp4 file, and the decoded frames and sound as raw arrays: an .npz file carries a timestamp, so its own hash would
   change on every write) and durations, and every skipped clip with the reason. It is written twice: to
   data/vggsound/manifest.json, which the experiment reads, and to scripts/vggsound_manifest.json, which is committed.

Once scripts/vggsound_manifest.json is committed it is the reference: a later run on any machine downloads the clips it
lists, at its revisions, and stops if an mp4 or a decoded clip has a different hash. A run whose data folder already holds
every clip only checks the decoded arrays. To draw a new set (another --candidates), delete the committed manifest on purpose.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image

from jspace import media
from jspace.config import DATA_DIR, ROOT, VGG_CLIPS_REPO, VGG_CLIPS_REVISION, VGG_CSV_REPO, VGG_CSV_REVISION
from jspace.runs import load_env


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


TRACKED = ROOT / "scripts/vggsound_manifest.json"


def array_sha256(*arrays):
    """the hash of the decoded clip: the raw bytes of its arrays, in order"""
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def square(img):
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    return img.crop((left, top, left + s, top + s)).resize((media.SIDE, media.SIDE), Image.BICUBIC)


def decode(path):
    """(frames uint8 [8, 224, 224, 3], audio float32 [160000], facts) or raises ValueError with the reason"""
    import av

    facts = {}
    with av.open(str(path)) as c:
        if not c.streams.video:
            raise ValueError("no video stream")
        out, t0, last, idx = [None] * media.N_FRAMES, None, None, 0
        for f in c.decode(video=0):
            t = float(f.pts * f.time_base) if f.pts is not None else None
            if t is None:
                continue
            t0 = t if t0 is None else t0
            last = (t - t0, f)
            while idx < media.N_FRAMES and t - t0 >= media.FRAME_TIMES[idx]:
                out[idx] = square(f.to_image())
                idx += 1
        if last is None:
            raise ValueError("no decodable video frame")
        facts["video_seconds"] = round(last[0], 3)
        if facts["video_seconds"] < 9.0:
            raise ValueError(f"video only {facts['video_seconds']} s")
        facts["frames_repeated_at_end"] = media.N_FRAMES - idx  # a clip whose last frame comes before 9.375 s repeats it
        for j in range(idx, media.N_FRAMES):
            out[j] = square(last[1].to_image())
    with av.open(str(path)) as c:
        if not c.streams.audio:
            raise ValueError("no audio stream")
        rs = av.AudioResampler(format="flt", layout="mono", rate=media.SAMPLE_RATE)
        parts = []
        for f in c.decode(audio=0):
            parts += [g.to_ndarray().reshape(-1) for g in rs.resample(f)]
        parts += [g.to_ndarray().reshape(-1) for g in rs.resample(None)]
    if not parts:
        raise ValueError("no decodable audio")
    a = np.concatenate(parts).astype(np.float32)
    n = int(media.CLIP_SECONDS * media.SAMPLE_RATE)
    facts["audio_seconds"] = round(len(a) / media.SAMPLE_RATE, 3)
    facts["audio_padded_seconds"] = round(max(0, n - len(a)) / media.SAMPLE_RATE, 3)
    audio = np.zeros(n, np.float32)
    audio[: min(n, len(a))] = a[:n]
    frames = np.stack([np.asarray(im.convert("RGB")) for im in out]).astype(np.uint8)
    return frames, audio, facts


def fetch_and_decode(cid, root):
    mp4 = Path(hf_hub_download(VGG_CLIPS_REPO, f"data/{cid}.mp4", repo_type="dataset", revision=VGG_CLIPS_REVISION, local_dir=root / "mp4"))
    frames, audio, facts = decode(mp4)
    np.savez(root / "clips" / f"{cid}.npz", frames=frames, audio=audio)
    return dict(mp4_sha256=sha256(mp4), frames_sha256=array_sha256(frames), audio_sha256=array_sha256(audio), **facts)


def check_decoded(man, clips_dir):
    """every clip of a manifest present and identical to it, as decoded arrays: [(id, problem)]"""
    bad = []
    for c in man["clips"]:
        path = clips_dir / f"{c['id']}.npz"
        if not path.exists():
            bad.append((c["id"], "missing"))
            continue
        z = np.load(path)
        if array_sha256(z["frames"]) != c["frames_sha256"] or array_sha256(z["audio"]) != c["audio_sha256"]:
            bad.append((c["id"], "decoded arrays differ"))
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", type=int, default=16, help="candidate clips per class")
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = ap.parse_args()
    load_env()
    if VGG_CLIPS_REVISION is None or VGG_CSV_REVISION is None:
        raise SystemExit("the VGGSound revisions are not pinned in src/jspace/config.py: run `uv run scripts/pin_media.py` first")
    import av

    root = args.data_dir / "vggsound"
    clips_dir = root / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    man_path = root / "manifest.json"
    revisions = dict(clips=VGG_CLIPS_REVISION, csv=VGG_CSV_REVISION)

    if TRACKED.exists():  # the committed reference: the same clips, the same bytes
        ref = json.loads(TRACKED.read_text())
        if ref["revisions"] != revisions or ref["candidates_per_class"] != args.candidates:
            raise SystemExit(
                f"{TRACKED.name} was drawn at {ref['revisions']} with {ref['candidates_per_class']} candidates per class, not {revisions} with {args.candidates}: "
                "delete it on purpose to draw a new set"
            )
        todo = [c for c in ref["clips"] if check_decoded(dict(clips=[c]), clips_dir)]  # missing here, or not the reference's arrays
        bad = []
        for c in todo:
            got = fetch_and_decode(c["id"], root)
            for k in ("mp4_sha256", "frames_sha256", "audio_sha256"):
                if got[k] != c[k]:
                    bad.append((c["id"], k))
        if bad:
            raise SystemExit(f"{len(bad)} hashes differ from {TRACKED.name} (e.g. {bad[:3]}): the clips or the decoder are not the ones the manifest was made with")
        man_path.write_text(TRACKED.read_text())
        print(f"ok  {len(ref['clips'])} clips match {TRACKED.name} ({len(todo)} downloaded and decoded now)")
        return

    csv_path = Path(hf_hub_download(VGG_CSV_REPO, "vggsound.csv", repo_type="dataset", revision=VGG_CSV_REVISION, local_dir=root))
    labels = {}
    with open(csv_path, newline="") as f:
        for ytid, start, label, split in csv.reader(f):
            labels[f"{ytid}_{int(start):06d}"] = (label, split)
    files = HfApi().list_repo_files(VGG_CLIPS_REPO, repo_type="dataset", revision=VGG_CLIPS_REVISION)
    shard = sorted(Path(p).stem for p in files if p.startswith("data/") and p.endswith(".mp4"))
    missing = [s for s in shard if s not in labels]
    print(f"{len(shard)} clips in {VGG_CLIPS_REPO}; {len(missing)} without a row in vggsound.csv")
    label_map = {lab: media.label_class(lab) for lab in sorted({labels[s][0] for s in shard if s in labels})}
    by_class = {w: [s for s in shard if s in labels and label_map[labels[s][0]] == w] for w in media.CLASSES}
    print("clips per class in the shard: " + ", ".join(f"{w} {len(v)}" for w, v in by_class.items()))
    rng = np.random.default_rng(0)
    order = {w: [by_class[w][i] for i in rng.permutation(len(by_class[w]))] for w in media.CLASSES}

    kept, skipped = [], []
    for w in media.CLASSES:
        n = 0
        for cid in order[w]:
            if n == args.candidates:
                break
            try:
                facts = fetch_and_decode(cid, root)
            except (ValueError, av.error.FFmpegError) as e:
                skipped.append({"id": cid, "class": w, "reason": str(e)[:200]})
                print(f"skip {cid} ({w}): {e}")
                continue
            kept.append({"id": cid, "class": w, "label": labels[cid][0], "split": labels[cid][1], **facts})
            n += 1
        if n < args.candidates:
            print(f"warning: only {n} decodable clips of {w} in the shard")
    man = dict(
        revisions=revisions,
        repos=dict(clips=VGG_CLIPS_REPO, csv=VGG_CSV_REPO),
        candidates_per_class=args.candidates,
        seed=0,
        av_version=av.__version__,
        frame_times=media.FRAME_TIMES,
        side=media.SIDE,
        sample_rate=media.SAMPLE_RATE,
        label_map={k: v for k, v in label_map.items() if v is not None},
        shard_counts={w: len(v) for w, v in by_class.items()},
        clips=kept,
        skipped=skipped,
    )
    text = json.dumps(man, indent=2) + "\n"
    man_path.write_text(text)
    TRACKED.write_text(text)
    print(f"saved {man_path}: {len(kept)} clips, {len(skipped)} skipped")
    print(f"saved {TRACKED}: commit it, so every later download is checked against these hashes")


if __name__ == "__main__":
    main()
