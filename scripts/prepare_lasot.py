#!/usr/bin/env python
"""Fetch and crop the LaSOT clips that experiments/media_patch.py reads (plan.md, part 2): curated YouTube videos with the object
boxed by hand on every frame, as 8 frames at the same times as the VGGSound clips.

    uv run scripts/prepare_lasot.py

LaSOT (Fan et al., CVPR 2019) has 20 sequences in each of our 8 classes, at about 30 frames per second and 83 seconds long on
average. Each class is one zip of 2 to 5 GB on Hugging Face; the zip is read in place over HTTP range requests, so only the
annotations and the 8 chosen frames of each sequence are downloaded (a few MB per class), never the whole archive.

Per sequence, fixed before any model sees a frame:
1. The window: 10 seconds (300 frames at the assumed 30 fps), starting at frame 0, 150, 300, ... The 8 frames sit at
   media.FRAME_TIMES within it. A window is usable when the object is in all 8 frames (not fully occluded, not out of view,
   a box of positive size). Of the usable windows, the first whose object is at least 0.3 of the crop side (mean of the 8
   boxes' longer sides) is taken; if none reaches 0.3, the one with the largest ratio. A sequence with no usable window is
   skipped and listed.
2. The crop: one square for all 8 frames, so the motion stays in the clip: the union of the 8 boxes, its longer side plus 15%,
   no larger than the frame's shorter side, centred on the union and kept inside the frame; resized to 224 x 224 (bicubic).
3. data/lasot/clips/<sequence>.npz holds frames uint8 [8, 224, 224, 3].

The manifest (the revision, every sequence's window, frame numbers, crop, object ratio and the hash of its frames, every
skipped sequence and why) is written to data/lasot/manifest.json and to scripts/lasot_manifest.json, which is committed. Once
committed it is the reference: a later run fetches the frames it names and stops on a different hash.
"""

import argparse
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from jspace import media
from jspace.config import DATA_DIR, LASOT_REPO, LASOT_REVISION, ROOT
from jspace.runs import load_env

TRACKED = ROOT / "scripts/lasot_manifest.json"
FPS = 30  # LaSOT's frame rate, assumed for every sequence (recorded in the manifest)
WINDOW_STEP = 150  # window starts every 5 s
MIN_RATIO = 0.3  # the object's size against the crop side
MARGIN = 0.15


def array_sha256(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def parse_boxes(text):
    """groundtruth.txt: one x,y,w,h line per frame (comma- or whitespace-separated) -> float [n, 4]"""
    rows = [re.split(r"[,\s]+", line.strip()) for line in text.splitlines() if line.strip()]
    return np.array([[float(v) for v in r[:4]] for r in rows], dtype=np.float64)


def parse_flags(text):
    """full_occlusion.txt / out_of_view.txt: 0/1 per frame, comma-separated on one line or one per line -> bool [n]"""
    return np.array([v == "1" for v in re.split(r"[,\s]+", text.strip()) if v != ""], dtype=bool)


def frame_offsets():
    return [round(t * FPS) for t in media.FRAME_TIMES]


def crop_square(boxes, width, height):
    """(left, top, side): one square holding the union of the boxes plus MARGIN, inside the frame"""
    x0, y0 = boxes[:, 0].min(), boxes[:, 1].min()
    x1, y1 = (boxes[:, 0] + boxes[:, 2]).max(), (boxes[:, 1] + boxes[:, 3]).max()
    side = min(max(x1 - x0, y1 - y0) * (1 + MARGIN), width, height)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    left = min(max(cx - side / 2, 0), width - side)
    top = min(max(cy - side / 2, 0), height - side)
    return float(left), float(top), float(side)


CLIP_FRAMES = int(media.CLIP_SECONDS * FPS)


def choose_window(boxes, occluded, out_of_view, width, height):
    """the window rule above: (start, frame indices, crop, ratio), or None when no window has the object in all 8 frames"""
    n = min(len(boxes), len(occluded), len(out_of_view))
    offs = frame_offsets()
    usable = []
    for start in range(0, n - CLIP_FRAMES + 1, WINDOW_STEP):
        idx = [start + o for o in offs]
        b = boxes[idx]
        if occluded[idx].any() or out_of_view[idx].any() or (b[:, 2] <= 0).any() or (b[:, 3] <= 0).any():
            continue
        crop = crop_square(b, width, height)
        ratio = float(np.maximum(b[:, 2], b[:, 3]).mean() / crop[2])
        usable.append((start, idx, crop, ratio))
    if not usable:
        return None
    return next((u for u in usable if u[3] >= MIN_RATIO), max(usable, key=lambda u: u[3]))


def render(images, crop):
    left, top, side = crop
    box = (round(left), round(top), round(left + side), round(top + side))
    return np.stack([np.asarray(im.convert("RGB").crop(box).resize((media.SIDE, media.SIDE), Image.BICUBIC)) for im in images]).astype(np.uint8)


class ClassZip:
    """one class's zip on the Hub, read in place: annotations and single frames by sequence"""

    def __init__(self, cls):
        from huggingface_hub import HfFileSystem

        self.f = HfFileSystem().open(f"datasets/{LASOT_REPO}@{LASOT_REVISION}/{cls}.zip", "rb", block_size=1 << 20)
        self.z = zipfile.ZipFile(self.f)
        pat = re.compile(rf"(?:^|/)({re.escape(cls)}-\d+)/(.+)$")
        self.index = {}  # sequence -> path inside the sequence's folder -> zip member
        for name in self.z.namelist():
            m = pat.search(name)
            if m and not name.split("/")[-1].startswith("._"):  # macOS resource forks are not frames
                self.index.setdefault(m.group(1), {})[m.group(2)] = name
        self.seqs = sorted(self.index, key=lambda s: int(s.rsplit("-", 1)[1]))

    def member(self, seq, rel):
        try:
            return self.index[seq][rel]
        except KeyError:
            raise ValueError(f"{seq}: no {rel} in the zip") from None

    def text(self, seq, rel):
        return self.z.read(self.member(seq, rel)).decode("utf-8", "replace")

    def frame(self, seq, i):
        """frame i (0-based) of a sequence: img/00000001.jpg is frame 0"""
        return Image.open(io.BytesIO(self.z.read(self.member(seq, f"img/{i + 1:08d}.jpg"))))


def prepare_sequence(cz, seq):
    boxes = parse_boxes(cz.text(seq, "groundtruth.txt"))
    occ = parse_flags(cz.text(seq, "full_occlusion.txt"))
    oov = parse_flags(cz.text(seq, "out_of_view.txt"))
    first = cz.frame(seq, 0)
    w, h = first.size
    pick = choose_window(boxes, occ, oov, w, h)
    if pick is None:
        raise ValueError("no 10-second window with the object in all 8 frames")
    start, idx, crop, ratio = pick
    frames = render([cz.frame(seq, i) for i in idx], crop)
    return frames, dict(start_frame=int(start), frames=[int(i) for i in idx], crop=[round(v, 1) for v in crop], object_ratio=round(ratio, 3), frame_size=[w, h], n_frames=int(len(boxes)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = ap.parse_args()
    load_env()
    if LASOT_REVISION is None:
        raise SystemExit("LaSOT's revision is not pinned in src/jspace/config.py: run `uv run scripts/pin_media.py` first")
    root = args.data_dir / "lasot"
    (root / "clips").mkdir(parents=True, exist_ok=True)
    ref = json.loads(TRACKED.read_text()) if TRACKED.exists() else None
    if ref is not None and ref["revision"] != LASOT_REVISION:
        raise SystemExit(f"{TRACKED.name} was made at {ref['revision']}, not {LASOT_REVISION}: delete it on purpose to redraw")
    kept, skipped, bad = [], [], []
    for cls in media.CLASSES:
        cz = ClassZip(cls)
        want = [c["id"] for c in ref["clips"] if c["class"] == cls] if ref else cz.seqs
        print(f"{cls}: {len(cz.seqs)} sequences in the zip, {len(want)} to prepare")
        for seq in want:
            path = root / "clips" / f"{seq}.npz"
            try:
                frames, facts = prepare_sequence(cz, seq)
            except (ValueError, KeyError, OSError) as e:
                skipped.append({"id": seq, "class": cls, "reason": str(e)[:200]})
                print(f"skip {seq}: {e}")
                if ref is not None:
                    bad.append(seq)  # a clip of the reference that can no longer be made
                continue
            facts = {"id": seq, "class": cls, **facts, "frames_sha256": array_sha256(frames)}
            if ref is not None:
                exp = next(c for c in ref["clips"] if c["id"] == seq)
                if exp["frames_sha256"] != facts["frames_sha256"]:
                    bad.append(seq)
            np.savez(path, frames=frames)
            kept.append(facts)
    if bad:
        raise SystemExit(f"{len(bad)} clips differ from {TRACKED.name} (e.g. {bad[:3]})")
    man = dict(
        repo=LASOT_REPO,
        revision=LASOT_REVISION,
        fps_assumed=FPS,
        window_seconds=media.CLIP_SECONDS,
        window_step_frames=WINDOW_STEP,
        min_ratio=MIN_RATIO,
        margin=MARGIN,
        frame_times=media.FRAME_TIMES,
        side=media.SIDE,
        clips=kept,
        skipped=skipped,
    )
    text = json.dumps(man, indent=2) + "\n"
    (root / "manifest.json").write_text(text)
    if ref is None:
        TRACKED.write_text(text)
        print(f"saved {TRACKED}: commit it, so every later run is checked against these hashes")
    print(f"{len(kept)} clips ready, {len(skipped)} skipped: " + ", ".join(f"{w} {sum(c['class'] == w for c in kept)}" for w in media.CLASSES))


if __name__ == "__main__":
    main()
