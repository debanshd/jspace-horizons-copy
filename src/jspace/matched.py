"""The matched video set for experiments/video_sweep.py (plan.md, part 3): Nayan Saxena's saxenan3/jspace-matched-video.

64 scenes on the image experiment's schedule: a donor video and a receiver video, their labels (the 16 COCO classes of
stimuli.VOCAB), the numbered canvas slot, and the three names of the membership question. 73 silent H.264 clips at 24 fps from
Wikimedia Commons and uCO3D, with a sha256 per clip in ATTRIBUTION.tsv.

Frames. Each model's own frame sampler picks the frames (`native_indices`): Qwen3.5's samples 2 frames per second, Gemma-4's a
fixed 32 spread over the clip. Patching writes a donor's media-token states into the receiver position by position, so both
clips of a scene need the same frame count: the scene's count is the receiver's native count, capped at both clips' lengths
(Gemma's sampler refuses to take more frames than a clip has), and both clips are sampled at that count by the same sampler.
Every sampled frame is drawn into the slot of the six-slot canvas as the image experiment draws a crop: fitted into the
184-pixel tile on the grey background (as the dataset's render_inputs.py does).
"""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import stimuli
from .analysis import noun_matches

DIR = "matched_video"
VOCAB = list(stimuli.VOCAB)
ANIMALS, RIDDEN = stimuli.ANIMALS, stimuli.RIDDEN
INPUTS = ["moving", "frozen", "still"]  # the video, its middle sampled frame repeated, that frame as a picture
MEDIUM = {"moving": "video", "frozen": "video", "still": "picture"}


def scenes(data_dir):
    path = Path(data_dir) / DIR / "test/metadata.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} not found: run `uv run scripts/prepare_matched.py` first")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return sorted(rows, key=lambda r: r["scene_id"])


def attribution(data_dir):
    with open(Path(data_dir) / DIR / "ATTRIBUTION.tsv", newline="") as f:
        return {r["clip_id"]: r for r in csv.DictReader(f, delimiter="\t")}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


# ---- the questions, as depth_sweep asks and scores them


def truth(task, w, membership):
    if task == "say":
        return w
    if task == "letter":
        return w[0].upper()
    if task == "set":
        return "7" if w in membership else "3"
    return "7" if w in (RIDDEN if task == "ride" else ANIMALS) else "3"


def matches(task, text, target):
    """depth_sweep's scoring: the first vocabulary word for say, the first character otherwise"""
    if task == "say":
        return bool(noun_matches([text], [target], set(VOCAB))[0])
    return text.strip().strip("*").upper()[:1] == target


def same_kind_word(receiver, donor, scene_id):
    """the steering control Z: a word of the receiver's kind (animal or not, ridden or not, as name_steering chooses its Z),
    not the receiver or the donor, with another first letter where possible; chosen by the scene number"""
    pool = [w for w in VOCAB if w not in (receiver, donor) and (w in ANIMALS) == (receiver in ANIMALS) and (w in RIDDEN) == (receiver in RIDDEN)]
    pool = pool or [w for w in VOCAB if w not in (receiver, donor) and (w in ANIMALS) == (receiver in ANIMALS)]  # horse: the only ridden animal
    pool = [w for w in pool if w[0] != receiver[0]] or pool
    return pool[scene_id % len(pool)]


# ---- frames


def native_indices(video_processor, total, fps, num_frames=None):
    """the model's own sampler on a clip of `total` frames at `fps`: its default (num_frames None) or a given count"""
    from transformers.video_utils import VideoMetadata

    md = VideoMetadata(total_num_frames=int(total), fps=float(fps), duration=total / fps)
    idx = video_processor.sample_frames(md) if num_frames is None else video_processor.sample_frames(md, num_frames=int(num_frames))
    return [int(i) for i in np.asarray(idx)]


def scene_frame_count(video_processor, receiver_total, donor_total, fps):
    """the scene's frame count: the receiver's native count, capped at both clips' lengths"""
    try:
        n = len(native_indices(video_processor, receiver_total, fps))
    except ValueError:  # Gemma's sampler: more frames asked than the clip has
        n = receiver_total
    return int(min(n, receiver_total, donor_total))


def video_info(path):
    """(frame count, frames per second) of a clip, counting its video packets"""
    import av

    with av.open(str(path)) as c:
        s = c.streams.video[0]
        n = sum(1 for p in c.demux(s) if p.size)
        return n, float(s.average_rate)


def decode(path, indices):
    """the clip's frames at the given 0-based indices, in order, as RGB images (an index may repeat)"""
    import av

    want = sorted(set(indices))
    got, i = {}, 0
    with av.open(str(path)) as c:
        for f in c.decode(video=0):
            if i in want:
                got[i] = f.to_image().convert("RGB")
            i += 1
            if len(got) == len(want):
                break
    missing = [j for j in want if j not in got]
    if missing:
        raise ValueError(f"{path.name}: frames {missing[:5]} not decoded ({i} frames read)")
    return [got[j] for j in indices]


def canvas_frames(images, slot):
    """each frame fitted into the crop tile and drawn into the canvas slot: uint8 [n, 448, 672, 3]"""
    out = []
    for im in images:
        im = im.copy()
        im.thumbnail((stimuli.CROP, stimuli.CROP), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (stimuli.CROP, stimuli.CROP), stimuli.BG)
        tile.paste(im, ((stimuli.CROP - im.width) // 2, (stimuli.CROP - im.height) // 2))
        out.append(np.asarray(stimuli.scene(slot, tile)))
    return np.stack(out)


def input_of(frames, kind):
    """one input of a clip from its canvases: the video, the middle canvas repeated as often, or the middle canvas as a picture
    (the middle of the sampled frames, as the dataset's render_inputs.py takes it)"""
    if kind == "moving":
        return frames
    mid = frames[len(frames) // 2]
    return np.repeat(mid[None], len(frames), axis=0) if kind == "frozen" else Image.fromarray(mid)
