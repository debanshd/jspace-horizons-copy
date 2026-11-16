"""Video clips and sounds for experiments/media_steer.py (plan.md): the classes, the clips as the models receive them, the
prompts, the scoring, and which word steers which clip.

A clip is a 10-second VGGSound video. scripts/prepare_media.py decodes each one once into data/vggsound/clips/<id>.npz:
    frames  uint8 [8, 224, 224, 3]  eight frames at fixed times (FRAME_TIMES), centre-cropped to a square and resized
    audio   float32 [160000]        the sound track, mono, 16 kHz, exactly 10 s (a shorter track is padded with silence)
Fixed frame times and a fixed audio length give every clip the same number of media tokens, so every prompt of one input kind
and question has the same length and batches need no padding (left padding moved Gemma-4's numbers; Session then ran it at
batch 1).

The four inputs per clip (plan.md, "Design"): the clip (8 frames as a video), the static video (the middle frame 8 times), the
middle frame as a picture, and the sound (Gemma-4 E2B and E4B only).
"""

import json
import re
from pathlib import Path

import numpy as np

from . import stimuli

CLASSES = ["dog", "cat", "bird", "horse", "car", "bus", "train", "airplane"]  # four animals, four vehicles
ANIMALS = set(stimuli.ANIMALS)
assert ANIMALS == {"dog", "cat", "bird", "horse"} and ANIMALS <= set(CLASSES)
SMOKE_CLASSES = ["airplane", "bird", "dog", "car", "train"]  # plan.md, "Data": the 5-clip smoke check
SMOKE_Y = {"airplane": "bird", "bird": "airplane", "dog": "car", "car": "dog", "train": "bird"}  # plan.md's pairs
SMOKE_Z = {"airplane": "train", "bird": "dog", "dog": "cat", "car": "bus", "train": "airplane"}

N_FRAMES, SIDE, CLIP_SECONDS, SAMPLE_RATE = 8, 224, 10.0, 16000
FRAME_TIMES = [(i + 0.5) * CLIP_SECONDS / N_FRAMES for i in range(N_FRAMES)]  # 0.625, 1.875, ..., 9.375 s
MIDDLE = N_FRAMES // 2  # the frame at 5.625 s: the static video and the picture
META_FPS = 1000.0  # the frame indices handed to the processor are milliseconds, so its timestamps are the true times

INPUTS = ["clip", "static", "frame", "sound"]
MEDIUM = {"clip": "video", "static": "video", "frame": "picture", "sound": "sound"}
TASKS = ["say", "animal"]
QUESTIONS = {
    ("video", "say"): "What is the object in the video? Reply with the noun only.",
    ("video", "animal"): "Rule: if the object in the video is an animal, answer 7; otherwise answer 3. Reply with the number only.",
    ("picture", "say"): "What is the object in the picture? Reply with the noun only.",
    ("picture", "animal"): "Rule: if the object in the picture is an animal, answer 7; otherwise answer 3. Reply with the number only.",
    ("sound", "say"): "What is making this sound? Reply with the noun only.",
    ("sound", "animal"): "Rule: if the thing making this sound is an animal, answer 7; otherwise answer 3. Reply with the number only.",
}

# The words an answer may use for a class, fixed before any run: the image experiments kept only crops named with the class word;
# a clip named "a plane" names the airplane.
SYNONYMS = {
    "dog": ["dog", "puppy"],
    "cat": ["cat", "kitten"],
    "bird": ["bird"],
    "horse": ["horse", "pony"],
    "car": ["car", "automobile"],
    "bus": ["bus", "buses"],
    "train": ["train", "locomotive"],
    "airplane": ["airplane", "aeroplane", "plane", "aircraft", "airliner", "jet"],
}
WORD_CLASS = {s: w for w, ss in SYNONYMS.items() for s in ss}

# A VGGSound label belongs to a class when it names exactly one class word (as a whole word, singular or plural); a label that
# names two ("railroad car, train wagon") or none belongs to no class.
_LABEL_PATTERNS = {
    w: re.compile(rf"\b{p}\b")
    for w, p in {"dog": "dogs?", "cat": "cats?", "bird": "birds?", "horse": "horses?", "car": "cars?", "bus": "bus(es)?", "train": "trains?", "airplane": "airplanes?"}.items()
}


def label_class(label):
    """the class a VGGSound label names, or None"""
    hits = [w for w, p in _LABEL_PATTERNS.items() if p.search(label.lower())]
    return hits[0] if len(hits) == 1 else None


def truth(task, w):
    return w if task == "say" else ("7" if w in ANIMALS else "3")


def answer_class(text):
    """the class of the first word of an answer that names one (a trailing plural 's' is dropped), or None"""
    for t in re.findall(r"[a-z]+", text.lower()):
        for v in (t, t[:-1] if t.endswith("s") else t):
            if v in WORD_CLASS:
                return WORD_CLASS[v]
    return None


def answer_matches(task, text, target):
    """whether an answer gives the target: the class for say, the digit for the rule"""
    if task == "say":
        return answer_class(text) == target
    s = text.strip().strip("*").strip()
    return bool(s) and s[0] == target


def assign(clips_by_class, n_per_class):
    """The scenes: clip k of each class, ordered round-robin (scene = 8 k + class index), so the first 16 scenes, the calibration
    split, hold two clips of every class. Y is a word of the other kind and Z of the same kind, cycled over the clips of a class
    so that every ordered pair appears (each Y twice, each Z two or three times, among the 8 clips of a class)."""
    rows = []
    for k in range(n_per_class):
        for c, x in enumerate(CLASSES):
            ys = [w for w in CLASSES if (w in ANIMALS) != (x in ANIMALS)]
            zs = [w for w in CLASSES if w != x and (w in ANIMALS) == (x in ANIMALS)]
            rows.append(dict(scene=k * len(CLASSES) + c, X=x, Y=ys[k % len(ys)], Z=zs[k % len(zs)], clip=clips_by_class[x][k]))
    return rows


def assign_smoke(clips_by_class):
    """the smoke check: the first clip of each smoke class, with plan.md's pairs"""
    return [dict(scene=i, X=x, Y=SMOKE_Y[x], Z=SMOKE_Z[x], clip=clips_by_class[x][0]) for i, x in enumerate(SMOKE_CLASSES)]


def blocks(n_layers):
    """the three layer blocks as shares of depth (plan.md): early up to a third, middle to 0.7, late after; on the 32-layer 4B
    -1 to 10, 11 to 22 and 23 to 31, the blocks of RECORD row 93"""
    a, b = round(n_layers / 3), int(np.floor(0.7 * n_layers))
    return {"early": list(range(-1, a)), "middle": list(range(a, b + 1)), "late": list(range(b + 1, n_layers))}


def sweep_starts(n_layers):
    """the prefix ends and suffix starts of the sweeps: -1, 3, 7, ... (steps of 4 from the input)"""
    return list(range(-1, n_layers, 4))


def frame_metadata():
    """the frame indices and rate handed to the processor (milliseconds at 1000 fps), the same for every clip"""
    return dict(total_num_frames=int(CLIP_SECONDS * META_FPS), fps=META_FPS, frames_indices=[round(t * META_FPS) for t in FRAME_TIMES], duration=CLIP_SECONDS, width=SIDE, height=SIDE)


# ---- the clips on disk


def manifest(data_dir):
    path = Path(data_dir) / "vggsound/manifest.json"
    if not path.exists():
        raise SystemExit(f"{path} not found: run `uv run --with av==15.1.0 scripts/prepare_media.py` first")
    return json.loads(path.read_text())


def load_clip(data_dir, clip_id):
    z = np.load(Path(data_dir) / "vggsound/clips" / f"{clip_id}.npz")
    frames, audio = z["frames"], z["audio"]
    assert frames.shape == (N_FRAMES, SIDE, SIDE, 3) and frames.dtype == np.uint8, (clip_id, frames.shape, frames.dtype)
    assert audio.shape == (int(CLIP_SECONDS * SAMPLE_RATE),) and audio.dtype == np.float32, (clip_id, audio.shape, audio.dtype)
    return frames, audio


def media_of(input_kind, frames, audio):
    """what the processor receives for one input of one clip"""
    if input_kind == "clip":
        return frames
    if input_kind == "static":
        return np.repeat(frames[MIDDLE : MIDDLE + 1], N_FRAMES, axis=0)
    if input_kind == "frame":
        from PIL import Image

        return Image.fromarray(frames[MIDDLE])
    return audio
