"""Clip naming: does each model name the matched video clips when they are shown whole and asked for their COCO category?

Why. video_sweep (PR #5) failed its naming gate on every model: the moving clip, drawn into one slot of the six-slot canvas
and asked about with depth_sweep's noun question, was named on 0.58 to 0.89 of scenes. Reading the answers showed two kinds of
miss: names the exact-match scoring counts wrong (sailboat, plane, vehicle) and confusions between similar classes (a cat
called a dog). This run asks the naming question alone, with two changes made together, so it measures their joint effect and
does not separate them: each clip is shown whole, without the canvas, and the question asks for the MS COCO category name,
which names the 16 labels of the set.

Data. The 73 clips of saxenan3/jspace-matched-video at the pinned revision (src/jspace/matched.py), each with its label from
ATTRIBUTION.tsv. Three inputs per clip, as video_sweep makes them but whole: no canvas, every frame at its own size, which the
model's processor then resizes as it normally does (for Qwen, cap_pixels_per_frame=False is transformers 5.19's default):
  moving    the whole clip at the frames the model's own processor sampler picks over its full length (the sample_frames call
            the processor makes itself; Qwen3.5: 2 per second; Gemma-4: 32 spread over the clip), capped at the clip's length
            because Gemma's sampler refuses to take more frames than a clip has (matched.scene_frame_count)
  frozen    the middle sampled frame repeated as often (the same tokens, no motion)
  still     that frame as a picture
Prompt: the media, then QUESTION (below, verbatim), which lists no names. A CPU pilot (Qwen3.5-0.8B, still frames, 19 Nov
2026, before any GPU run) tried listing the 80 COCO names in the question, with and without the more specific EXTRA_NAMES:
both named fewer clips (64 and 65 of 73, against 66 with no list), because the model then picked other listed objects in
the frame ("person" for bicycles with riders, "potted plant" for cats, "scissors" for a pizza).
Greedy decoding, N_NEW new tokens. One prompt per forward: the clips differ in length, and a batch of one keeps the result
independent of how prompts would be grouped.

Scoring (written before the run). An answer is normalized: lowercased, every character other than a letter replaced by a
space, spaces collapsed and stripped. It is correct when it equals the clip's label, one of the label's EXTRA_NAMES, or the
plural of either (with "s" or "es" added), so "Airplane.", "cats", "buses", "sailboat" and "plane" are correct and "animal",
"vehicle", "coffee" or "pond bird" are not. The plurals and EXTRA_NAMES were added on 19 Nov 2026, before any GPU run,
after the CPU pilot answered "cats" and "cars" for single animals and cars, and "sailboat" for a sailboat. Also recorded:
whether the answer (its singular, or the category an extra name stands for) is one of the 80 COCO category names.

The rule (written before the run):
- Per input, the share of clips named correctly, with a 95% bootstrap interval over clips. The gate passes for an input when
  that share is >= 0.90, video_sweep's naming gate.
- A clip is reliable for a model when it is named correctly on all three inputs.
- The gated schedule for the model: the scenes of the 64-scene schedule whose receiver and donor clips are both reliable. The
  schedule reliable for every model is the intersection of the per-model lists; a video_sweep rerun on it is a new experiment
  with its own run name and rule.

Outputs in runs/<run>/<model>/: clip_naming.csv (every answer), summary.json (shares, gates, reliable clips, the gated
schedule, setup), clips.json (per clip the frame count and sampled indices), stimuli.png, config.json, run.log. stimuli.png
shows the first clip's first, middle and last sampled frames.
"""

import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from video_sweep import encode  # noqa: E402  the media and prompt go through the processor exactly as video_sweep sends them

from jspace import matched, stimuli  # noqa: E402
from jspace.analysis import boot_scene  # noqa: E402
from jspace.config import require_data  # noqa: E402
from jspace.model import Session  # noqa: E402
from jspace.runs import Run, parser  # noqa: E402

N_NEW = 24  # room for a two-word COCO name and anything the model adds; the scoring still wants the bare name
GATE = 0.90
COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife",
    "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]  # fmt: skip
# more specific names scored as their COCO category: kinds and spellings of the category, never a generic word ("animal",
# "vehicle") or a different object ("coffee" for a cup)
EXTRA_NAMES = {
    "plane": "airplane", "aeroplane": "airplane", "jet": "airplane", "sailboat": "boat", "ship": "boat", "yacht": "boat",
    "kitten": "cat", "puppy": "dog", "pony": "horse", "mug": "cup", "bike": "bicycle", "motorbike": "motorcycle",
    "locomotive": "train",
}  # fmt: skip
QUESTION = "What category best describes the main object shown? Identify the main object using its standard MS COCO category name. Return only the category name, nothing else."


def normalize(text):
    return " ".join(re.sub(r"[^a-z]", " ", text.lower()).split())


def forms(a):
    """the normalized answer and its possible singulars: "cats" -> cat, "buses" -> bus (a plural is not a different object)"""
    out = {a}
    if a.endswith("es"):
        out.add(a[:-2])
    if a.endswith("s"):
        out.add(a[:-1])
    return out


def score(text, label):
    """(correct, a COCO name): the answer is the label, one of its EXTRA_NAMES, or a plural of either; the answer (or its
    singular, or the category an extra name stands for) is one of the 80 COCO names"""
    f = forms(normalize(text))
    f |= {EXTRA_NAMES[w] for w in f if w in EXTRA_NAMES}
    return label in f, bool(f & set(COCO_NAMES))


def prompt(lm, inp):
    part = {"type": "video"} if matched.MEDIUM[inp] == "video" else {"type": "image"}
    return lm.chat_messages([{"role": "user", "content": [part, {"type": "text", "text": QUESTION}]}])


def summarize(R, schedule):
    """the rule above on the answers R (one row per clip and input) and the 64-scene schedule [(scene, receiver, donor)]"""
    clips = sorted(R["clip"].unique())
    out = dict(n_clips=len(clips), share={}, gate={})
    for inp in matched.INPUTS:
        d = R[R["input"] == inp].set_index("clip").reindex(clips)
        assert not d["correct"].isna().any(), f"missing answers for {inp}"
        out["share"][inp] = boot_scene(d["correct"].astype(float).to_numpy(), np.arange(len(clips)))
        out["gate"][inp] = bool(out["share"][inp][0] >= GATE)
        out.setdefault("coco_name", {})[inp] = float(d["coco"].astype(float).mean())
    reliable = R.groupby("clip")["correct"].all()
    out["reliable"] = sorted(c for c, ok in reliable.items() if ok)
    out["unreliable"] = sorted(c for c, ok in reliable.items() if not ok)
    rel = set(out["reliable"])
    out["gated_scenes"] = [s for s, rc, dc in schedule if rc in rel and dc in rel]
    return out


def main():
    ap = parser(__doc__, "clip_naming")
    args = ap.parse_args()
    require_data(args.data_dir, "matched")
    run = Run(args, "clip_naming")
    lm = Session(args.model)
    run.write_config(lm)
    t0 = time.time()
    root = args.data_dir / matched.DIR / "test"
    A = matched.attribution(args.data_dir)
    schedule = [(m["scene_id"], m["receiver_clip_id"], m["donor_clip_id"]) for m in matched.scenes(args.data_dir)]
    vp = lm.processor.video_processor

    rows, clips, sheet = [], {}, []
    for cid in sorted(A):
        path = root / Path(A[cid]["path"]).name
        total, fps = matched.video_info(path)
        n = matched.scene_frame_count(vp, total, total, fps)
        idx = matched.native_indices(vp, total, fps, n)
        frames = np.stack([np.asarray(f) for f in matched.decode(path, idx)])
        md = dict(total_num_frames=total, fps=fps, frames_indices=idx, duration=total / fps, width=frames.shape[2], height=frames.shape[1])
        clips[cid] = dict(label=A[cid]["label"], frames=n, indices=idx, total=total)
        for inp in matched.INPUTS:
            obj = matched.input_of(frames, inp)
            enc = encode(lm, inp, prompt(lm, inp), obj, md)
            kw = {k: (v.to(lm.device) if torch.is_tensor(v) else v) for k, v in enc.items() if k != "token_type_ids"}
            with torch.no_grad():
                g = lm.hf.generate(**kw, max_new_tokens=N_NEW, do_sample=False, pad_token_id=lm.tokenizer.pad_token_id)
            text = lm.tokenizer.decode(g[0, kw["input_ids"].shape[1] :], skip_special_tokens=True).strip()
            correct, coco = score(text, A[cid]["label"])
            rows.append(dict(clip=cid, label=A[cid]["label"], input=inp, frames=n, tokens=int(kw["input_ids"].shape[1]), answer=text, normalized=normalize(text), correct=correct, coco=coco))
            if not sheet:  # the first clip's first, middle and last sampled frames, at the size the processor receives them
                sheet = [(f"{cid} sampled frame {j}", Image.fromarray(frames[j])) for j in (0, len(frames) // 2, len(frames) - 1)]
        print(f"{cid:34s} {A[cid]['label']:8s} " + " | ".join(f"{r['input']}: {r['answer'][:16]!r}{'' if r['correct'] else ' x'}" for r in rows[-3:]) + f"  [{time.time() - t0:.0f}s]")

    R = pd.DataFrame(rows)
    run.save_csv(R, "clip_naming.csv")
    run.save_json(dict(clips=clips, sampler=type(vp).__name__), "clips.json")
    stimuli.contact_sheet(sheet, 3, run.path("stimuli.png"))
    out = summarize(R, schedule)
    out["setup"] = dict(question=QUESTION, n_new=N_NEW, sampler=type(vp).__name__, seconds=time.time() - t0)
    run.save_json(out, "summary.json")
    for inp in matched.INPUTS:
        print(f"{inp}: named {out['share'][inp][0]:.3f} [{out['share'][inp][1]:.3f}, {out['share'][inp][2]:.3f}], gate {'passes' if out['gate'][inp] else 'fails'}")
    print(f"reliable clips {len(out['reliable'])} of {out['n_clips']}; gated scenes {len(out['gated_scenes'])} of {len(schedule)}")


if __name__ == "__main__":
    main()
