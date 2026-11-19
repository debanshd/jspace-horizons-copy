"""Clip naming 2: clip_naming with a question that rules out general words and a scoring that reads a phrase's head noun.

Why. clip_naming (PR #6, results in PR #7) failed its 0.90 gate on both Gemma models, in two different ways read from the
answers: Gemma-4-E4B named the right object too specifically ("race car", "sports car", "coffee cup", "office chair"), and
Gemma-4-E2B too generally ("animal", "food" for pizza, "furniture"). Both fixes below were chosen after reading those
answers, so this is a new experiment with its own run name; clip_naming's results stand as they are. Re-scored with the new
scoring, PR #7's own answers would have given Gemma-4-E4B 69, 69 and 70 of 73 (moving, frozen, still) and Gemma-4-E2B 67, 61
and 68: the scoring is aimed at E4B, the question at E2B.

Everything else is clip_naming's: the 73 clips of the matched set at the pinned revision, each shown whole (moving, frozen,
still; the model's own processor sampler over the full clip, its normal resizing), greedy decoding with N_NEW = 24 new tokens,
one prompt per forward, the next clip prepared on a worker thread.

Question (verbatim, QUESTION below): clip_naming's, with one sentence added before the last: "Name the object itself, not a
general word such as animal, food, vehicle or furniture." The four general words are none of the 16 labels, so the sentence
names no answer.

Scoring (written before the run). The answer's first line is normalized as in clip_naming (lowercased, non-letters to spaces,
spaces collapsed). Its candidate names are the whole answer and, unless the whole answer is itself a COCO name (so "hot dog"
stays a hot dog, not a dog) or the line holds a list (a comma, semicolon, slash or bar, or the word "or" or "and"), its last
word and its last two words: the head noun of a phrase ("race car" -> car, "coffee cup" -> cup, "pond bird" -> bird). Each
candidate also counts as its singular (an "s" or "es" removed) and, through EXTRA_NAMES2, as the COCO category it is a kind or
spelling of (clip_naming's EXTRA_NAMES plus aircraft and airliner for airplane, automobile for car, and duck, chicken, goose,
pigeon and gull for bird). The answer is correct when one of these is the clip's label. "Subway car" for a train, "animal",
"food", "vehicle" and "dog" for a horse stay wrong.

The rule (written before the run) is clip_naming's: per input, the share of clips named correctly with a 95% bootstrap
interval over clips, and the gate passes at >= 0.90; a clip is reliable for a model when it is named correctly on all three
inputs; the gated schedule is the scenes whose receiver and donor are both reliable. Run first on the two Gemma models (the
ones that failed); a Qwen model run under this experiment is read against its own clip_naming run, not against Gemma's.

Outputs in runs/<run>/<model>/: clip_naming2.csv (every answer), summary.json, clips.json, stimuli.png, gpu.csv, config.json,
run.log, as clip_naming writes them.
"""

import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import clip_naming as base  # noqa: E402  the clips, inputs, summary and rule are clip_naming's
from video_sweep import encode  # noqa: E402

from jspace import batched, matched, stimuli  # noqa: E402
from jspace.config import require_data  # noqa: E402
from jspace.model import Session  # noqa: E402
from jspace.runs import Run, parser  # noqa: E402

QUESTION = (
    "What category best describes the main object shown? Identify the main object using its standard MS COCO category name. "
    "Name the object itself, not a general word such as animal, food, vehicle or furniture. "
    "Return only the category name, nothing else."
)
N_NEW = base.N_NEW
# clip_naming's kinds and spellings, and a few more of the same sort: never a generic word or a different object
EXTRA_NAMES2 = {
    **base.EXTRA_NAMES,
    "aircraft": "airplane", "airliner": "airplane", "automobile": "car",
    "duck": "bird", "chicken": "bird", "goose": "bird", "pigeon": "bird", "gull": "bird",
}  # fmt: skip
LIST_MARK = re.compile(r"[,;/|]")


def candidates(text):
    """the names an answer can stand for: the whole first line, and its head noun unless it is a COCO name or a list"""
    lines = text.strip().splitlines()
    first = lines[0] if lines else ""
    a = base.normalize(first)
    words = a.split()
    out = {a}
    is_list = bool(LIST_MARK.search(first)) or "or" in words or "and" in words
    whole_is_name = bool(base.forms(a) & set(base.COCO_NAMES))
    if words and not is_list and not whole_is_name:
        out.add(words[-1])
        if len(words) >= 2:
            out.add(" ".join(words[-2:]))  # a two-word COCO name at the end of a phrase ("a wooden dining table")
    return out


def score(text, label):
    """(correct, a COCO name): one of the answer's candidate names, as itself, its singular or the category it is a kind of, is
    the label; or is one of the 80 COCO names"""
    f = set().union(*(base.forms(c) for c in candidates(text)))
    f |= {EXTRA_NAMES2[w] for w in f if w in EXTRA_NAMES2}
    return label in f, bool(f & set(base.COCO_NAMES))


def prompt(lm, inp):
    part = {"type": "video"} if matched.MEDIUM[inp] == "video" else {"type": "image"}
    return lm.chat_messages([{"role": "user", "content": [part, {"type": "text", "text": QUESTION}]}])


def main():
    ap = parser(__doc__, "clip_naming2")
    args = ap.parse_args()
    require_data(args.data_dir, "matched")
    run = Run(args, "clip_naming2")
    lm = Session(args.model)
    run.write_config(lm)
    t0 = time.time()
    root = args.data_dir / matched.DIR / "test"
    A = matched.attribution(args.data_dir)
    schedule = [(m["scene_id"], m["receiver_clip_id"], m["donor_clip_id"]) for m in matched.scenes(args.data_dir)]
    vp = lm.processor.video_processor

    def prepare(cid):
        """the clip's frames and its three encoded prompts, on the CPU (run on the worker thread)"""
        path = root / Path(A[cid]["path"]).name
        total, fps = matched.video_info(path)
        n = matched.scene_frame_count(vp, total, total, fps)
        idx = matched.native_indices(vp, total, fps, n)
        frames = np.stack([np.asarray(f) for f in matched.decode(path, idx)])
        md = dict(total_num_frames=total, fps=fps, frames_indices=idx, duration=total / fps, width=frames.shape[2], height=frames.shape[1])
        encs = {inp: encode(lm, inp, prompt(lm, inp), matched.input_of(frames, inp), md) for inp in matched.INPUTS}
        return dict(n=n, idx=idx, total=total, frames=frames, encs=encs)

    mon = batched.GpuMonitor(run.path("gpu.csv"))
    rows, clips, sheet = [], {}, []
    order = sorted(A)
    with ThreadPoolExecutor(max_workers=1) as pool:
        nxt = pool.submit(prepare, order[0])
        for k, cid in enumerate(order):
            c = nxt.result()
            if k + 1 < len(order):  # the next clip is decoded while this one runs on the GPU
                nxt = pool.submit(prepare, order[k + 1])
            n, frames = c["n"], c["frames"]
            clips[cid] = dict(label=A[cid]["label"], frames=n, indices=c["idx"], total=c["total"])
            for inp in matched.INPUTS:
                kw = {k2: (v.to(lm.device) if torch.is_tensor(v) else v) for k2, v in c["encs"][inp].items() if k2 != "token_type_ids"}
                with torch.no_grad():
                    g = lm.hf.generate(**kw, max_new_tokens=N_NEW, do_sample=False, pad_token_id=lm.tokenizer.pad_token_id)
                text = lm.tokenizer.decode(g[0, kw["input_ids"].shape[1] :], skip_special_tokens=True).strip()
                correct, coco = score(text, A[cid]["label"])
                rows.append(dict(clip=cid, label=A[cid]["label"], input=inp, frames=n, tokens=int(kw["input_ids"].shape[1]), answer=text, normalized=base.normalize(text), correct=correct, coco=coco))
                if not sheet:  # the first clip's first, middle and last sampled frames, at the size the processor receives them
                    sheet = [(f"{cid} sampled frame {j}", Image.fromarray(frames[j])) for j in (0, len(frames) // 2, len(frames) - 1)]
            print(f"{cid:34s} {A[cid]['label']:8s} " + " | ".join(f"{r['input']}: {r['answer'][:16]!r}{'' if r['correct'] else ' x'}" for r in rows[-3:]) + f"  [{time.time() - t0:.0f}s]")
    gpu = mon.stop()

    R = pd.DataFrame(rows)
    run.save_csv(R, "clip_naming2.csv")
    run.save_json(dict(clips=clips, sampler=type(vp).__name__), "clips.json")
    stimuli.contact_sheet(sheet, 3, run.path("stimuli.png"))
    out = base.summarize(R, schedule)
    out["setup"] = dict(question=QUESTION, n_new=N_NEW, sampler=type(vp).__name__, gpu=gpu, seconds=time.time() - t0)
    run.save_json(out, "summary.json")
    for inp in matched.INPUTS:
        print(f"{inp}: named {out['share'][inp][0]:.3f} [{out['share'][inp][1]:.3f}, {out['share'][inp][2]:.3f}], gate {'passes' if out['gate'][inp] else 'fails'}")
    print(f"reliable clips {len(out['reliable'])} of {out['n_clips']}; gated scenes {len(out['gated_scenes'])} of {len(schedule)}")


if __name__ == "__main__":
    main()
