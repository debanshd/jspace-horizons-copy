"""Clip naming 3: clip_naming2 on all four models, with a video counted as reliable when its moving clip is named, and the first
name of a listed answer scored.

Why. The naming gate decides which scene pairs video_sweep2 can use, and on the two smallest models it left few: a pair needs
both its videos reliable, and clip_naming counted a video reliable only when the model named it on all three inputs (moving,
frozen, still), so Qwen3.5-0.8B kept 31 of the 64 pairs and Gemma-4-E2B 43 (clip_naming2). Reading those answers showed two
things a stricter rule than needed was throwing away: videos named on the moving clip but missed as a frozen frame or a
picture, and answers that start with the right name and then list others ("Horse, cow, horse, hor..."). Both changes below were
chosen after reading those answers, so this is a new experiment with its own run name; clip_naming's and clip_naming2's results
stand as they are. Re-scored with this rule, PR #7's own moving answers would have kept 54 pairs for Qwen3.5-0.8B (from 31),
59 for Qwen3.5-4B (57), 50 for Gemma-4-E2B (43) and 57 for Gemma-4-E4B (57); those re-scored answers came from clip_naming's
question for the Qwen models, while this run asks every model clip_naming2's.

Everything else is clip_naming2's: the 73 clips of the matched set at the pinned revision, each shown whole (moving, frozen,
still: the model's own processor sampler over the full clip, its normal resizing), clip_naming2's question (verbatim, for every
model), greedy decoding with 24 new tokens, one prompt per forward, the next clip prepared on a worker thread.

Scoring (written before the run). The answer's first name is scored with clip_naming2's scoring: the first line, cut at the
first list mark (a comma, semicolon, slash or bar, or the word "or" or "and" standing on its own, not inside a hyphenated
word such as "black-and-white"); then clip_naming2's rule on what is left (the label, its head noun in a phrase, its plural,
or a kind or spelling of it), with a hyphenated word holding "and" or "or" read as one word, so clip_naming2's list check does
not take it for a list. So "Horse, cow, horse" counts as horse, "Black-and-white cat" as cat, and "animal", "food" or "dog"
for a horse stay wrong.

The rule (written before the run):
- Per input, the share of clips named correctly with a 95% bootstrap interval over clips; the gate passes at >= 0.90.
- A clip is reliable for an input when it is named on that input. The gated schedule for an input is the scenes whose receiver
  and donor are both reliable for it (gated_by_input).
- The gated schedule that video_sweep2 reads (gated_scenes) is the moving input's: the moving clip is what the patching is
  about, and video_sweep2's own clean gate measures the frozen and still inputs on those scenes again.
- Also reported, for comparison with clip_naming: the clips named on all three inputs and the scenes they leave (reliable_all,
  gated_all).

Outputs in runs/<run>/<model>/: clip_naming3.csv (every answer, with the first name scored), summary.json, clips.json,
stimuli.png, gpu.csv, config.json, run.log.
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
import clip_naming as base  # noqa: E402  the summary's shares and gates
import clip_naming2 as n2  # noqa: E402  the question and the scoring of one name
from video_sweep import encode  # noqa: E402

from jspace import batched, matched, stimuli  # noqa: E402
from jspace.config import require_data  # noqa: E402
from jspace.model import Session  # noqa: E402
from jspace.runs import Run, parser  # noqa: E402

QUESTION = n2.QUESTION
N_NEW = n2.N_NEW
GATED_INPUT = "moving"
# "or" and "and" mark a list only as words of their own: a hyphen on either side makes them part of a word ("black-and-white")
LIST_SPLIT = re.compile(r"[,;/|]|(?<![\w-])(?:or|and)(?![\w-])", re.IGNORECASE)
# a hyphenated word; one with "and" or "or" among its parts is joined, since clip_naming2 would read those parts as a list
HYPHENATED = re.compile(r"\w+(?:-\w+)+")


def first_name(text):
    """the answer's first name: its first line, cut at the first list mark"""
    lines = text.strip().splitlines()
    return LIST_SPLIT.split(lines[0], maxsplit=1)[0].strip() if lines else ""


def score(text, label):
    """(correct, a COCO name) for the answer's first name, by clip_naming2's scoring; the list is already cut off, so a
    hyphenated "and" or "or" left in the name is joined into its word ("black-and-white cat" -> "blackandwhite cat")"""

    def join(m):
        w = m.group(0)
        return w.replace("-", "") if {p.lower() for p in w.split("-")} & {"and", "or"} else w

    return n2.score(HYPHENATED.sub(join, first_name(text)), label)


def summarize(R, schedule):
    """the rule above on the answers R (one row per clip and input) and the 64-scene schedule [(scene, receiver, donor)]"""
    out = base.summarize(R, schedule)  # shares and gates per input; its reliable / gated_scenes are the all-three-inputs rule
    out["reliable_all"], out["gated_all"] = out.pop("reliable"), out.pop("gated_scenes")
    out.pop("unreliable")
    out["reliable_by_input"], out["gated_by_input"] = {}, {}
    for inp in matched.INPUTS:
        ok = R[R["input"] == inp].set_index("clip")["correct"]
        rel = {c for c, v in ok.items() if v}
        out["reliable_by_input"][inp] = sorted(rel)
        out["gated_by_input"][inp] = [s for s, rc, dc in schedule if rc in rel and dc in rel]
    out["gated_input"] = GATED_INPUT
    out["reliable"] = out["reliable_by_input"][GATED_INPUT]
    out["gated_scenes"] = out["gated_by_input"][GATED_INPUT]
    return out


def prompt(lm, inp):
    part = {"type": "video"} if matched.MEDIUM[inp] == "video" else {"type": "image"}
    return lm.chat_messages([{"role": "user", "content": [part, {"type": "text", "text": QUESTION}]}])


def main():
    ap = parser(__doc__, "clip_naming3")
    args = ap.parse_args()
    require_data(args.data_dir, "matched")
    run = Run(args, "clip_naming3")
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
                rows.append(
                    dict(
                        clip=cid,
                        label=A[cid]["label"],
                        input=inp,
                        frames=n,
                        tokens=int(kw["input_ids"].shape[1]),
                        answer=text,
                        first_name=first_name(text),
                        normalized=base.normalize(text),
                        correct=correct,
                        coco=coco,
                    )
                )
                if not sheet:  # the first clip's first, middle and last sampled frames, at the size the processor receives them
                    sheet = [(f"{cid} sampled frame {j}", Image.fromarray(frames[j])) for j in (0, len(frames) // 2, len(frames) - 1)]
            print(f"{cid:34s} {A[cid]['label']:8s} " + " | ".join(f"{r['input']}: {r['answer'][:16]!r}{'' if r['correct'] else ' x'}" for r in rows[-3:]) + f"  [{time.time() - t0:.0f}s]")
    gpu = mon.stop()

    R = pd.DataFrame(rows)
    run.save_csv(R, "clip_naming3.csv")
    run.save_json(dict(clips=clips, sampler=type(vp).__name__), "clips.json")
    stimuli.contact_sheet(sheet, 3, run.path("stimuli.png"))
    out = summarize(R, schedule)
    out["setup"] = dict(question=QUESTION, n_new=N_NEW, sampler=type(vp).__name__, gpu=gpu, seconds=time.time() - t0)
    run.save_json(out, "summary.json")
    for inp in matched.INPUTS:
        print(f"{inp}: named {out['share'][inp][0]:.3f}, gate {'passes' if out['gate'][inp] else 'fails'}, {len(out['gated_by_input'][inp])} scene pairs")
    print(f"gated schedule ({GATED_INPUT}): {len(out['gated_scenes'])} of {len(schedule)}; all three inputs: {len(out['gated_all'])}")


if __name__ == "__main__":
    main()
