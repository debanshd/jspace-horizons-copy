"""Clip naming 4: clip_naming3 with clip_naming's original question, for Qwen3.5-0.8B.

Why. clip_naming3 (PR #6, results in PR #7) asked every model clip_naming2's question, which adds "Name the object itself, not
a general word such as animal, food, vehicle or furniture." Qwen3.5-0.8B answered with exactly those words far more often than
under clip_naming's question (on the moving clip 12 of 73 answers, against 2; on the still picture 31, against 3) and kept 42
of the 64 scene pairs. This run asks it clip_naming's question instead, with everything else clip_naming3's: the 73 clips
shown whole (moving, frozen, still), greedy decoding with 24 new tokens, the hedge follow-up (FOLLOWUP, the same text), the
scoring (one name scored with clip_naming2's rule; two or more different names a hedge, asked once more, only the single reply
scored, a second hedge wrong; a cut-off last fragment dropped only when the answer hit the token limit) and the rule (per-input
shares with their 0.90 gate; a video reliable for an input when named on it; the gated schedule video_sweep2 reads is the
moving input's). Choosing the question after reading clip_naming3's answers makes this a new experiment with its own run
name; clip_naming3's results stand. Re-scored with clip_naming3's rule, Qwen3.5-0.8B's clip_naming answers (this question, a
run without the follow-up) keep at least 52 moving pairs with every hedge counted wrong.

Question (verbatim, QUESTION below): clip_naming's. The pod script runs Qwen3.5-0.8B only; another model run under this
experiment is read against its own clip_naming3 run.

Outputs in runs/<run>/<model>/: clip_naming4.csv (every answer, whether it hedged, the follow-up reply, the name scored),
summary.json, clips.json, stimuli.png, gpu.csv, config.json, run.log, as clip_naming3 writes them.
"""

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import clip_naming as base  # noqa: E402  the question
from clip_naming3 import FOLLOWUP, GATED_INPUT, N_NEW, hedged, names, score, summarize  # noqa: E402  everything else
from video_sweep import encode  # noqa: E402

from jspace import batched, matched, stimuli  # noqa: E402
from jspace.config import require_data  # noqa: E402
from jspace.model import Session  # noqa: E402
from jspace.runs import Run, parser  # noqa: E402

QUESTION = base.QUESTION


def prompt(lm, inp, first_answer=None):
    """the question about the media; with first_answer, the same conversation continued by the follow-up question"""
    part = {"type": "video"} if matched.MEDIUM[inp] == "video" else {"type": "image"}
    msgs = [{"role": "user", "content": [part, {"type": "text", "text": QUESTION}]}]
    if first_answer is not None:
        msgs += [{"role": "assistant", "content": [{"type": "text", "text": first_answer}]}, {"role": "user", "content": [{"type": "text", "text": FOLLOWUP}]}]
    return lm.chat_messages(msgs)


def main():
    ap = parser(__doc__, "clip_naming4")
    args = ap.parse_args()
    require_data(args.data_dir, "matched")
    run = Run(args, "clip_naming4")
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
        objs = {inp: matched.input_of(frames, inp) for inp in matched.INPUTS}
        encs = {inp: encode(lm, inp, prompt(lm, inp), objs[inp], md) for inp in matched.INPUTS}
        return dict(n=n, idx=idx, total=total, frames=frames, encs=encs, objs=objs, md=md)

    eos = lm.hf.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, (list, tuple)) else [eos])

    def answer(enc):
        """(the reply, the prompt's length in tokens, whether the reply was cut off at N_NEW tokens rather than ended)"""
        kw = {k2: (v.to(lm.device) if torch.is_tensor(v) else v) for k2, v in enc.items() if k2 != "token_type_ids"}
        with torch.no_grad():
            g = lm.hf.generate(**kw, max_new_tokens=N_NEW, do_sample=False, pad_token_id=lm.tokenizer.pad_token_id)
        new = g[0, kw["input_ids"].shape[1] :]
        cut = len(new) >= N_NEW and int(new[-1]) not in eos
        return lm.tokenizer.decode(new, skip_special_tokens=True).strip(), int(kw["input_ids"].shape[1]), bool(cut)

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
                text, ntok, cut = answer(c["encs"][inp])
                hedge, reply, reply_cut = hedged(text, cut), "", False
                if hedge:  # one more question in the same conversation; only the single name it gives can count
                    reply, _, reply_cut = answer(encode(lm, inp, prompt(lm, inp, text), c["objs"][inp], c["md"]))
                scored, scored_cut = (reply, reply_cut) if hedge else (text, cut)
                correct, coco = score(scored, A[cid]["label"], scored_cut)
                rows.append(
                    dict(
                        clip=cid,
                        label=A[cid]["label"],
                        input=inp,
                        frames=n,
                        tokens=ntok,
                        answer=text,
                        cut=cut,
                        hedged=hedge,
                        followup=reply,
                        followup_cut=reply_cut,
                        scored=" | ".join(names(scored, scored_cut)),
                        normalized=base.normalize(text),
                        correct=correct,
                        coco=coco,
                    )
                )
                if not sheet:  # the first clip's first, middle and last sampled frames, at the size the processor receives them
                    sheet = [(f"{cid} sampled frame {j}", Image.fromarray(frames[j])) for j in (0, len(frames) // 2, len(frames) - 1)]
            print(
                f"{cid:34s} {A[cid]['label']:8s} "
                + " | ".join(f"{r['input']}: {r['answer'][:16]!r}{' -> ' + repr(r['followup'][:12]) if r['hedged'] else ''}{'' if r['correct'] else ' x'}" for r in rows[-3:])
                + f"  [{time.time() - t0:.0f}s]"
            )
    gpu = mon.stop()

    R = pd.DataFrame(rows)
    run.save_csv(R, "clip_naming4.csv")
    run.save_json(dict(clips=clips, sampler=type(vp).__name__), "clips.json")
    stimuli.contact_sheet(sheet, 3, run.path("stimuli.png"))
    out = summarize(R, schedule)
    out["setup"] = dict(question=QUESTION, followup=FOLLOWUP, n_new=N_NEW, sampler=type(vp).__name__, gpu=gpu, seconds=time.time() - t0)
    run.save_json(out, "summary.json")
    for inp in matched.INPUTS:
        print(f"{inp}: named {out['share'][inp][0]:.3f}, gate {'passes' if out['gate'][inp] else 'fails'}, {len(out['gated_by_input'][inp])} scene pairs")
    print(f"gated schedule ({GATED_INPUT}): {len(out['gated_scenes'])} of {len(schedule)}; all three inputs: {len(out['gated_all'])}")


if __name__ == "__main__":
    main()
