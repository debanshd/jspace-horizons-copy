"""Audio naming: does each model name the spoken object names of the matched audio set, in either prompt order?

Why. Every patching result depends on the model naming both recordings of a scene on its own first (video_sweep2 needed its
naming gate for the same reason). This run asks the naming question alone, so the audio sweep can run on the scenes whose two
recordings the model names, and it measures the controls the dataset asks for before any patching: silence, noise, and a typed
transcript of the word.

Data. The 15 recordings of saxenan3/jspace-matched-spoken-audio at the pinned revision (src/jspace/spoken.py): one synthetic
voice saying one of 15 object names, each a 3-second 16 kHz mono WAV, fed to the model's own processor unchanged. Inputs:
  q_audio   the question, then the audio (Gemma-4's card's order)
  audio_q   the audio, then the question (the video experiments' order)
  typed     the question, then the word as text ("Transcript of the audio: <word>"): the text baseline the dataset asks for
and two controls in both audio orders: 3 s of silence and 3 s of white noise (seed 0), which name no object.
Question: the dataset's say question (spoken.QUESTIONS["say"], verbatim). Greedy decoding with 24 new tokens; one prompt per
forward.

Scoring (written before the run): clip_naming3's. An answer that names one thing is scored with clip_naming2's rule (the
label, its head noun in a phrase, its plural, or a kind or spelling of it); an answer naming two or more different things is a
hedge, asked once more in the same conversation (clip_naming3's FOLLOWUP), and only the single reply is scored; a second
hedge is wrong; a cut-off last fragment is dropped only when the answer hit the token limit. For a control, the answer
"names an object" when it scores as any of the 15 labels.

The rule (written before the run):
- Per input, the share of the 15 recordings named, with a 95% bootstrap interval over recordings, and the share of the 64
  scene receivers named (recordings recur, so this weights them by use). The gate passes at >= 0.90 of recordings.
- A recording is reliable for an audio order when it is named in that order. gated_by_input: per order, the scenes whose
  receiver and donor recordings are both reliable. gated_scenes, the schedule audio_sweep reads: the scenes reliable in both
  orders, so the sweep compares the orders on the same scenes.
- Controls: per audio order, whether the model names an object for silence and for noise (reported, not gated).
Run on the audio-capable models (Gemma-4-E2B and E4B); a model without an audio tower stops the run.

Outputs in runs/<run>/<model>/: audio_naming.csv (every answer), summary.json, config.json, run.log.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import clip_naming as base  # noqa: E402  normalize
import clip_naming3 as c3  # noqa: E402  the scoring and the follow-up question
from media_steer import has_audio  # noqa: E402

from jspace import batched, spoken  # noqa: E402
from jspace.analysis import boot_scene  # noqa: E402
from jspace.config import require_data  # noqa: E402
from jspace.model import Session  # noqa: E402
from jspace.runs import Run, parser  # noqa: E402

QUESTION = spoken.QUESTIONS["say"]
N_NEW = 24
GATE = 0.90
INPUTS = spoken.INPUTS + ["typed"]
CONTROLS = {"silence": spoken.silence, "noise": spoken.noise}


def messages(inp, word=None, first_answer=None):
    """one user turn in the input's order (the typed input: the question, then the transcript); with first_answer, the same
    conversation continued by clip_naming3's follow-up question"""
    if inp == "typed":
        parts = [{"type": "text", "text": QUESTION}, {"type": "text", "text": f"Transcript of the audio: {word}"}]
    else:
        parts = spoken.content(inp, QUESTION)
    msgs = [{"role": "user", "content": parts}]
    if first_answer is not None:
        msgs += [{"role": "assistant", "content": [{"type": "text", "text": first_answer}]}, {"role": "user", "content": [{"type": "text", "text": c3.FOLLOWUP}]}]
    return msgs


def names_object(answer, cut, labels):
    """does a control's answer score as any of the labels"""
    return any(c3.score(answer, w, cut)[0] for w in labels)


def summarize(R, schedule, recs):
    """the rule above on the answers R and the scene schedule [(scene, receiver audio id, donor audio id)]"""
    out = dict(n_recordings=len(recs), share={}, share_by_scene={}, gate={}, reliable_by_input={}, gated_by_input={}, hedged={}, controls={})
    for inp in INPUTS:
        d = R[(R["input"] == inp) & (R["item"].isin(recs))].set_index("item").reindex(sorted(recs))
        assert not d["correct"].isna().any(), f"missing answers for {inp}"
        out["share"][inp] = boot_scene(d["correct"].astype(float).to_numpy(), np.arange(len(d)))
        out["gate"][inp] = bool(out["share"][inp][0] >= GATE)
        out["share_by_scene"][inp] = float(np.mean([d.loc[rc, "correct"] for _, rc, _ in schedule]))
        out["hedged"][inp] = float(d["hedged"].astype(float).mean())
        rel = {a for a, v in d["correct"].items() if v}
        out["reliable_by_input"][inp] = sorted(rel)
        if inp in spoken.INPUTS:
            out["gated_by_input"][inp] = [s for s, rc, dc in schedule if rc in rel and dc in rel]
            c = R[(R["input"] == inp) & (R["item"].isin(CONTROLS))].set_index("item")
            out["controls"][inp] = {k: dict(answer=str(c.loc[k, "answer"]), names_object=bool(c.loc[k, "names_object"])) for k in CONTROLS}
    both = set(out["gated_by_input"]["q_audio"]) & set(out["gated_by_input"]["audio_q"])
    out["gated_scenes"] = [s for s, _, _ in schedule if s in both]
    return out


def main():
    ap = parser(__doc__, "audio_naming")
    args = ap.parse_args()
    require_data(args.data_dir, "spoken")
    run = Run(args, "audio_naming")
    lm = Session(args.model)
    if not has_audio(lm):
        raise SystemExit(f"{lm.name} takes no audio")
    run.write_config(lm)
    t0 = time.time()
    root = args.data_dir / spoken.DIR
    S = spoken.scenes(args.data_dir)
    recs = spoken.recordings(S)
    schedule = [(r["scene_id"], r["receiver_audio_id"], r["donor_audio_id"]) for r in S]
    labels = sorted({lab for lab, _, _ in recs.values()})
    items = {a: (lab, spoken.load_wav(root / rel)) for a, (lab, rel, _) in sorted(recs.items())}
    items.update({k: (None, f()) for k, f in CONTROLS.items()})
    eos = lm.hf.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, (list, tuple)) else [eos])

    def answer(msgs, wav):
        text = lm.chat_messages(msgs)
        enc = lm.processor(text=[text], audio=[wav], return_tensors="pt") if wav is not None else lm.processor(text=[text], return_tensors="pt")
        kw = {k: (v.to(lm.device).to(lm.dtype) if k in batched.FLOAT_KEYS else v.to(lm.device)) if torch.is_tensor(v) else v for k, v in enc.items() if k != "token_type_ids"}
        with torch.no_grad():
            g = lm.hf.generate(**kw, max_new_tokens=N_NEW, do_sample=False, pad_token_id=lm.tokenizer.pad_token_id)
        new = g[0, kw["input_ids"].shape[1] :]
        cut = len(new) >= N_NEW and int(new[-1]) not in eos
        return lm.tokenizer.decode(new, skip_special_tokens=True).strip(), int(kw["input_ids"].shape[1]), bool(cut)

    rows = []
    for item, (label, wav) in items.items():
        for inp in INPUTS:
            if inp == "typed" and label is None:
                continue  # the controls have no word to type
            audio = None if inp == "typed" else wav
            text, ntok, cut = answer(messages(inp, label), audio)
            hedge, reply, reply_cut = c3.hedged(text, cut), "", False
            if hedge:  # one more question in the same conversation; only the single name it gives can count
                reply, _, reply_cut = answer(messages(inp, label, text), audio)
            scored, scored_cut = (reply, reply_cut) if hedge else (text, cut)
            correct = c3.score(scored, label, scored_cut)[0] if label else False
            rows.append(
                dict(
                    item=item,
                    label=label or "",
                    input=inp,
                    tokens=ntok,
                    answer=text,
                    cut=cut,
                    hedged=hedge,
                    followup=reply,
                    followup_cut=reply_cut,
                    scored=" | ".join(c3.names(scored, scored_cut)),
                    normalized=base.normalize(text),
                    correct=correct,
                    names_object=names_object(scored, scored_cut, labels),
                )
            )
        print(
            f"{item:16s} {label or '-':9s} "
            + " | ".join(f"{r['input']}: {r['answer'][:14]!r}{'' if r['correct'] or not label else ' x'}" for r in rows if r["item"] == item)
            + f"  [{time.time() - t0:.0f}s]"
        )
    R = pd.DataFrame(rows)
    run.save_csv(R, "audio_naming.csv")
    out = summarize(R, schedule, recs)
    out["setup"] = dict(question=QUESTION, followup=c3.FOLLOWUP, n_new=N_NEW, labels=labels, seconds=time.time() - t0)
    run.save_json(out, "summary.json")
    for inp in INPUTS:
        print(f"{inp}: named {out['share'][inp][0]:.3f} of recordings, {out['share_by_scene'][inp]:.3f} of scene receivers, gate {'passes' if out['gate'][inp] else 'fails'}")
    print(f"controls: {out['controls']}; gated schedule (both orders): {len(out['gated_scenes'])} of {len(schedule)}")


if __name__ == "__main__":
    main()
