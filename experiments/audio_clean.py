"""Audio clean answers: before any patching, each model's own answers to the five questions on the matched spoken-audio set,
in both prompt orders, under the generation settings of the sweep, scored whole-answer.

Why. Every patching result rests on the model answering the clean prompts first: the receiver's and, under the receiver's exact
question, the donor's (protocol.json's required baselines). This run asks every question the sweep asks, on every scene, with
the sweep's decoding (greedy, media_steer.N_NEW = 4 new tokens), records whether each answer hit that limit, and scores the
first answer only. It also measures the controls the dataset asks for: silence, noise and a typed transcript of the word.

Data. The 15 recordings of saxenan3/jspace-matched-spoken-audio at the pinned revision (src/jspace/spoken.py), one synthetic voice
saying one of 15 object names, each a 3-second 16 kHz mono WAV, fed to the model's own processor unchanged. Inputs:
  q_audio   the question, then the audio (Gemma-4's card's order)
  audio_q   the audio, then the question (the video experiments' order)
  typed     the question, then the word as text ("Transcript of the audio: <word>"): the text baseline the dataset asks for
Questions: the dataset's five (spoken.QUESTIONS, verbatim), the set question with the scene's three names. Per scene, both the
receiver's and the donor's recording are asked the scene's five questions in every input (an identical prompt is generated once
and its answer reused). Controls: 3 s of silence and 3 s of white noise (seed 0), asked the say question in both audio orders.
One prompt per forward (the sweep batches its rows, so its own clean rows, pclean, are the check under its batching).

Scoring (written before the run): src/jspace/strict.py, whole answer, first answer only. say is correct when the answer is the
spoken word itself (exact recognition); a plural, kind or spelling is reported apart as a synonym. letter: the one letter; set,
animal, ride: the one digit. An answer that hit the token limit is neither correct nor valid. No answer is asked again: a say
answer naming two or more things (clip_naming3's hedge) is wrong here, and is asked clip_naming3's follow-up only as a separate,
secondary measure (followup, followup_correct, summary["secondary"]), which no gate or schedule reads.

The rule (written before the run):
- Recognition, per input: how many of the 15 recordings the say answer names exactly (and as a synonym, validly, at the token
  limit), the share with a 95% bootstrap interval over recordings, and, for each recording missed, how many of the 64 scenes use
  it (2 to 18). The 0.90 share (14 of 15: one miss passes, two fail) is reported, not used: the schedule is per recording.
- A recording is reliable for an audio order when named exactly in it. gated_by_input: per order, the scenes whose receiver and
  donor recordings are both reliable; gated_scenes, the schedule audio_sweep reads: the scenes reliable in both orders.
- Clean answers, per input, question and role (receiver, donor): the share correct over the 64 scenes and over the gated ones,
  with 95% intervals bootstrapped over recordings (scenes that share a recording are resampled together).
- ready: at least 5 gated scenes, and in both audio orders the receivers of the gated scenes pass video_sweep's patching gate on
  say (>= 0.90) and animal (>= 0.85), and so do the donors under the receiver's question. audio_sweep and the launcher stop
  unless ready. The other three questions are reported, not gated, as in video_sweep.
- Controls: per audio order, whether the say answer to silence and to noise is a valid object name (reported, not gated).
Run on the audio-capable models (Gemma-4-E2B, E4B and, from 24 Nov, 12B); a model that takes no audio stops the run.

Outputs in runs/<run>/<model>/: audio_clean.csv (every answer), summary.json, config.json, run.log.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import clip_naming3 as c3  # noqa: E402  the hedge and its follow-up question (secondary only)
from media_steer import N_NEW, has_audio  # noqa: E402  the sweep's number of new tokens
from video_sweep import GATE  # noqa: E402  the patching gate

from jspace import batched, spoken, strict  # noqa: E402
from jspace.analysis import boot_scene  # noqa: E402
from jspace.config import require_data  # noqa: E402
from jspace.model import Session  # noqa: E402
from jspace.runs import Run, parser  # noqa: E402

TASKS = list(spoken.QUESTIONS)
INPUTS = spoken.INPUTS + ["typed"]
CONTROLS = {"silence": spoken.silence, "noise": spoken.noise}
ROLES = ("receiver", "donor")
MIN_SCENES = 5  # the pilot's scenes


def question(task, membership):
    return spoken.QUESTIONS[task].format(a=membership[0], b=membership[1], c=membership[2]) if task == "set" else spoken.QUESTIONS[task]


def messages(inp, q, word=None, first_answer=None):
    """one user turn in the input's order (typed: the question, then the transcript); with first_answer, the same
    conversation continued by clip_naming3's follow-up question"""
    parts = [{"type": "text", "text": q}, {"type": "text", "text": f"Transcript of the audio: {word}"}] if inp == "typed" else spoken.content(inp, q)
    msgs = [{"role": "user", "content": parts}]
    if first_answer is not None:
        msgs += [{"role": "assistant", "content": [{"type": "text", "text": first_answer}]}, {"role": "user", "content": [{"type": "text", "text": c3.FOLLOWUP}]}]
    return msgs


def interval(values, groups):
    return boot_scene(np.asarray(values, float), np.asarray(groups)) if len(values) else None


def summarize(R, schedule, recs):
    """the rule above on the answers R and the schedule [(scene, receiver recording, donor recording)]"""
    uses = {a: sum(a in (rc, dc) for _, rc, dc in schedule) for a in recs}
    out = dict(n_recordings=len(recs), n_scenes=len(schedule), recognition={}, reliable_by_input={}, gated_by_input={}, clean={}, controls={}, secondary=dict(followup={}))
    S = R[R["kind"] == "scene"]
    for inp in INPUTS:
        say = S[(S["input"] == inp) & (S["task"] == "say")].drop_duplicates("recording").set_index("recording").reindex(sorted(recs))
        assert not say["correct"].isna().any(), f"missing say answers for {inp}"
        named = {a for a, v in say["correct"].items() if v}
        out["recognition"][inp] = dict(
            named=len(named),
            share=interval(say["correct"].astype(float), np.arange(len(say))),
            synonym=int(say["synonym"].sum()),
            valid=int(say["valid"].sum()),
            hit_limit=int(say["hit_limit"].sum()),
            gate_090=bool(len(named) >= 0.90 * len(recs)),
            missed={a: dict(label=recs[a][0], answer=str(say.loc[a, "answer"]), scenes_lost=uses[a]) for a in sorted(set(recs) - named)},
        )
        out["secondary"]["followup"][inp] = dict(asked=int(say["hedged"].sum()), correct_after=int((say["correct"] | say["followup_correct"]).sum()))
        out["reliable_by_input"][inp] = sorted(named)
        if inp in spoken.INPUTS:
            out["gated_by_input"][inp] = [s for s, rc, dc in schedule if rc in named and dc in named]
    both = set(out["gated_by_input"]["q_audio"]) & set(out["gated_by_input"]["audio_q"])
    out["gated_scenes"] = [s for s, _, _ in schedule if s in both]
    ready = len(out["gated_scenes"]) >= MIN_SCENES
    for inp in INPUTS:
        out["clean"][inp] = {}
        for task in TASKS:
            out["clean"][inp][task] = {}
            for role in ROLES:
                d = S[(S["input"] == inp) & (S["task"] == task) & (S["role"] == role)].set_index("scene").reindex([s for s, _, _ in schedule])
                assert not d["correct"].isna().any(), f"missing {role} answers for {inp} {task}"
                g = d.loc[out["gated_scenes"]]
                e = dict(
                    all=interval(d["correct"].astype(float), d["recording"]),
                    gated=interval(g["correct"].astype(float), g["recording"]),
                    valid=float(d["valid"].astype(float).mean()),
                    hit_limit=int(d["hit_limit"].sum()),
                )
                if task in GATE and inp in spoken.INPUTS:
                    e["passes"] = bool(e["gated"] is not None and e["gated"][0] >= GATE[task])
                    ready = ready and e["passes"]
                out["clean"][inp][task][role] = e
        if inp in spoken.INPUTS:
            c = R[(R["kind"] == "control") & (R["input"] == inp)].set_index("recording")
            out["controls"][inp] = {k: dict(answer=str(c.loc[k, "answer"]), names_object=bool(c.loc[k, "valid"])) for k in CONTROLS}
    out["ready"] = bool(ready)
    return out


def order_ready(R, schedule, recs, order):
    """the rule's readiness for one prompt order alone (audio_sweep_qfirst's schedule): the scenes whose two recordings are named
    exactly in that order, at least MIN_SCENES of them, and on them the receivers and the donors pass the say and animal gates"""
    S = R[(R["kind"] == "scene") & (R["input"] == order)]
    say = S[S["task"] == "say"].drop_duplicates("recording").set_index("recording")["correct"].reindex(sorted(recs))
    assert not say.isna().any(), f"missing say answers for {order}"
    named = {a for a, v in say.items() if v}
    gated = [s for s, rc, dc in schedule if rc in named and dc in named]
    out = dict(order=order, named=len(named), gated_scenes=gated, clean={})
    ready = len(gated) >= MIN_SCENES
    for task in TASKS:
        for role in ROLES:
            d = S[(S["task"] == task) & (S["role"] == role)].set_index("scene").reindex(gated)
            assert not d["correct"].isna().any(), f"missing {role} answers for {order} {task}"
            share = float(d["correct"].astype(float).mean()) if len(d) else float("nan")
            out["clean"][f"{task} {role}"] = share
            if task in GATE:
                ready = ready and share >= GATE[task]
    out["ready"] = bool(ready)
    return out


def main():
    ap = parser(__doc__, "audio_clean")
    args = ap.parse_args()
    require_data(args.data_dir, "spoken")
    run = Run(args, "audio_clean")
    lm = Session(args.model)
    if not has_audio(lm):
        raise SystemExit(f"{lm.name} takes no audio")
    run.write_config(lm)
    t0 = time.time()
    root = args.data_dir / spoken.DIR
    S = spoken.scenes(args.data_dir)
    recs = spoken.recordings(S)
    schedule = [(r["scene_id"], r["receiver_audio_id"], r["donor_audio_id"]) for r in S]
    wav = {a: spoken.load_wav(root / rel) for a, (_, rel, _) in sorted(recs.items())}
    wav.update({k: f() for k, f in CONTROLS.items()})
    eos = lm.hf.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, (list, tuple)) else [eos])
    done = {}

    def answer(msgs, audio):
        """greedy, N_NEW new tokens, one prompt per forward: (text, prompt tokens, hit the limit)"""
        text = lm.chat_messages(msgs)
        enc = lm.processor(text=[text], audio=[audio], return_tensors="pt") if audio is not None else lm.processor(text=[text], return_tensors="pt")
        kw = {k: (v.to(lm.device).to(lm.dtype) if k in batched.FLOAT_KEYS else v.to(lm.device)) if torch.is_tensor(v) else v for k, v in enc.items() if k != "token_type_ids"}
        with torch.no_grad():
            g = lm.hf.generate(**kw, max_new_tokens=N_NEW, do_sample=False, pad_token_id=lm.tokenizer.pad_token_id)
        new = g[0, kw["input_ids"].shape[1] :]
        hit = len(new) >= N_NEW and not any(int(t) in eos for t in new)
        return lm.tokenizer.decode(new, skip_special_tokens=True).strip(), int(kw["input_ids"].shape[1]), bool(hit)

    def ask(inp, task, q, item, word):
        """the first answer, scored; a say hedge also asked the follow-up (secondary). Identical prompts run once."""
        key = (inp, item, q)
        if key not in done:
            audio = None if inp == "typed" else wav[item]
            text, ntok, hit = answer(messages(inp, q, word), audio)
            hedge, reply, reply_hit = task == "say" and c3.hedged(text, hit), "", False
            if hedge:
                reply, _, reply_hit = answer(messages(inp, q, word, text), audio)
            done[key] = dict(answer=text, tokens=ntok, hit_limit=hit, hedged=hedge, followup=reply, followup_hit_limit=reply_hit)
        return dict(done[key])

    rows = []
    for m in S:
        for role in ROLES:
            a, word = m[f"{role}_audio_id"], m[f"{role}_label"]
            for inp in INPUTS:
                for task in TASKS:
                    q = question(task, m["membership"])
                    x = ask(inp, task, q, a, word)
                    sc = strict.score(task, x["answer"], word, m["membership"], x["hit_limit"])
                    fc = bool(x["followup"]) and strict.matches(task, x["followup"], word, m["membership"], x["followup_hit_limit"])
                    rows.append(dict(kind="scene", scene=m["scene_id"], role=role, recording=a, label=word, input=inp, task=task, question=q, **x, **sc, followup_correct=fc))
        print(f"scene {m['scene_id']:2d} {m['receiver_label']:>9s} <- {m['donor_label']:9s}: {len(done)} prompts so far  [{time.time() - t0:.0f}s]")
    for k in CONTROLS:
        for inp in spoken.INPUTS:
            x = ask(inp, "say", spoken.QUESTIONS["say"], k, None)
            rows.append(
                dict(
                    kind="control",
                    scene=-1,
                    role="",
                    recording=k,
                    label="",
                    input=inp,
                    task="say",
                    question=spoken.QUESTIONS["say"],
                    **x,
                    correct=False,
                    synonym=False,
                    valid=strict.valid("say", x["answer"], x["hit_limit"]),
                    followup_correct=False,
                )
            )
    R = pd.DataFrame(rows)
    run.save_csv(R, "audio_clean.csv")
    out = summarize(R, schedule, recs)
    out["setup"] = dict(questions=spoken.QUESTIONS, n_new=N_NEW, followup=c3.FOLLOWUP, prompts=len(done), seconds=time.time() - t0)
    run.save_json(out, "summary.json")
    for inp in INPUTS:
        rc = out["recognition"][inp]
        print(f"{inp}: named {rc['named']} of {len(recs)} recordings exactly ({rc['synonym']} as a synonym, {rc['hit_limit']} at the token limit); missed {rc['missed']}")
        print("   " + "  ".join(f"{t} {out['clean'][inp][t]['receiver']['all'][0]:.2f}/{out['clean'][inp][t]['donor']['all'][0]:.2f}" for t in TASKS) + "  (receiver/donor, all scenes)")
    print(f"controls: {out['controls']}; gated scenes (both orders): {len(out['gated_scenes'])} of {len(schedule)}; ready: {out['ready']}")


if __name__ == "__main__":
    main()
