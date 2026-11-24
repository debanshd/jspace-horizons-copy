"""Audio rules: why do the Gemma models misapply the three 7-or-3 rules when the spoken word comes before the question?

Why. audio_clean (PR #11) found both audio models name the spoken words (14 and 15 of 15) and apply every rule in the
question-first order, but with the audio first they answer "7" far too often (99 of 128 animal answers, against 64), so its gate
stopped the patching. With the original wording three explanations give the same "7": the digit 7 itself, the rule's "yes" answer
(the condition is taken as true), and the first digit the rule names. This run tells them apart by rewording the rules, before any
new patching design is chosen.

Variants of each rule (animal, ride, set), the condition in the dataset's words:
  A  "Rule: if the object named in the audio is <condition>, answer 7; otherwise answer 3. Reply with the number only."
     (spoken.QUESTIONS, verbatim): yes = 7, first named = 7
  B  "Rule: if the object named in the audio is <condition>, answer 3; otherwise answer 7. Reply with the number only."
     yes = 3, first named = 3
  D  "Rule: answer 3, unless the object named in the audio is <condition>, in which case answer 7. Reply with the number only."
     yes = 7, first named = 3
Inputs as audio_clean's: q_audio, audio_q, and typed (the transcript as text). Per scene, the receiver's and the donor's recording
under the scene's own condition (the set question's three names), every variant, every input; an identical prompt runs once.
Decoding and scoring are audio_clean's: greedy, media_steer.N_NEW = 4 new tokens, hit_limit recorded, strict.py's whole answer,
first answer only. Variant A repeats audio_clean's prompts exactly, so its answers are checked against the gate run's
(--clean-run) and the agreement is reported.

The rule (written before the run):
- Per input, task and variant: the share correct over the 128 receiver and donor answers, with a 95% bootstrap interval over
  recordings; the share of answers that are "7"; and, among the wrong valid answers, the share that are 7, the yes digit, and the
  first digit named.
- Diagnosis per input, from variants B and D pooled over the three rules (A cannot separate the three): an explanation fits
  when at least 0.8 of the wrong valid answers in B and, separately, in D are the digit it predicts. digit 7: 7 in B and 7 in D.
  yes: 3 in B, 7 in D. first named: 3 in B, 3 in D. Fewer than 10 wrong valid answers in B or D: "too few errors". None fits:
  "mixed".
- passes: per input and variant, whether the animal rule reaches video_sweep's patching gate (>= 0.85) for receivers and donors
  over the 64 scenes. A variant that passes with the audio first is a candidate wording for an audio-first sweep, which would be a
  new experiment with its own rule.
Run on the audio-capable models (Gemma-4-E2B, E4B and 12B).

Outputs in runs/<run>/<model>/: audio_rules.csv (every answer), summary.json, config.json, run.log.
"""

import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from media_steer import N_NEW, has_audio  # noqa: E402  the sweep's number of new tokens

from jspace import batched, spoken, strict  # noqa: E402
from jspace.analysis import boot_scene  # noqa: E402
from jspace.config import require_data  # noqa: E402
from jspace.model import Session  # noqa: E402
from jspace.runs import Run, parser  # noqa: E402

RULES = ["animal", "ride", "set"]
CONDITION = {"animal": "an animal", "ride": "something people ride or drive", "set": "a {a}, a {b} or a {c}"}
TEMPLATE = {
    "A": "Rule: if the object named in the audio is {cond}, answer 7; otherwise answer 3. Reply with the number only.",
    "B": "Rule: if the object named in the audio is {cond}, answer 3; otherwise answer 7. Reply with the number only.",
    "D": "Rule: answer 3, unless the object named in the audio is {cond}, in which case answer 7. Reply with the number only.",
}
YES = {"A": "7", "B": "3", "D": "7"}
FIRST = {"A": "7", "B": "3", "D": "3"}
PREDICTS = {"digit 7": ("7", "7"), "yes": ("3", "7"), "first named": ("3", "3")}  # the wrong answer in B, in D
INPUTS = spoken.INPUTS + ["typed"]
ROLES = ("receiver", "donor")
FIT, MIN_WRONG, GATE = 0.8, 10, 0.85
assert all(TEMPLATE["A"].format(cond=CONDITION[t]) == spoken.QUESTIONS[t] for t in RULES)  # A is the dataset's wording


def question(variant, task, membership):
    q = TEMPLATE[variant].format(cond=CONDITION[task])
    return q.format(a=membership[0], b=membership[1], c=membership[2]) if task == "set" else q


def truth(variant, task, word, membership):
    """the right digit: the yes digit when the condition holds"""
    holds = strict.truth(task, word, membership) == "7"
    return YES[variant] if holds else ("3" if YES[variant] == "7" else "7")


def messages(inp, q, word):
    parts = [{"type": "text", "text": q}, {"type": "text", "text": f"Transcript of the audio: {word}"}] if inp == "typed" else spoken.content(inp, q)
    return [{"role": "user", "content": parts}]


def diagnose(R):
    """the rule's diagnosis on the answers of one input (B and D, the three rules pooled)"""
    share = {}
    for v in ("B", "D"):
        w = R[(R["variant"] == v) & ~R["correct"] & R["valid"]]
        if len(w) < MIN_WRONG:
            return dict(verdict="too few errors", wrong={v: len(R[(R["variant"] == v) & ~R["correct"] & R["valid"]]) for v in ("B", "D")})
        share[v] = float((w["digit"] == "3").mean())
    fits = [h for h, (b, d) in PREDICTS.items() if (share["B"] if b == "3" else 1 - share["B"]) >= FIT and (share["D"] if d == "3" else 1 - share["D"]) >= FIT]
    return dict(verdict=fits[0] if fits else "mixed", wrong_share_3=share)


def summarize(R):
    out = dict(by={}, diagnosis={}, passes={})
    for inp in INPUTS:
        Ri = R[R["input"] == inp]
        out["by"][inp] = {}
        for task in RULES:
            for v in TEMPLATE:
                d = Ri[(Ri["task"] == task) & (Ri["variant"] == v)]
                wrong = d[~d["correct"] & d["valid"]]
                out["by"][inp][f"{task} {v}"] = dict(
                    correct=boot_scene(d["correct"].astype(float).to_numpy(), d["recording"].to_numpy()),
                    share_7=float((d["digit"] == "7").mean()),
                    valid=float(d["valid"].mean()),
                    hit_limit=int(d["hit_limit"].sum()),
                    wrong=len(wrong),
                    wrong_is_7=float((wrong["digit"] == "7").mean()) if len(wrong) else None,
                    wrong_is_yes=float((wrong["digit"] == YES[v]).mean()) if len(wrong) else None,
                    wrong_is_first=float((wrong["digit"] == FIRST[v]).mean()) if len(wrong) else None,
                )
        out["diagnosis"][inp] = diagnose(Ri[Ri["variant"].isin(["B", "D"])])
        out["passes"][inp] = {v: bool(all(Ri[(Ri["task"] == "animal") & (Ri["variant"] == v) & (Ri["role"] == r)]["correct"].mean() >= GATE for r in ROLES)) for v in TEMPLATE}
    return out


def agreement(R, clean_csv):
    """variant A against the gate run's answers to the same prompts: the share identical, per input"""
    if not clean_csv.exists():
        raise SystemExit(f"{clean_csv} not found: variant A is checked against audio_clean's answers (--clean-run)")
    C = pd.read_csv(clean_csv)
    C = C[(C["kind"] == "scene") & C["task"].isin(RULES)].set_index(["scene", "role", "input", "task"])["answer"].astype(str)
    A = R[R["variant"] == "A"].set_index(["scene", "role", "input", "task"])["answer"].astype(str)
    j = A.to_frame("a").join(C.to_frame("c"), how="inner")
    assert len(j) == len(A), "variant A has prompts the gate run did not ask"
    return {inp: float((g["a"] == g["c"]).mean()) for inp, g in j.groupby(level="input")}


def main():
    ap = parser(__doc__, "audio_rules")
    ap.add_argument("--clean-run", default="audio_clean", help="the audio_clean run whose answers variant A must repeat")
    args = ap.parse_args()
    require_data(args.data_dir, "spoken")
    run = Run(args, "audio_rules")
    lm = Session(args.model)
    if not has_audio(lm):
        raise SystemExit(f"{lm.name} takes no audio")
    run.write_config(lm)
    t0 = time.time()
    root = args.data_dir / spoken.DIR
    S = spoken.scenes(args.data_dir)
    recs = spoken.recordings(S)
    wav = {a: spoken.load_wav(root / rel) for a, (_, rel, _) in sorted(recs.items())}
    eos = lm.hf.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, (list, tuple)) else [eos])
    done = {}

    def answer(msgs, audio):
        """audio_clean's: greedy, N_NEW new tokens, one prompt per forward"""
        text = lm.chat_messages(msgs)
        enc = lm.processor(text=[text], audio=[audio], return_tensors="pt") if audio is not None else lm.processor(text=[text], return_tensors="pt")
        kw = {k: (v.to(lm.device).to(lm.dtype) if k in batched.FLOAT_KEYS else v.to(lm.device)) if torch.is_tensor(v) else v for k, v in enc.items() if k != "token_type_ids"}
        with torch.no_grad():
            g = lm.hf.generate(**kw, max_new_tokens=N_NEW, do_sample=False, pad_token_id=lm.tokenizer.pad_token_id)
        new = g[0, kw["input_ids"].shape[1] :]
        hit = len(new) >= N_NEW and not any(int(t) in eos for t in new)
        return lm.tokenizer.decode(new, skip_special_tokens=True).strip(), bool(hit)

    rows = []
    for m in S:
        for role in ROLES:
            a, word = m[f"{role}_audio_id"], m[f"{role}_label"]
            for inp in INPUTS:
                for task in RULES:
                    for v in TEMPLATE:
                        q = question(v, task, m["membership"])
                        key = (inp, a, q)
                        if key not in done:
                            done[key] = answer(messages(inp, q, word), None if inp == "typed" else wav[a])
                        text, hit = done[key]
                        right = truth(v, task, word, m["membership"])
                        digit = strict.clean(text).upper() if strict.valid(task, text, hit) else ""
                        rows.append(
                            dict(
                                scene=m["scene_id"],
                                role=role,
                                recording=a,
                                label=word,
                                input=inp,
                                task=task,
                                variant=v,
                                question=q,
                                answer=text,
                                hit_limit=hit,
                                digit=digit,
                                right=right,
                                correct=digit == right,
                                valid=bool(digit),
                            )
                        )
        print(f"scene {m['scene_id']:2d}: {len(done)} prompts  [{time.time() - t0:.0f}s]")
    R = pd.DataFrame(rows)
    run.save_csv(R, "audio_rules.csv")
    out = summarize(R)
    out["variant_A_matches_clean_run"] = agreement(R, args.runs_dir / args.clean_run / args.model.split("/")[-1] / "audio_clean.csv")
    out["setup"] = dict(templates=TEMPLATE, conditions=CONDITION, n_new=N_NEW, prompts=len(done), seconds=time.time() - t0)
    run.save_json(out, "summary.json")
    for inp in INPUTS:
        cells = "  ".join(f"{k} {v['correct'][0]:.2f}" for k, v in out["by"][inp].items())
        print(f"{inp}: {cells}\n   diagnosis {json.dumps(out['diagnosis'][inp])}; animal gate passes {out['passes'][inp]}")
    print(f"variant A identical to {args.clean_run}: {out['variant_A_matches_clean_run']}")


if __name__ == "__main__":
    main()
