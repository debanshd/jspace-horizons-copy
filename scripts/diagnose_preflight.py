#!/usr/bin/env python
"""Why does a model fail Session's startup picture check? A diagnosis with no pass/fail rule: it measures, and the next step is
decided from what it shows. Written for Qwen3-Omni-30B-A3B-Instruct, which picked the right letters (ACD) but put only 0.32 of its
next-token probability on the letter tokens, under the check's bar of 0.5.

    uv run scripts/diagnose_preflight.py --model Qwen/Qwen3-Omni-30B-A3B-Instruct --run diagnose_preflight

The check itself is not changed and its threshold stays: the model is loaded as Session loads it, the picture check runs as
written, and its failure is recorded instead of stopping the load. Then, on the check's own three prompts (a card showing A, C or
D, the question, the chosen answer prefix):
1. the prompt as the model reads it (the end of the rendered chat template), and the answer prefix Session chose from the
   text-only questions (_check_batching), with its letter mass there;
2. the ten most likely next tokens with their probabilities, so the 0.68 the letters do not get is named: a formatting token
   (a newline, a space, "The"), another spelling of the letter, or nothing in particular;
3. the same with each of the two answer prefixes ("(" and none), whichever Session chose;
4. the model's greedy answer (12 tokens) to the check's prompt, with and without the prefix.
Outputs in runs/<run>/<model>/: summary.json, config.json, run.log.
"""

import torch

from jspace import stimuli
from jspace.model import INSTR, Session
from jspace.runs import Run, parser

TOP = 10


class Probe(Session):
    """Session, with the picture check's failure recorded instead of raised: everything else as Session does it"""

    def _preflight(self):
        self.preflight_error = None
        try:
            super()._preflight()
        except RuntimeError as e:
            self.preflight_error = str(e)
            print(f"picture check failed (recorded, the load goes on for the diagnosis): {e}")


def top_tokens(lm, texts, images):
    """per prompt: the TOP most likely next tokens as (token, probability)"""
    pr = torch.softmax(lm.forward(texts, images)[0], -1)
    p, ids = pr.topk(TOP, -1)
    return [[(lm.tokenizer.decode([int(i)]), float(q)) for q, i in zip(pp, ii)] for pp, ii in zip(p, ids)]


def main():
    ap = parser(__doc__, "diagnose_preflight")
    args = ap.parse_args()
    run = Run(args, "diagnose_preflight")
    lm = Probe(args.model)
    run.write_config(lm)
    # the picture check's prompt, exactly as Session._preflight builds it
    q = "Which letter is written in the picture?\n(A) A\n(B) B\n(C) C\n(D) D" + INSTR
    base = lm.chat_messages([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}])
    imgs = [stimuli.text_picture(c) for c in "ACD"]
    letters = {c: [lm.tokenizer.decode([t]) for t in ids] for c, ids in lm.letter_ids.items()}
    print(f"answer prefix chosen from the text-only questions: {lm.answer_prefix!r}; letter tokens counted: {letters}")
    print(f"the prompt's end as the model reads it: {base[-300:]!r}")
    out = dict(
        preflight_error=lm.preflight_error,
        vision_path=lm.vision_path,
        answer_prefix=lm.answer_prefix,
        letter_tokens=letters,
        prompt_end=base[-300:],
        top={},
        answers={},
    )
    for prefix in sorted({lm.answer_prefix, "(", ""}):
        tops = top_tokens(lm, [base + prefix] * 3, imgs)
        out["top"][repr(prefix)] = {c: t for c, t in zip("ACD", tops)}
        for c, t in zip("ACD", tops):
            print(f"prefix {prefix!r}, card {c}: " + ", ".join(f"{tok!r} {p:.3f}" for tok, p in t))
    for prefix in sorted({lm.answer_prefix, ""}):
        ans = lm.generate([base + prefix] * 3, imgs, 12)
        out["answers"][repr(prefix)] = dict(zip("ACD", ans))
        print(f"greedy answers after {prefix!r}: {dict(zip('ACD', ans))}")
    run.save_json(out, "summary.json")


if __name__ == "__main__":
    main()
