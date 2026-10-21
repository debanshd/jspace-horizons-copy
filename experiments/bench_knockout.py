"""Benchmark knockout: does graded difficulty move the depth at which a question stops reading the image? (R6 in RECORD.md)

MathVision (MathLLMs/MathVision, test split: 3,040 problems with a figure, five difficulty levels, half multiple choice) and
CountBenchQA (vikhyatk/CountBenchQA: 491 photos, "how many X are there in the image?", counts 2 to 10). First a clean pass over every
item; the items the model answers correctly (exact match on the first word, the option letter for multiple choice) are kept, which
selects on the clean answer and never on the knockout statistic. Then attention from the text to the image tokens is blocked at the
softmax layers (single, from L on, up to L), as in attention_knockout.py. Per item and arm: the answer, correct, and log p(the
reference's first token). Cost = clean log p minus the knockout's, in nats, grouped by level (MathVision) or by count (CountBench),
and by answer form (option letter / number).

Rules (frozen 21 Oct 05:40): per group, last reader = largest softmax layer whose from-L-on knockout costs >= 1 nat (item-grouped
interval above 0). MathVision: "difficulty does not move the read" when the last readers of levels 1 and 5 are within one softmax
layer of each other; "harder problems read earlier" (just in time) when level 5's is two or more softmax layers before level 1's.
CountBench: the same for counts 2 to 4 against 7 to 10. Prediction: no movement in either benchmark, every decision at the model's
decision station (11 on the 4B); numeric answers that are read off the figure may show the copy station (23) and are reported by
answer form.

Outputs in runs/<run>/<model>/: bench_knockout.csv, clean_pass.csv, summary.json, stimuli.png.
"""

import ast
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import io  # noqa: E402

PREFILL = "The final answer is "

from attention_knockout import Knockout, attention_layers  # noqa: E402
from gqa_knockout import generate_logp  # noqa: E402

from jspace import stimuli  # noqa: E402
from jspace.analysis import boot_scene  # noqa: E402
from jspace.model import Session  # noqa: E402
from jspace.runs import Run, parser  # noqa: E402


def photo(d):
    return stimuli.fit_photo(d["bytes"]) if isinstance(d, dict) else stimuli.fit_photo(io.BytesIO(d).getvalue())


def norm(a, form="number"):
    """VLMEvalKit-style rule-based extraction for the direct-answer setting: the option letter for multiple choice (can_infer_option),
    else the first number or the first word (can_infer_text), after stripping markdown, LaTeX and boxes"""
    a = str(a).strip().strip("*").strip().lower()
    a = re.sub(r"\\boxed\{([^}]*)\}", r"\1", a).replace("$", "").replace("\\", "").strip()
    if form == "letter":
        m = re.match(r"[\(\[]?([a-e])[\)\].:,]?(\s|$)", a)
        return m.group(1).upper() if m else ""
    m = re.match(r"-?\d+(\.\d+)?", a.replace(",", ""))
    if m:
        v = m.group(0)
        return v[:-2] if v.endswith(".0") else v
    return a.split()[0] if a.split() else ""


def load_mathvision(data_dir, n_max, rng):
    d = pd.read_parquet(data_dir / "mathvision" / "data" / "test-00000-of-00001-3532b8d3f1b4047a.parquet")
    rows = []
    for r in d.itertuples():
        opts = ast.literal_eval(r.options) if isinstance(r.options, str) else ([] if r.options is None else [str(o) for o in list(r.options)])
        q = r.question.replace("<image1>", "").strip()
        if "<image" in q:  # problems with several figures are skipped
            continue
        if opts:
            q = q + "\n" + "\n".join(f"({chr(65 + i)}) {o}" for i, o in enumerate(opts)) + "\nAnswer with the option's letter from the given choices directly."
            form = "letter"
        else:
            q = q + "\nAnswer the question using a single word or number."
            form = "number" if re.fullmatch(r"-?\d+(\.\d+)?", str(r.answer).strip()) else "other"
        rows.append(dict(item=str(r.id), prompt=q, reference=str(r.answer).strip(), group=f"level {r.level}", level=int(r.level), form=form, subject=r.subject, image=r.decoded_image))
    D = pd.DataFrame(rows)
    D = D[D.form != "other"]
    if n_max and len(D) > n_max:
        D = D.iloc[rng.permutation(len(D))[:n_max]]
    return D.reset_index(drop=True)


def load_countbench(data_dir, n_max, rng):
    d = pd.read_parquet(data_dir / "countbench" / "data" / "test-00000-of-00001.parquet")
    rows = [
        dict(
            item=str(i),
            prompt=r.question.strip() + "\nAnswer with a number.",
            reference=str(r.number),
            group=("count 2 to 4" if int(r.number) <= 4 else "count 5 to 6" if int(r.number) <= 6 else "count 7 to 10"),
            level=int(r.number),
            form="number",
            subject="count",
            image=r.image,
        )
        for i, r in enumerate(d.itertuples())
    ]
    D = pd.DataFrame(rows)
    if n_max and len(D) > n_max:
        D = D.iloc[rng.permutation(len(D))[:n_max]]
    return D.reset_index(drop=True)


def main():
    ap = parser(__doc__, "bench_knockout")
    ap.add_argument("--bench", choices=["mathvision", "countbench"], required=True)
    ap.add_argument("--n-max", type=int, default=0, help="cap on items before the clean pass (0 = all)")
    ap.add_argument("--n-keep", type=int, default=400, help="cap on correctly answered items that enter the knockout")
    args = ap.parse_args()
    run = Run(args, f"bench_knockout_{args.bench}")
    lm = Session(args.model)
    run.write_config(lm)
    t0 = time.time()
    rng = np.random.default_rng(0)
    D = load_mathvision(args.data_dir, args.n_max, rng) if args.bench == "mathvision" else load_countbench(args.data_dir, args.n_max, rng)
    images = [photo(im) for im in D.image]
    D = D.drop(columns=["image"])
    texts = [lm.chat_media(["IMAGE", p]) + PREFILL for p in D.prompt]  # VLMEvalKit direct-answer setting: the first generated token is the answer
    ids = [lm.single_ids([a, " " + a]) or [lm.tokenizer.encode(" " + a, add_special_tokens=False)[0]] for a in D.reference]
    print(f"{args.bench}: {len(D)} items, forms {D.form.value_counts().to_dict()}, groups {D.group.value_counts().to_dict()}  [{time.time() - t0:.0f}s]")
    out0 = generate_logp(lm, texts, images, 8, None, ids)
    D["generation"] = [o[0] for o in out0]
    D["logp_clean"] = [o[1] for o in out0]
    D["correct_clean"] = [norm(g, f) == norm(a, f) for g, a, f in zip(D.generation, D.reference, D.form)]
    run.save_csv(D, "clean_pass.csv")
    acc = D.groupby("group").correct_clean.mean().round(2).to_dict()
    print(f"clean accuracy by group {acc}; by form {D.groupby('form').correct_clean.mean().round(2).to_dict()}  [{time.time() - t0:.0f}s]")
    keep = D[D.correct_clean].index.values
    if len(keep) == 0:
        raise SystemExit("no item answered correctly on the clean pass: check the prompt and the scoring in clean_pass.csv")
    if len(keep) > args.n_keep:
        keep = np.sort(rng.permutation(keep)[: args.n_keep])
    K = D.loc[keep].reset_index(drop=True)
    kt = [texts[i] for i in keep]
    ki = [images[i] for i in keep]
    kids = [ids[i] for i in keep]
    stimuli.contact_sheet([(f"{r.group}: {r.reference}", ki[j]) for j, r in enumerate(K.head(6).itertuples())], 3, run.path("stimuli.png"))
    AL = [i for i, _ in attention_layers(lm)]
    arms = [("no knockout", None, "clean", None)]
    arms += [(f"block layer {L}", [L], "single", L) for L in AL]
    arms += [(f"block layers {L} on", [M for M in AL if M >= L], "from", L) for L in AL]
    arms += [(f"block layers up to {L}", [M for M in AL if M <= L], "upto", L) for L in AL]
    res = []
    for name, block, kind_, L in arms:
        ko = Knockout(lm, block, lm.image_token_id) if block else None
        out = generate_logp(lm, kt, ki, 8, ko, kids)
        for j, r in enumerate(K.itertuples()):
            res.append(
                dict(
                    arm=name,
                    kind=kind_,
                    layer=L,
                    item=r.item,
                    group=r.group,
                    level=r.level,
                    form=r.form,
                    subject=r.subject,
                    reference=r.reference,
                    generation=out[j][0],
                    correct=norm(out[j][0], r.form) == norm(r.reference, r.form),
                    logp=out[j][1],
                )
            )
        d_ = pd.DataFrame([x for x in res if x["arm"] == name])
        print(
            f"{name:24s} correct {d_.correct.mean():.2f} logp {d_.logp.mean():+.2f} | "
            + " ".join(f"{g}: {d_[d_.group == g].correct.mean():.2f}" for g in sorted(d_.group.unique()))
            + f"  [{time.time() - t0:.0f}s]"
        )
    R = pd.DataFrame(res)
    run.save_csv(R, "bench_knockout.csv")
    out = dict(bench=args.bench, n_items=int(len(D)), n_kept=int(len(K)), clean_accuracy_by_group=acc, attention_layers=AL, groups={}, forms={})
    for key, col in (("groups", "group"), ("forms", "form")):
        for g in sorted(R[col].unique()):
            d = R[R[col] == g]
            clean = d[d.kind == "clean"].set_index("item")
            e = dict(n=int(len(clean)), clean_logp=float(clean.logp.mean()))
            for kind_ in ("single", "from", "upto"):
                e[kind_] = {}
                for L in AL:
                    gg = d[(d.kind == kind_) & (d.layer == L)].set_index("item").reindex(clean.index)
                    e[kind_][L] = dict(correct=boot_scene(gg.correct.astype(float).fillna(0).values, gg.index.values), cost=boot_scene((clean.logp - gg.logp).values, gg.index.values))
            e["readers"] = [L for L in AL if e["single"][L]["cost"][0] >= 1 and e["single"][L]["cost"][1] > 0]
            e["last_reader"] = max([L for L in AL if e["from"][L]["cost"][0] >= 1 and e["from"][L]["cost"][1] > 0], default=None)
            e["all_blocked_cost"] = e["from"][AL[0]]["cost"]
            out[key][g] = e
    print(
        f"{args.bench} knockout: " + "; ".join(f"{g}: n {v['n']}, readers {v['readers']}, last reader {v['last_reader']}, all blocked {v['all_blocked_cost'][0]:.1f}" for g, v in out["groups"].items())
    )
    print("by form: " + "; ".join(f"{g}: n {v['n']}, readers {v['readers']}, last reader {v['last_reader']}" for g, v in out["forms"].items()))
    out["seconds"] = time.time() - t0
    run.save_json(out, "summary.json")


if __name__ == "__main__":
    main()
