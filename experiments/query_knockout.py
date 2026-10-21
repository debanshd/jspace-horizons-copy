"""Which question token reads the picture? The attention knockout restricted to one query token: the rule's predicate word
("animal" in "if the object in image slot q is an animal"), the answer position, or every text token but one. (R15 in RECORD.md)

The position-resolved alignment (runs/run13a, figCrossPositions) says the decision heads of layer 7 deposit the object on the
predicate token and the name heads of layer 19 at the answer position. Here the reads of the image tokens are blocked for one
query token only, at the station layers, and the cost in nats is compared with blocking every text token's reads at the same
layers.

Rule (frozen 21 Oct 14:00): "the decision is read at the predicate" when blocking only the predicate token's reads at the decision
station costs >= 0.5 of the whole-layer cost and blocking every other token's reads costs <= 0.5 of it; "the name is read at the
answer position" when the same holds for the answer position at the name station. Prediction: both hold on the 2B (predicate at
layer 7, answer position at 11, 15 and 19), and the answer-position-only block at layer 7 costs the decisions little.

Outputs in runs/<run>/<model>/: query_knockout.csv, summary.json.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from attention_knockout import Knockout, attention_layers, generate_scored  # noqa: E402
from cross_modal_patch import scenes  # noqa: E402
from read_battery import INTRO, QUESTIONS, first_token_ids, matches, truth  # noqa: E402
from source_conflict import STATIONS  # noqa: E402

from jspace import stimuli  # noqa: E402
from jspace.analysis import boot_scene  # noqa: E402
from jspace.config import require_data  # noqa: E402
from jspace.model import Session  # noqa: E402
from jspace.runs import Run, parser  # noqa: E402

# the token whose reads are tested, per question
PREDICATE = {"animal": " animal", "category": " animal", "set": " a", "say": " object", "yesno": " animal", "ride": " ride or drive", "legs": " legs", "notanimal": " animal"}


def positions_after_image(lm, text, image, needle, occurrence_after=0):
    """absolute positions of the tokens covering `needle` (first occurrence after the character index) in a picture prompt's encoding"""
    enc = lm.encode([text], [image])
    ids = enc["input_ids"][0].tolist()
    e = max(j for j, t in enumerate(ids) if t == lm.image_token_id)
    cut = text.index("<|vision_end|>")
    suffix = text[cut:]
    tok = lm.tokenizer(suffix, return_offsets_mapping=True, add_special_tokens=False)
    assert tok["input_ids"] == ids[e + 1 :], "the suffix tokenizes differently alone"
    i = suffix.index(needle, max(0, occurrence_after - cut))
    rel = [t for t, (a, b) in enumerate(tok["offset_mapping"]) if a < i + len(needle) and b > i]
    return [e + 1 + r for r in rel], len(ids)


def main():
    ap = parser(__doc__, "query_knockout")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--crop-set", type=int, default=1)
    ap.add_argument("--questions", default="animal,category,say")
    args = ap.parse_args()
    require_data(args.data_dir, "coco")
    run = Run(args, "query_knockout")
    lm = Session(args.model)
    run.write_config(lm)
    crops = stimuli.load_crops(lm, args.data_dir / "coco2017", args.crop_set)
    crops.save(run)
    S = scenes(crops.words, crops.tiles, args.n)
    words = crops.words
    AL = [L for L, _ in attention_layers(lm)]
    dec, nam = STATIONS.get(args.model, (AL[1:3], AL[2:5]))
    t0 = time.time()
    res = []
    for task in args.questions.split(","):
        tmpl, needs, kind = QUESTIONS[task]
        TX = [truth(task, r, r.X) for r in S.itertuples()]
        TZ = [truth(task, r, r.Z) for r in S.itertuples()]
        ids_x = [first_token_ids(lm, t, kind) for t in TX]
        ids_z = [first_token_ids(lm, t, kind) for t in TZ]
        texts = [lm.chat_media([INTRO, "IMAGE", tmpl.format(q=r.q, a=r.set5[0], b=r.set5[1], c=r.set5[2], d=r.set5[3], e=r.set5[4])]) for r in S.itertuples()]
        images = [stimuli.scene(r.q, r.tz) for r in S.itertuples()]
        pred, last = [], []
        for text, img in zip(texts, images):
            p, n = positions_after_image(lm, text, img, PREDICATE[task], text.index("<|vision_end|>"))
            pred.append(p)
            last.append([n - 1])
        stations = {"decision station": dec, "name station": nam}
        arms = [("clean", None)]
        for sname, layers in stations.items():
            arms += [
                (f"all text tokens, {sname}", Knockout(lm, layers, lm.image_token_id)),
                (f"predicate token only, {sname}", Knockout(lm, layers, lm.image_token_id, query_positions=pred)),
                (f"answer position only, {sname}", Knockout(lm, layers, lm.image_token_id, query_positions=last)),
                (f"all but the predicate, {sname}", Knockout(lm, layers, lm.image_token_id, query_positions=pred, query_all_but=True)),
            ]
        for name, ko in arms:
            out = generate_scored(lm, texts, images, 5, ko, ids_x, ids_z)
            for i, r in enumerate(S.itertuples()):
                gen, lpx, lpz = out[i]
                res.append(dict(task=task, arm=name, scene=r.scene, X=r.X, Z=r.Z, generation=gen, correct=matches(task, gen, TZ[i], words), ld=lpz - lpx, distinct=TX[i] != TZ[i]))
            d_ = [x for x in res if x["task"] == task and x["arm"] == name and x["distinct"]]
            print(f"{task:9s} {name:40s} correct {np.mean([x['correct'] for x in d_]):.2f}  ld {np.mean([x['ld'] for x in d_]):+.2f}  [{time.time() - t0:.0f}s]")
    R = pd.DataFrame(res)
    run.save_csv(R, "query_knockout.csv")
    out = dict(n=int(len(S)), decision_station=dec, name_station=nam, predicate=PREDICATE, questions={})
    for task in R.task.unique():
        d = R[(R.task == task) & R.distinct]
        sc = np.sort(d.scene.unique())
        clean = d[d.arm == "clean"].set_index("scene").reindex(sc)
        e = dict(clean_correct=float(clean.correct.mean()), clean_ld=boot_scene(clean.ld.values, sc), arms={})
        for name in d.arm.unique():
            if name == "clean":
                continue
            g = d[d.arm == name].set_index("scene").reindex(sc)
            e["arms"][name] = dict(correct=boot_scene(g.correct.astype(float).values, sc), cost=boot_scene((clean.ld - g.ld).values, sc))
        for sname in stations:
            whole = e["arms"][f"all text tokens, {sname}"]["cost"][0]
            e[f"share at the predicate, {sname}"] = (e["arms"][f"predicate token only, {sname}"]["cost"][0] / whole) if whole > 0.5 else None
            e[f"share at the answer position, {sname}"] = (e["arms"][f"answer position only, {sname}"]["cost"][0] / whole) if whole > 0.5 else None
            e[f"share elsewhere, {sname}"] = (e["arms"][f"all but the predicate, {sname}"]["cost"][0] / whole) if whole > 0.5 else None
        out["questions"][task] = e
        print(f"{task}: " + ", ".join(f"{k} {v:.2f}" for k, v in e.items() if k.startswith("share") and v is not None))
    out["seconds"] = time.time() - t0
    run.save_json(out, "summary.json")


if __name__ == "__main__":
    main()
