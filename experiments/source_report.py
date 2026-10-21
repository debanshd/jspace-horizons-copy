"""Source report after a cross-modal interchange: when the content of a read word is written into the run that sees a picture, does
the model say it saw the object or read it? (R10 in RECORD.md; rubric gates XXXIX, XL; Johnson's source-monitoring question)

The receiver sees a picture of Z in slot q and a note about another slot: "A note says: slot p contains a W." The donor reads two
notes and no picture: "A note says: slot q contains a X. A note says: slot p contains a W." Both end with the same question, which
asks for the object in slot q (or, as the positive control, in slot p) and whether it was seen in the picture or read in the note,
in the form "name, seen" or "name, read". The donor's per-head attention outputs at the shared suffix (the note about p and the
question) are written into the receiver at the chosen layers, as in cross_modal_patch.py.

Gates: the clean receiver answers "Z, seen" for slot q and "W, read" for slot p on >= 0.85 of scenes; the donor answers "X, read"
for slot q on >= 0.85. Rule (frozen 21 Oct 09:00): with the station's heads written in (all heads of 7 and 11 on the 2B, the arm
that moved the decision on 0.98 of scenes in run13), the receiver's name follows X on >= 0.5 of scenes (the interchange holds under
this question), and the source report on those scenes is "seen" on >= 0.8 (the content arrives without its provenance: the report
follows the channel the slot belongs to) or "read" on >= 0.5 (the heads' content carries its provenance and the report tracks it).
Either outcome is a finding; between them is undetermined. Prediction: "seen" on >= 0.8.

Outputs in runs/<run>/<model>/: source_report.csv, summary.json.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from attention_knockout import attention_layers  # noqa: E402
from cross_modal_patch import capture, common_suffix, generate_patched, head_geometry, scenes  # noqa: E402
from read_battery import INTRO, first_token_ids  # noqa: E402

from jspace import stimuli  # noqa: E402
from jspace.analysis import boot_scene  # noqa: E402
from jspace.config import require_data  # noqa: E402
from jspace.model import Session  # noqa: E402
from jspace.runs import Run, parser  # noqa: E402

QUESTION = (
    "Question: what is the object in slot {s}, and did you see it in the picture or read it in the note? "
    "Reply with two words only, the object's name and then seen or read, like 'cup, seen' or 'cup, read'."
)


def parse(gen, words):
    """(name or None, 'seen' | 'read' | None) from the first line of the answer"""
    g = gen.strip().lower().split("\n")[0]
    name = next((w for w in words if g.startswith(w)), None)
    src = "seen" if "seen" in g or "saw" in g or "picture" in g else "read" if "read" in g or "note" in g else None
    return name, src


def main():
    ap = parser(__doc__, "source_report")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--crop-set", type=int, default=1)
    ap.add_argument("--layers", default="7+11,11+15+19,7+11+15+19", help="joint all-head patches over these layer groups")
    ap.add_argument(
        "--donor-control",
        action="store_true",
        help="the donor reads only the note about slot p (no object for slot q): do the written-in heads carry a free 'read' report, or the object's own channel?",
    )
    args = ap.parse_args()
    require_data(args.data_dir, "coco")
    run = Run(args, "source_report")
    lm = Session(args.model)
    run.write_config(lm)
    crops = stimuli.load_crops(lm, args.data_dir / "coco2017", args.crop_set)
    crops.save(run)
    S = scenes(crops.words, crops.tiles, args.n)
    words = crops.words
    rng = np.random.default_rng(23)
    S["p"] = [int(rng.choice([s for s in range(1, stimuli.N_SLOTS + 1) if s != q])) for q in S.q]
    S["W"] = [words[int(rng.integers(len(words)))] for _ in range(len(S))]
    S["W"] = [w if w not in (x, z) else next(v for v in words if v not in (x, z)) for w, x, z in zip(S.W, S.X, S.Z)]
    groups = [[int(v) for v in g.split("+")] for g in args.layers.split(",")]
    layers = sorted({L for g in groups for L in g})
    mods = {L: m for L, m in dict(attention_layers(lm)).items() if L in layers}
    H, head_dim = head_geometry(mods)
    t0 = time.time()
    res = []
    for target in ("q", "p"):
        note = [f"A note says: slot {r.p} contains a {r.W}." for r in S.itertuples()]
        question = [QUESTION.format(s=getattr(r, target)) for r in S.itertuples()]
        rec_txt = [lm.chat_media([INTRO, "IMAGE", f"{n} {qu}"]) for n, qu in zip(note, question)]
        rec_img = [stimuli.scene(r.q, r.tz) for r in S.itertuples()]
        if args.donor_control:
            don_txt = [lm.chat([("user", f"{INTRO} {n} {qu}")]) for n, qu in zip(note, question)]
        else:
            don_txt = [lm.chat([("user", f"{INTRO} A note says: slot {r.q} contains a {r.X}. {n} {qu}")]) for r, n, qu in zip(S.itertuples(), note, question)]
        K = [common_suffix(lm, a, b) for a, b in zip(rec_txt, don_txt)]
        print(f"slot {target}: shared suffix {min(K)} to {max(K)} tokens  [{time.time() - t0:.0f}s]")
        donor = capture(lm, don_txt, [None] * len(S), mods, K)
        ids_x = [first_token_ids(lm, r.X, "word") for r in S.itertuples()]
        ids_z = [first_token_ids(lm, r.Z, "word") for r in S.itertuples()]
        arms = [("clean", {}), ("donor", None)] + [("all heads of " + "+".join(map(str, g)), {L: list(range(H)) for L in g}) for g in groups]
        for name, spec in arms:
            if spec is None:  # the donor itself, unpatched
                out = generate_patched(lm, don_txt, [None] * len(S), 8, mods, head_dim, {}, donor, ids_x, ids_z)
            else:
                out = generate_patched(lm, rec_txt, rec_img, 8, mods, head_dim, spec, donor, ids_x, ids_z)
            for i, r in enumerate(S.itertuples()):
                gen, lpx, lpz = out[i]
                nm, src = parse(gen, words)
                res.append(
                    dict(
                        target=target,
                        arm=name,
                        scene=r.scene,
                        X=r.X,
                        Z=r.Z,
                        W=r.W,
                        q=r.q,
                        p=r.p,
                        generation=gen,
                        name=nm,
                        source=src,
                        follows_X=nm == r.X,
                        keeps_Z=nm == r.Z,
                        names_W=nm == r.W,
                        ld=lpx - lpz,
                    )
                )
            d_ = [x for x in res if x["target"] == target and x["arm"] == name]
            print(
                f"slot {target} {name:26s} names X {np.mean([x['follows_X'] for x in d_]):.2f}  Z {np.mean([x['keeps_Z'] for x in d_]):.2f}  W {np.mean([x['names_W'] for x in d_]):.2f}  "
                f"seen {np.mean([x['source'] == 'seen' for x in d_]):.2f}  read {np.mean([x['source'] == 'read' for x in d_]):.2f}  [{time.time() - t0:.0f}s]"
            )
    R = pd.DataFrame(res)
    run.save_csv(R, "source_report.csv")
    out = dict(n=int(len(S)), groups=groups, heads=H, head_dim=head_dim, question=QUESTION, arms={})
    sc = np.sort(R.scene.unique())
    for target in ("q", "p"):
        out["arms"][target] = {}
        for name in R[R.target == target].arm.unique():
            g = R[(R.target == target) & (R.arm == name)].set_index("scene").reindex(sc)
            e = dict(
                names_X=boot_scene(g.follows_X.astype(float).values, sc),
                names_Z=boot_scene(g.keeps_Z.astype(float).values, sc),
                names_W=boot_scene(g.names_W.astype(float).values, sc),
                seen=boot_scene((g.source == "seen").astype(float).values, sc),
                read=boot_scene((g.source == "read").astype(float).values, sc),
            )
            fx = g[g.follows_X.astype(bool)]
            e["seen_given_X"] = boot_scene((fx.source == "seen").astype(float).values, fx.index.values) if len(fx) else None
            e["read_given_X"] = boot_scene((fx.source == "read").astype(float).values, fx.index.values) if len(fx) else None
            out["arms"][target][name] = e
    c, dn = out["arms"]["q"]["clean"], out["arms"]["q"]["donor"]
    pc = out["arms"]["p"]["clean"]
    out["gates"] = dict(
        clean_Z_seen=bool(c["names_Z"][0] >= 0.85 and c["seen"][0] >= 0.85),
        clean_W_read=bool(pc["names_W"][0] >= 0.85 and pc["read"][0] >= 0.85),
        donor_X_read=bool(dn["names_X"][0] >= 0.85 and dn["read"][0] >= 0.85),
    )
    print("gates", out["gates"])
    for name, e in out["arms"]["q"].items():
        if name.startswith("all"):
            sx = e["seen_given_X"][0] if e["seen_given_X"] else float("nan")
            rx = e["read_given_X"][0] if e["read_given_X"] else float("nan")
            print(f"{name}: names X {e['names_X'][0]:.2f}; of those, seen {sx:.2f}, read {rx:.2f}")
    out["seconds"] = time.time() - t0
    run.save_json(out, "summary.json")


if __name__ == "__main__":
    main()
