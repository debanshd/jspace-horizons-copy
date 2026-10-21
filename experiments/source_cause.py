"""Does the source report record the pathway that delivered the content, or detect that the picture's evidence is missing? (R23 in
RECORD.md; the matched-disruption test asked for by the outside reviews after row 115; rubric gates CXXIX to CXXXI, CXC)

Row 115 set the cause by blocking the picture's reads, which also removes the picture's evidence at the question's tokens, so a
report computed from what evidence is present would move the same way (Singh, Linzen and Ravfogel 2026: anomaly detection). Here
nothing is blocked. The receiver sees Z in slot q and reads "A note says: slot q contains a Z." (the same answer from either
source, as in run19). The cause is set by writing a donor's per-head attention outputs into the receiver at the tokens after the
image (the note, the question and the answer position), at chosen layers, with the receiver's attention to the picture intact:

  donor A, the note as the cause: the same note and question with an EMPTY canvas, so its answer came from the note;
  donor B, the picture as the cause with a matched disruption: ANOTHER crop of the same object Z in slot q with the same note, so
           its head outputs differ from the receiver's own while the picture is still what delivered the object;
  donor N, noise matched to A: the receiver's own head outputs plus Gaussian noise with, at every position, the norm of A's
           displacement from them, so the disruption is A's size with no content;
  donor C, the receiver itself: a null patch (the machinery check).

Gates: the clean receiver answers "Z, seen" on >= 0.85 of scenes; donor A alone answers "Z, read" on >= 0.85; donor B alone
"Z, seen" on >= 0.85; the null patch leaves the clean answer and report on >= 0.95. Rule (frozen 21 Oct 15:05 local, before any
run): among scenes naming Z under both the clean run and the arm, "the report records the pathway" when A's heads at all five
station layers move the report seen -> read on >= 0.5 of scenes and B's and N's at the same layers move it on <= 0.2 each; "the
report detects the disruption" when B's or N's move it on >= 0.5; between is undetermined. B counts as matched only when its
relative displacement of the head outputs is at least half of A's at every patched layer (reported per layer); N is matched by
construction. Prediction: records the pathway (A >= 0.5 with all five layers, >= 0.3 with the decision station alone, as the
knockout gave 31 of 64 in row 115; B and N <= 0.1).

Outputs in runs/<run>/<model>/: source_cause.csv, summary.json.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from attention_knockout import attention_layers  # noqa: E402
from cross_modal_patch import capture, generate_patched, head_geometry, scenes  # noqa: E402
from read_battery import INTRO, first_token_ids  # noqa: E402
from source_report import QUESTION, parse  # noqa: E402

from jspace import stimuli  # noqa: E402
from jspace.analysis import boot_scene  # noqa: E402
from jspace.config import require_data  # noqa: E402
from jspace.model import Session  # noqa: E402
from jspace.runs import Run, parser  # noqa: E402


def suffix_length(lm, text, image):
    """number of tokens after the last image token (the note, the question and the assistant prefix)"""
    ids = lm.encode([text], [image])["input_ids"][0].tolist()
    e = max(j for j, t in enumerate(ids) if t == lm.image_token_id)
    return len(ids) - e - 1


def main():
    ap = parser(__doc__, "source_cause")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--crop-set", type=int, default=1)
    ap.add_argument("--layers", default="7+11,15+19+23,7+11+15+19+23", help="joint all-head patches over these layer groups")
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()
    require_data(args.data_dir, "coco")
    run = Run(args, "source_cause")
    lm = Session(args.model)
    run.write_config(lm)
    crops = stimuli.load_crops(lm, args.data_dir / "coco2017", args.crop_set)
    crops.save(run)
    S = scenes(crops.words, crops.tiles, args.n)
    words = crops.words
    bank = crops.tiles
    S["tz2"] = [bank[r.Z][(r.scene + 2) % len(bank[r.Z])] for r in S.itertuples()]  # another crop of the same object
    groups = [[int(v) for v in g.split("+")] for g in args.layers.split(",")]
    layers = sorted({L for g in groups for L in g})
    mods = {L: m for L, m in dict(attention_layers(lm)).items() if L in layers}
    H, head_dim = head_geometry(mods)
    t0 = time.time()
    texts = [lm.chat_media([INTRO, "IMAGE", f"A note says: slot {r.q} contains a {r.Z}. {QUESTION.format(s=r.q)}"]) for r in S.itertuples()]
    img_rec = [stimuli.scene(r.q, r.tz) for r in S.itertuples()]
    img_a = [stimuli.scene(r.q, None) for r in S.itertuples()]
    img_b = [stimuli.scene(r.q, r.tz2) for r in S.itertuples()]
    K = [suffix_length(lm, t, im) for t, im in zip(texts, img_rec)]
    assert all(suffix_length(lm, t, im) == k for t, im, k in zip(texts, img_a, K)), "the empty canvas changes the token count"
    print(f"suffix after the image: {min(K)} to {max(K)} tokens  [{time.time() - t0:.0f}s]")
    ids_x = [first_token_ids(lm, r.X, "word") for r in S.itertuples()]
    ids_z = [first_token_ids(lm, r.Z, "word") for r in S.itertuples()]
    don = {}
    don["C"] = capture(lm, texts, img_rec, mods, K)
    don["A"] = capture(lm, texts, img_a, mods, K)
    don["B"] = capture(lm, texts, img_b, mods, K)
    rng = np.random.default_rng(args.seed)
    don["N"] = {L: [None] * len(S) for L in mods}
    disp = {d: {L: [] for L in mods} for d in ("A", "B", "N")}
    for L in mods:
        for i in range(len(S)):
            c, a = don["C"][L][i], don["A"][L][i]
            eps = rng.standard_normal(c.shape).astype(np.float32)
            eps *= np.linalg.norm(a - c, axis=1, keepdims=True) / (np.linalg.norm(eps, axis=1, keepdims=True) + 1e-8)
            don["N"][L][i] = c + eps
            for d in ("A", "B", "N"):
                x = don[d][L][i]
                disp[d][L].append(float(np.mean(np.linalg.norm(x - c, axis=1) / (np.linalg.norm(c, axis=1) + 1e-8))))
    for L in mods:
        print(f"layer {L}: relative displacement A {np.mean(disp['A'][L]):.3f}  B {np.mean(disp['B'][L]):.3f}  N {np.mean(disp['N'][L]):.3f}")
    print(f"captures done  [{time.time() - t0:.0f}s]")
    res = []

    def record(donor, arm, out):
        for i, r in enumerate(S.itertuples()):
            gen, lpx, lpz = out[i]
            nm, src = parse(gen, words)
            res.append(dict(donor=donor, arm=arm, scene=r.scene, Z=r.Z, q=r.q, generation=gen, name=nm, source=src, names_Z=nm == r.Z, lpz=lpz))
        d_ = [x for x in res if x["donor"] == donor and x["arm"] == arm]
        print(
            f"{donor} {arm:28s} names Z {np.mean([x['names_Z'] for x in d_]):.2f}  seen {np.mean([x['source'] == 'seen' for x in d_]):.2f}  "
            f"read {np.mean([x['source'] == 'read' for x in d_]):.2f}  [{time.time() - t0:.0f}s]"
        )

    record("-", "clean", generate_patched(lm, texts, img_rec, 8, mods, head_dim, {}, don["C"], ids_x, ids_z))
    record("A", "donor alone", generate_patched(lm, texts, img_a, 8, mods, head_dim, {}, don["A"], ids_x, ids_z))
    record("B", "donor alone", generate_patched(lm, texts, img_b, 8, mods, head_dim, {}, don["B"], ids_x, ids_z))
    for d in ("C", "A", "B", "N"):
        for g in groups:
            if d == "C" and g != groups[-1]:
                continue
            spec = {L: list(range(H)) for L in g}
            record(d, "all heads of " + "+".join(map(str, g)), generate_patched(lm, texts, img_rec, 8, mods, head_dim, spec, don[d], ids_x, ids_z))
    R = pd.DataFrame(res)
    run.save_csv(R, "source_cause.csv")
    sc = np.sort(R.scene.unique())
    C = R[R.arm == "clean"].set_index("scene").reindex(sc)
    out = dict(n=int(len(S)), groups=groups, heads=H, question=QUESTION, displacement={d: {str(L): float(np.mean(v)) for L, v in disp[d].items()} for d in disp}, arms={}, paired={})
    for (d, arm), g in R.groupby(["donor", "arm"], sort=False):
        g = g.set_index("scene").reindex(sc)
        key = f"{d}: {arm}"
        out["arms"][key] = dict(
            names_Z=boot_scene(g.names_Z.astype(float).values, sc),
            seen=boot_scene((g.source == "seen").astype(float).values, sc),
            read=boot_scene((g.source == "read").astype(float).values, sc),
            logp_Z=boot_scene(g.lpz.values, sc),
        )
        if arm == "clean":
            continue
        held = C.names_Z & g.names_Z
        out["paired"][key] = dict(
            answer_held_Z=int(held.sum()),
            held_report_read=float((g.source[held] == "read").mean()) if held.any() else None,
            held_seen_to_read=int(((C.source[held] == "seen") & (g.source[held] == "read")).sum()),
            held_read_to_seen=int(((C.source[held] == "read") & (g.source[held] == "seen")).sum()),
            same_as_clean=float(((g.name == C.name) & (g.source == C.source)).mean()),
        )
    a, b = out["arms"]["A: donor alone"], out["arms"]["B: donor alone"]
    c = out["arms"]["-: clean"]
    null_key = f"C: all heads of {'+'.join(map(str, groups[-1]))}"
    out["gates"] = dict(
        clean_Z_seen=bool(c["names_Z"][0] >= 0.85 and c["seen"][0] >= 0.85),
        donor_A_Z_read=bool(a["names_Z"][0] >= 0.85 and a["read"][0] >= 0.85),
        donor_B_Z_seen=bool(b["names_Z"][0] >= 0.85 and b["seen"][0] >= 0.85),
        null_patch_unchanged=bool(out["paired"][null_key]["same_as_clean"] >= 0.95),
        B_matched=bool(all(np.mean(disp["B"][L]) >= 0.5 * np.mean(disp["A"][L]) for L in mods)),
    )
    print("gates", out["gates"])
    for key, e in out["paired"].items():
        moved = f"seen->read {e['held_seen_to_read']}, read->seen {e['held_read_to_seen']}"
        print(f"{key:36s} answer held on {e['answer_held_Z']:2d}: read {e['held_report_read']}, {moved}, same as clean {e['same_as_clean']:.2f}")
    out["seconds"] = time.time() - t0
    run.save_json(out, "summary.json")


if __name__ == "__main__":
    main()
