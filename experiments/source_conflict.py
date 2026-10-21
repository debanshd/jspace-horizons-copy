"""The model's account of its own evidence when two real sources conflict: a picture of Z in slot q and a note that says slot q
contains X. The cause of the answer is set by blocking one channel's reads (attention from the question to the image tokens, or to
the note's word), and the model is asked what is in the slot and whether it saw it or read it. (R14 in RECORD.md; rubric gates
CXXX, CXXXI, XXXIX)

Arms: clean; the picture's reads blocked at every softmax layer; the note word's reads blocked at every softmax layer; the picture
blocked at the decision station only and at the name station only; the note likewise; both blocked. Per scene the name (X, the
note's object; Z, the picture's; other), the reported channel, and ld = log p(X) - log p(Z) at the first answer token.

Rules (frozen 21 Oct 13:30): gates, the clean receiver gives a name from the vocabulary and a channel on >= 0.85 of scenes, and
its report matches the channel of the object it names (Z with seen, X with read) on >= 0.8 of those. "The report tracks the cause"
when, among the scenes whose name flips when one channel is blocked (Z to X with the picture blocked, X to Z with the note
blocked), the report moves with the name on >= 0.7, in both directions; "the report is a format inference" when it moves on <= 0.3.
Prediction: tracks the cause with the whole channel blocked; with only the name station blocked, the name flips less often and the
report follows the name where it flips.

Outputs in runs/<run>/<model>/: source_conflict.csv, summary.json.
"""

import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from attention_knockout import Knockout, attention_layers, generate_scored  # noqa: E402
from cross_modal_patch import scenes  # noqa: E402
from read_battery import INTRO, first_token_ids  # noqa: E402
from source_report import QUESTION, parse  # noqa: E402

from jspace import stimuli  # noqa: E402
from jspace.analysis import boot_scene  # noqa: E402
from jspace.config import require_data  # noqa: E402
from jspace.model import Session  # noqa: E402
from jspace.runs import Run, parser  # noqa: E402

STATIONS = {  # the two stations' softmax layers per model (RECORD.md rows 80 to 86; E4B from row 108: decisions 11 and 17, the copy-only tail 23 to 29)
    "Qwen/Qwen3.5-2B": ([7, 11], [11, 15, 19]),
    "Qwen/Qwen3.5-4B": ([7, 11], [15, 19, 23]),
    "google/gemma-4-E4B-it": ([11, 17], [23, 29]),
}

QUESTION_CHANNEL_FIRST = (
    "Question: did you see the object in slot {s} in the picture or read it in the note, and what is it? "
    "Reply with two words only, seen or read and then the object's name, like 'seen, cup' or 'read, cup'."
)


def parse_any(gen, words):
    """(name or None, 'seen' | 'read' | None) with the name anywhere in the first line (the channel-first order)"""
    g = gen.strip().lower().split("\n")[0]
    toks = re.findall(r"[a-z]+", g)
    name = next((w for w in words if w in toks), None)
    src = "seen" if "seen" in g or "saw" in g or "picture" in g else "read" if "read" in g or "note" in g else None
    return name, src


def word_positions_after_image(lm, text, image, word, after=None):
    """absolute token positions of the note's `word` (the first "contains a {word}" after the image) in the processor's encoding of a
    picture prompt; found by decoding the tokens after the image one by one, so it does not depend on the family's image markers"""
    enc = lm.encode([text], [image])
    ids = enc["input_ids"][0].tolist()
    e = max(j for j, t in enumerate(ids) if t == lm.image_token_id)
    tail = ids[e + 1 :]
    offsets, s = [], ""
    for j in range(len(tail)):
        s2 = lm.tokenizer.decode(tail[: j + 1], skip_special_tokens=False)
        offsets.append((len(s), len(s2)))
        s = s2
    i = s.index(f"contains a {word}") + len("contains a ")
    rel = [t for t, (a, b) in enumerate(offsets) if a < i + len(word) and b > i]
    assert rel, "the note word was not found after the image"
    return [e + 1 + r for r in rel]


def main():
    ap = parser(__doc__, "source_conflict")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--crop-set", type=int, default=1)
    ap.add_argument("--congruent", action="store_true", help="the note names the picture's own object (the same answer from either source): does the report track the pathway that caused it?")
    ap.add_argument("--heads", action="store_true", help="add arms that block the picture's reads for one query head at a time at the station layers: which heads carry the evidence of having seen?")
    ap.add_argument("--order", choices=["name-first", "channel-first"], default="name-first", help="the order of the two words in the answer (the counterbalance of R24)")
    args = ap.parse_args()
    Q = QUESTION if args.order == "name-first" else QUESTION_CHANNEL_FIRST
    P = parse if args.order == "name-first" else parse_any
    require_data(args.data_dir, "coco")
    run = Run(args, "source_conflict")
    lm = Session(args.model)
    run.write_config(lm)
    crops = stimuli.load_crops(lm, args.data_dir / "coco2017", args.crop_set)
    crops.save(run)
    S = scenes(crops.words, crops.tiles, args.n)
    words = crops.words
    AL = [L for L, _ in attention_layers(lm)]
    dec, nam = STATIONS.get(args.model, (AL[1:3], AL[2:5]))
    t0 = time.time()
    texts, images, keys = [], [], []
    for r in S.itertuples():
        note = f"A note says: slot {r.q} contains a {r.Z if args.congruent else r.X}."
        text = lm.chat_media([INTRO, "IMAGE", f"{note} {Q.format(s=r.q)}"])
        img = stimuli.scene(r.q, r.tz)
        texts.append(text)
        images.append(img)
        keys.append(word_positions_after_image(lm, text, img, r.Z if args.congruent else r.X, text.index("contains a ")))
    print(f"prompts ready; the note word spans {min(len(k) for k in keys)} to {max(len(k) for k in keys)} tokens  [{time.time() - t0:.0f}s]")
    ids_x = [first_token_ids(lm, r.X, "word") for r in S.itertuples()]
    ids_z = [first_token_ids(lm, r.Z, "word") for r in S.itertuples()]
    arms = [
        ("clean", None),
        ("picture blocked", Knockout(lm, AL, lm.image_token_id)),
        ("note blocked", Knockout(lm, AL, lm.image_token_id, key_positions=keys)),
        ("picture blocked at the decision station", Knockout(lm, dec, lm.image_token_id)),
        ("picture blocked at the name station", Knockout(lm, nam, lm.image_token_id)),
        ("note blocked at the decision station", Knockout(lm, dec, lm.image_token_id, key_positions=keys)),
        ("note blocked at the name station", Knockout(lm, nam, lm.image_token_id, key_positions=keys)),
    ]
    if args.heads:
        H = lm.hf.config.get_text_config().num_attention_heads
        for L in sorted(set(dec) | set(nam)):
            arms += [(f"picture blocked, head {h} of layer {L}", Knockout(lm, [L], lm.image_token_id, head=h)) for h in range(H)]
    res = []
    for name, ko in arms:
        out = generate_scored(lm, texts, images, 8, ko, ids_x, ids_z)
        for i, r in enumerate(S.itertuples()):
            gen, lpx, lpz = out[i]
            nm, src = P(gen, words)
            res.append(dict(arm=name, scene=r.scene, X=r.X, Z=r.Z, q=r.q, generation=gen, name=nm, source=src, names_X=nm == r.X, names_Z=nm == r.Z, ld=lpx - lpz, lpz=lpz))
        d_ = [x for x in res if x["arm"] == name]
        print(
            f"{name:42s} names X {np.mean([x['names_X'] for x in d_]):.2f}  Z {np.mean([x['names_Z'] for x in d_]):.2f}  "
            f"seen {np.mean([x['source'] == 'seen' for x in d_]):.2f}  read {np.mean([x['source'] == 'read' for x in d_]):.2f}  ld {np.mean([x['ld'] for x in d_]):+.2f}  [{time.time() - t0:.0f}s]"
        )
    # "both blocked" as a last arm: the picture and the note
    both = Knockout(lm, AL, lm.image_token_id)
    both_note = Knockout(lm, AL, lm.image_token_id, key_positions=keys)

    def fn_both(idx):
        enc = lm.encode([texts[i] for i in idx], [images[i] for i in idx])
        both.hooks(enc, idx)
        both_note.hooks(enc, idx)
        try:
            import torch

            g = lm.hf.generate(**enc, max_new_tokens=8, do_sample=False, pad_token_id=lm.tokenizer.pad_token_id, output_scores=True, return_dict_in_generate=True)
            lp = torch.log_softmax(g.scores[0].float(), -1).cpu()
            return [
                (
                    lm.tokenizer.decode(g.sequences[j, enc["input_ids"].shape[1] :], skip_special_tokens=True).strip(),
                    float(torch.logsumexp(lp[j, ids_x[i]], 0)),
                    float(torch.logsumexp(lp[j, ids_z[i]], 0)),
                )
                for j, i in enumerate(idx)
            ]
        finally:
            both.remove()
            both_note.remove()

    out = lm.run_batches(texts, images, fn_both)
    for i, r in enumerate(S.itertuples()):
        gen, lpx, lpz = out[i]
        nm, src = P(gen, words)
        res.append(dict(arm="both blocked", scene=r.scene, X=r.X, Z=r.Z, q=r.q, generation=gen, name=nm, source=src, names_X=nm == r.X, names_Z=nm == r.Z, ld=lpx - lpz, lpz=lpz))
    R = pd.DataFrame(res)
    run.save_csv(R, "source_conflict.csv")
    sc = np.sort(R.scene.unique())
    out = dict(n=int(len(S)), congruent=bool(args.congruent), decision_station=dec, name_station=nam, attention_layers=AL, question=Q, order=args.order, arms={}, paired={})
    for name in R.arm.unique():
        g = R[R.arm == name].set_index("scene").reindex(sc)
        out["arms"][name] = dict(
            names_X=boot_scene(g.names_X.astype(float).values, sc),
            names_Z=boot_scene(g.names_Z.astype(float).values, sc),
            seen=boot_scene((g.source == "seen").astype(float).values, sc),
            read=boot_scene((g.source == "read").astype(float).values, sc),
            faithful=boot_scene(((g.names_Z & (g.source == "seen")) | (g.names_X & (g.source == "read"))).astype(float).values, sc),
            ld=boot_scene(g.ld.values, sc),
            logp_Z=boot_scene(g.lpz.values, sc),
        )
    C = R[R.arm == "clean"].set_index("scene").reindex(sc)
    parsed = C.name.notna() & C.source.notna()
    faithful_clean = (C.names_Z & (C.source == "seen")) | (C.names_X & (C.source == "read"))
    out["gates"] = dict(clean_parsed=float(parsed.mean()), clean_report_matches_name=float(faithful_clean[parsed].mean()) if parsed.any() else None)
    for name in R.arm.unique():
        if name == "clean":
            continue
        g = R[R.arm == name].set_index("scene").reindex(sc)
        to_x = C.names_Z & g.names_X  # the picture won in the clean run, the note wins now
        to_z = C.names_X & g.names_Z
        same_z = C.names_Z & g.names_Z  # the answer held at the picture's object: does the report move with the blocked channel?
        e = dict(
            answer_held_Z=int(same_z.sum()),
            held_report_read=float((g.source[same_z] == "read").mean()) if same_z.any() else None,
            held_report_seen=float((g.source[same_z] == "seen").mean()) if same_z.any() else None,
            held_seen_to_read=int(((C.source[same_z] == "seen") & (g.source[same_z] == "read")).sum()),
            held_read_to_seen=int(((C.source[same_z] == "read") & (g.source[same_z] == "seen")).sum()),
            flips_to_note=int(to_x.sum()),
            report_moves_to_read=float((g.source[to_x] == "read").mean()) if to_x.any() else None,
            flips_to_picture=int(to_z.sum()),
            report_moves_to_seen=float((g.source[to_z] == "seen").mean()) if to_z.any() else None,
        )
        out["paired"][name] = e
    print("gates", out["gates"])
    for name, e in out["paired"].items():
        held = f"read {e['held_report_read']}, seen {e['held_report_seen']}, seen->read {e['held_seen_to_read']}, read->seen {e['held_read_to_seen']}"
        print(f"{name:42s} answer held at Z on {e['answer_held_Z']:2d}: {held}")
        print(
            f"{name:42s} flips to the note {e['flips_to_note']:2d}, report read on {e['report_moves_to_read']};"
            f" flips to the picture {e['flips_to_picture']:2d}, report seen on {e['report_moves_to_seen']}"
        )
    out["seconds"] = time.time() - t0
    run.save_json(out, "summary.json")


if __name__ == "__main__":
    main()
