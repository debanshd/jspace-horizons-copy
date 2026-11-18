"""Media patching: until which layer is a video's object still named, and still used? (plan.md part 2: depth_sweep for video)

Design. The 64 VGGSound clips of a finished media_steer run (--gate-run), those every model there named, each shown as a true
video and as its static version (the middle frame 8 times): the inputs clip and static. For each receiver clip (class X) the
donor is the clip of class Y, the other kind, at the same position in its class (media.assign's Y), and a second donor the clip
of class Z, the same kind. The donor's media-token states, recorded from its own
clean prompt (the video comes before the question, so these states do not depend on the question), replace the receiver's at
one window of 4 consecutive layers at a time, from the input (layer -1) to the last layer, as depth_sweep does for pictures. Two
questions in separate prompts: say (the noun) and the animal rule (7 or 3). Arms per receiver and question: clean; the
receiver's own states written back at every layer (the machinery check); the other-kind donor at every layer (the positive
control); and per window, the other-kind donor and the same-kind donor. No lens is used.

Clips. They come from runs/<gate-run>/*/gate.csv by media_steer's rule: per class the first 8 candidates every model named.

The rule (written before the run; depth_sweep's rules, with the same-kind donor in the place of its random steer):
- Gate: a question on an input is read only if clean accuracy is >= 0.90 (say) or >= 0.85 (rule); otherwise "not perceived,
  untestable".
- Machinery: the self-patch must give the clean answer on >= 0.95 of rows, else the input reads "patch machinery failed".
- Positive control: the every-layer other-kind patch makes say follow the donor on >= 0.9 of clips, else "positive control
  failed". Either failure leaves the input's horizons unread.
- Horizons: L_say is the last window start at which say follows the donor on >= 0.5 of clips. L_use is the last window start
  at which the rule follows the donor at least 0.3 more than it flips under the same-kind donor in that window.
- Verdict (depth_sweep's): "too late" when L_use <= L_say - 8 (two windows), "together" when both exist otherwise, "never"
  when only L_say exists, "nothing moves" when neither.
- Share of depth: a horizon's share is (window start + 4) / number of layers, the depth at which its window ends.
- Intervals: 95% bootstrap over clips, 2000 resamples, seed 0.

Predictions (written before the run, from the image patching runs in docs/REPORT.md and results/depth_sweep):
P1 the verdict is "too late" (the rule stops reading the video at least two windows before naming does);
P2 the rule's horizon share is within the family's image range widened by one window (Qwen 0.26-0.33, Gemma 0.38-0.48);
P3 naming's horizon share is within 0.67-0.75 widened by one window; on a model whose last layers reuse earlier layers' keys and
   values (Gemma-4 E2B and E4B) no later layer can read a media token, so naming's share is at most that limit plus one window;
P4 the true video and its static version have the same horizons within one window, for both questions.

Outputs in runs/<run>/<model>/: media_patch.csv (every row), summary.json (gates, controls, curves, horizons, verdicts,
predictions, setup), clips.json, stimuli.png (the first two clips as the model sees them), gpu.csv, config.json, run.log.
"""

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from media_steer import N_NEW, encode, make_row, media_token, pick  # noqa: E402

from jspace import batched, media, stimuli  # noqa: E402
from jspace.analysis import boot_scene  # noqa: E402
from jspace.config import require_data  # noqa: E402
from jspace.model import Session  # noqa: E402
from jspace.runs import Run, parser  # noqa: E402

WINDOW = 4
GATE = {"say": 0.90, "animal": 0.85}
VGG_INPUTS = ["clip", "static"]
FAMILY_RULE_RANGE = {"qwen": (0.26, 0.33), "gemma": (0.38, 0.48)}
NAME_RANGE = (0.67, 0.75)


def windows(n_layers):
    return [list(range(a, min(a + WINDOW, n_layers))) for a in range(-1, n_layers, WINDOW)]


def donor_scene(r, word):
    """the scene whose clip is of class `word` at the same position in its class (scene = 8 k + class index)"""
    k = r["scene"] // len(media.CLASSES)
    return k * len(media.CLASSES) + media.CLASSES.index(r[word])


def patch_rows(inputs, S, ID, n_layers):
    rows = []
    every = tuple(range(-1, n_layers))
    for inp in inputs:
        for task in media.TASKS:
            for r in S:
                rows.append(make_row(inp, task, r, ID, donor=None))
                rows.append(make_row(inp, task, r, ID, family="self", arm="own states, every layer", site="patch", layers=every, donor=r["scene"]))
                rows.append(make_row(inp, task, r, ID, family="all", arm="other-kind donor, every layer", site="patch", layers=every, donor=donor_scene(r, "Y")))
                for w in windows(n_layers):
                    for fam, word in (("other", "Y"), ("same", "Z")):
                        rows.append(make_row(inp, task, r, ID, family=fam, arm=f"{fam}-kind donor, layers {w[0]} to {w[-1]}", site="patch", layers=tuple(w), set=str(w[0]), donor=donor_scene(r, word)))
    return rows


def kv_limit(lm):
    """the share of depth after which no layer computes its own keys and values (Gemma-4 E2B and E4B share them), or None"""
    shared = int(getattr(lm.hf.config.get_text_config(), "num_kv_shared_layers", 0) or 0)
    n = lm.final + 1
    return (n - shared) / n if shared else None


def summarize(R, inputs, n_layers, family, kv):
    scenes = np.sort(R.scene.unique())
    boot = lambda v: boot_scene(v, scenes)  # noqa: E731
    starts = [w[0] for w in windows(n_layers)]
    out = dict(n=int(len(scenes)), n_layers=int(n_layers), windows=starts, gate={}, controls={}, curves={}, horizon={}, verdict={}, share={}, predictions={})
    for inp in inputs:
        Ri = R[R.input == inp]
        cl = {t: pick(Ri, scenes, "correct", task=t, family="clean") for t in media.TASKS}
        out["gate"][inp] = {t: dict(clean=float(cl[t].mean()), passes=bool(cl[t].mean() >= GATE[t])) for t in media.TASKS}
        same_as_clean = float(np.mean([(answers(Ri, scenes, t, "self") == answers(Ri, scenes, t, "clean")).mean() for t in media.TASKS]))
        pos = boot(pick(Ri, scenes, "follows_Y", task="say", family="all"))
        out["controls"][inp] = dict(self_patch_same=same_as_clean, machinery_ok=bool(same_as_clean >= 0.95), every_layer_say_follows_donor=pos, positive_ok=bool(pos[0] >= 0.9))
        cur = {}
        for t in media.TASKS:
            cur[f"{t} follows donor"] = [boot(pick(Ri, scenes, "follows_Y", task=t, family="other", set=str(s))) for s in starts]
        cur["rule flips, same-kind donor"] = [boot(1 - pick(Ri, scenes, "correct", task="animal", family="same", set=str(s))) for s in starts]
        cur["say follows same-kind donor"] = [boot(pick(Ri, scenes, "follows_Z", task="say", family="same", set=str(s))) for s in starts]
        out["curves"][inp] = cur
        ok = out["gate"][inp]["say"]["passes"] and out["controls"][inp]["machinery_ok"] and out["controls"][inp]["positive_ok"]
        if not ok:
            why = "not perceived, untestable" if not out["gate"][inp]["say"]["passes"] else "patch machinery failed" if not out["controls"][inp]["machinery_ok"] else "positive control failed"
            out["horizon"][inp], out["verdict"][inp] = why, why
            continue
        say_ok = [s for s, v in zip(starts, cur["say follows donor"]) if v[0] >= 0.5]
        use_ok = [s for s, v, f in zip(starts, cur["animal follows donor"], cur["rule flips, same-kind donor"]) if v[0] >= f[0] + 0.3] if out["gate"][inp]["animal"]["passes"] else []
        Ls, Lu = (max(say_ok) if say_ok else None), (max(use_ok) if use_ok else None)
        out["horizon"][inp] = dict(L_say=Ls, L_use=Lu, say_windows=say_ok, use_windows=use_ok, rule_read=bool(out["gate"][inp]["animal"]["passes"]))
        out["verdict"][inp] = verdict(Ls, Lu)
        out["share"][inp] = dict(say=None if Ls is None else (Ls + WINDOW) / n_layers, use=None if Lu is None else (Lu + WINDOW) / n_layers)
    out["predictions"] = predictions(out, inputs, n_layers, family, kv)
    return out


def answers(Ri, scenes, task, family):
    """the answer text per clip, in scene order (pick() reads numbers)"""
    s = Ri[(Ri["task"] == task) & (Ri["family"] == family)].set_index("scene")["generation"].astype(str)
    assert not s.index.duplicated().any(), (task, family)
    return s.reindex(scenes).to_numpy()


def verdict(Ls, Lu):
    if Ls is not None and Lu is None:
        return "never"
    if Ls is not None and Lu is not None:
        return "too late" if Lu <= Ls - 2 * WINDOW else "together"
    return "nothing moves"


def within(x, lo, hi, w):
    return x is not None and lo - w <= x <= hi + w


def same_horizons(h1, h2):
    return isinstance(h1, dict) and isinstance(h2, dict) and all(h1[k] is not None and h2[k] is not None and abs(h1[k] - h2[k]) <= WINDOW for k in ("L_say", "L_use"))


def predictions(out, inputs, n_layers, family, kv):
    w = WINDOW / n_layers
    P = {}
    for inp in inputs:
        h = out["horizon"][inp]
        if not isinstance(h, dict):
            P[inp] = {k: "untestable" for k in ("P1", "P2", "P3", "P4")}
            continue
        sh = out["share"][inp]
        p = {}
        p["P1"] = ("kept" if out["verdict"][inp] == "too late" else "missed") if h["rule_read"] else "untestable"
        p["P2"] = ("kept" if within(sh["use"], *FAMILY_RULE_RANGE[family], w) else "missed") if h["rule_read"] else "untestable"
        name_ok = (sh["say"] is not None and sh["say"] <= kv + w) if kv else within(sh["say"], *NAME_RANGE, w)
        p["P3"] = "kept" if name_ok else "missed"
        twin = {"clip": "static", "static": "clip"}[inp]
        p["P4"] = ("kept" if same_horizons(h, out["horizon"].get(twin)) else "missed") if twin in inputs and isinstance(out["horizon"].get(twin), dict) else "untestable"
        P[inp] = p
    return P


def to_frame(rows, res):
    recs = []
    for r, (text, lx, ly) in zip(rows, res):
        recs.append(
            dict(
                {k: v for k, v in r.items() if k not in ("ids_x", "ids_y", "ko", "layers")},
                layers=f"{r['layers'][0]}..{r['layers'][-1]}" if r["layers"] else "",
                generation=text,
                correct=media.answer_matches(r["task"], text, media.truth(r["task"], r["X"])),
                follows_Y=media.answer_matches(r["task"], text, media.truth(r["task"], r["Y"])),
                follows_Z=media.answer_matches(r["task"], text, media.truth(r["task"], r["Z"])),
                lp_X=lx,
                lp_Y=ly,
                ld=lx - ly,
            )
        )
    return pd.DataFrame(recs)


def main():
    ap = parser(__doc__, "media_patch")
    ap.add_argument("--gate-run", default="media_steer_all", help="the media_steer run whose gate.csv files choose the VGGSound clips")
    ap.add_argument("--inputs", default=",".join(VGG_INPUTS))
    ap.add_argument("--max-rows", type=int, default=512)
    ap.add_argument("--no-encoder-cache", action="store_true")
    args = ap.parse_args()
    require_data(args.data_dir, "vggsound")
    vman = media.manifest(args.data_dir)
    run = Run(args, "media_patch")
    lm = Session(args.model)
    run.write_config(lm)
    t0 = time.time()
    mon = batched.GpuMonitor(run.path("gpu.csv"))
    ID = {w: lm.single_ids([w, " " + w, w.capitalize(), " " + w.capitalize()]) for w in media.CLASSES}
    ID.update({d: lm.single_ids([d, " " + d]) for d in ("7", "3")})
    assert all(ID.values()), f"a class word or digit has no single-token form in {lm.name}'s tokenizer"
    inputs = args.inputs.split(",")
    token = {inp: media_token(lm, inp) for inp in inputs}
    cache = {}

    def frames_of(cid):
        if cid not in cache:
            cache[cid] = media.load_clip(args.data_dir, cid)
        return cache[cid]

    # ---- the clips
    gate_files = sorted((args.runs_dir / args.gate_run).glob("*/gate.csv"))
    if not gate_files:
        raise SystemExit(f"no gate.csv under {args.runs_dir / args.gate_run}: the VGGSound clips come from a finished media_steer run")
    passing = set.intersection(*[set(pd.read_csv(p)["clip"][pd.read_csv(p)["correct"]]) for p in gate_files])
    vok = {w: [c["id"] for c in vman["clips"] if c["class"] == w and c["id"] in passing] for w in media.CLASSES}
    short = {w: len(v) for w, v in vok.items() if len(v) < 8}
    if short:
        raise SystemExit(f"the gate run leaves fewer than 8 VGGSound clips in {short}")
    S = media.assign(vok, 8)
    run.save_json(
        dict(scenes=S, gate_files=[str(p) for p in gate_files], manifest_sha256=hashlib.sha256((args.data_dir / "vggsound/manifest.json").read_bytes()).hexdigest()),
        "clips.json",
    )
    stimuli.contact_sheet(
        [(f"{r['clip']}: {media.FRAME_TIMES[j]:.1f} s", Image.fromarray(frames_of(r["clip"])[0][j])) for r in S[:2] for j in range(media.N_FRAMES)],
        media.N_FRAMES,
        run.path("stimuli.png"),
    )

    n_layers = lm.final + 1
    LF = list(range(-1, n_layers))  # no lens is needed: every layer's state is patched as it is
    family = "gemma" if "gemma" in lm.name.lower() else "qwen"
    kv = kv_limit(lm)
    print(f"{n_layers} layers, windows {[w[0] for w in windows(n_layers)]}; key-value sharing limit {kv}; inputs {inputs}")
    T = batched.Tables()
    R_all, setup = [], dict(cache={}, batch_rows={}, layout={})
    for inp in inputs:  # one input at a time: its donor states (every layer, every clip) are freed before the next
        encs = {(inp, t, r["scene"]): encode(lm, inp, t, [frames_of(r["clip"])]) for t in media.TASKS for r in S}
        for t in media.TASKS:
            lengths = {int(encs[(inp, t, r["scene"])]["input_ids"].shape[1]) for r in S}
            assert len(lengths) == 1, f"{inp}/{t}: prompts differ in length {lengths}"
        n_media = int((encs[(inp, "say", S[0]["scene"])]["input_ids"] == token[inp]).sum())
        setup["layout"][inp] = dict(tokens=lengths.pop(), media_tokens=n_media)
        # one token budget for the whole run (as video_sweep): the first engine's, rounded down to 2,048 tokens, so the batch
        # sizes do not move with the memory the allocator holds when the next input's engine starts
        eng = batched.Engine(lm, encs, token, media.MEDIUM, T, lm.layers, args.max_rows, budget_tokens=setup.get("budget_tokens"))
        eng.budget = setup.setdefault("budget_tokens", max(2048, eng.budget // 2048 * 2048))
        if not args.no_encoder_cache and eng.build_encoder_cache(inp, [r["scene"] for r in S]):
            setup["cache"][inp] = eng.check_encoder_cache(inp, [make_row(inp, "say", r, ID, donor=None) for r in S[:4]], N_NEW)
        n_sc = max(r["scene"] for r in S) + 1
        gb = n_sc * n_media * lm.d_model * len(LF) * 2 / 2**30
        # pinned memory speeds the copy to the GPU, but pinning tens of GB can fail; above 8 GB the states stay pageable
        T.donor[inp] = {L: (lambda t: t.pin_memory() if gb <= 8 else t)(torch.zeros(n_sc, n_media, lm.d_model, dtype=torch.float16)) for L in LF}

        def on_layer(L, rows, v, inp=inp):
            for j, r in enumerate(rows):
                T.donor[inp][L][r["scene"]] = v[j].half().cpu()

        eng.capture([make_row(inp, "say", r, ID, donor=None) for r in S], LF, on_layer)
        print(f"{inp}: donor states of {len(S)} clips at {len(LF)} layers ({gb:.1f} GB)  [{time.time() - t0:.0f}s]")
        rows = patch_rows([inp], S, ID, n_layers)
        res = eng.generate(rows, N_NEW, f"{inp}: {len(rows)} rows")
        R_all.append(to_frame(rows, res))
        setup["batch_rows"].update({f"{k[0]}/{k[1]}": b for k, b in eng.cap.items()})
        del T.donor[inp]
        torch.cuda.empty_cache()
    R = pd.concat(R_all, ignore_index=True)
    run.save_csv(R, "media_patch.csv")
    gpu = mon.stop()
    out = summarize(R, inputs, n_layers, family, kv)
    out["setup"] = dict(setup, inputs=inputs, kv_limit=kv, family=family, gpu=gpu, seconds=time.time() - t0)
    run.save_json(out, "summary.json")
    for inp in inputs:
        c = out["controls"][inp]
        gates = json.dumps({t: round(v["clean"], 2) for t, v in out["gate"][inp].items()})
        print(f"{inp}: gate {gates}, self-patch {c['self_patch_same']:.2f}, every-layer patch {c['every_layer_say_follows_donor'][0]:.2f}")
        print(f"   horizons {out['horizon'][inp]}, verdict {out['verdict'][inp]}, predictions {out['predictions'][inp]}")
    if gpu:
        print(f"GPU utilization: mean {gpu['mean_utilization']}%, median {gpu['median_utilization']}% over {gpu['samples']} samples")


if __name__ == "__main__":
    main()
