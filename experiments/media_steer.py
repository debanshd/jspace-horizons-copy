"""Media steering: from which layers does a word's lens direction, written on a video's or a sound's tokens, get said? (plan.md v3)

Design. 64 VGGSound clips (8 classes by 8: dog, cat, bird, horse, car, bus, train, airplane), each shown four ways: the clip
(8 frames as a video), the static video (its middle frame 8 times), the middle frame as a picture, and its sound (Gemma-4 E2B
and E4B only). Two questions in separate prompts: say (the noun) and the animal rule (7 or 3). Per clip, Y is a word of the
other kind and Z of the same kind (media.assign). Three arms, mixed in the same batches (jspace.batched):
  steering   a direction at dose 1, 2, 4 or 8 sigma_L at every layer of a block (early up to 1/3 of depth, middle to 0.7, late
             after), on every media token: Y's J-lens direction, Z's, a random unit direction, Y's plain unembedding direction,
             the clip's top principal direction (the energy control); and at the last prompt token Y's J-lens and unembedding
             directions (the output-push control). Then, at the calibrated dose, prefix sweeps (-1 to L) and suffix sweeps (L to
             the last layer) for L = -1, 3, 7, ..., with Y and the random direction.
  knockout   attention from every text token to the media tokens blocked at every attention layer from L on, and up to L
             (Qwen3.5: its one-in-four softmax layers; Gemma-4: every fourth layer as L), on clean prompts.
  ablation   at the read layers, mean-ablated on the media tokens: the clip's own name's J-lens direction, Y's J-lens direction
             (name_steering's control), a random direction, the lens part (top r = 0.1 d right singular directions of W_U J_L;
             256 on the 4B), a random part of rank r.
sigma_L is the mean size of a media token's own-name component at layer L on the clean clips, per input kind. The J-lens map of
layer -1 is layer 0's and of the last layer the one before it (the nearest fitted map, as in depth_sweep).

Clips. scripts/prepare_media.py draws 16 candidates per class and decodes them. `--stage gate` asks every candidate's clip "What
is the object in the video?" and writes gate.csv; `--stage main` keeps, per class, the first 8 candidates that every model named
in --gate-models named correctly (plan.md: every model in the run), and stops if a class has fewer. Scenes are ordered round-robin
over the classes; the first 16 (two per class) are the calibration split, the other 48 carry the verdicts. `--stage smoke` runs
every arm on the first clip of five classes that this model names, and reports counts only. `--stage reference` is the smoke
check on the middle frame alone (plan.md step 5): it must reproduce RECORD row 93 on the 4B, and exits with status 1 if it does
not: at 4 sigma, say follows Y on at least 4 of the 5 clips from the early and from the middle block, and the random direction
on at most 1 of 5 in each.

Controls. The random, unembedding and principal directions and Z for the steering; the last-prompt-token site for an output
push; the static video and the frame for motion; the random direction and random part for the ablation; clean prompts for all.

Predictions (written before the run, plan.md): 1 naming follows Y from the early and middle blocks on >= 0.8 of held-out clips;
2 not from the late block (<= 0.1); 3 the random, unembedding and principal directions on the media tokens move <= 0.1; 4 the
rule follows Y more from the early block than the middle, and below 0.5; 5 the knockout's last naming read is at 0.67 to 0.75
of depth, and the latest suffix start that steers naming is at or before that read and at most one step (4 layers) before it; 6 removing the own name's direction or
the lens part costs naming <= 0.10; 7 the clip, the static video and the frame agree in every block; 8 the sound agrees with the
frame.

The rule (written before the run; thresholds as in name_steering R0 and R4 and RECORD row 93):
- Gate: a question on an input is read only if clean accuracy is >= 0.90 (say) or >= 0.85 (rule); otherwise "not perceived,
  untestable", never "no pathway". (The clip's say is the selection question, so it passes by construction.)
- The calibrated dose of an input: the smallest dose at which say follows Y on >= 0.5 of the calibration clips with Y on the media
  tokens in the middle block; if none, the sweeps run at 8 and the input is marked "no calibrated dose".
- A layer set steers naming at that dose: say follows Y on >= 0.5 of the held-out clips and the lower bound is above the random
  direction's upper bound.
- Seen: the rule follows Y at least 0.30 more than with the random direction, interval above 0, and Z flips the rule on <= 0.10.
  Only said: at most 0.10 more. Otherwise undetermined.
- J-specific: the unembedding direction on the media tokens makes say follow Y on <= 0.10.
- Not told apart from an output push: every block that steers naming from the media tokens also does from the last prompt token,
  with the unembedding direction there following Y on >= 0.5.
- Same as the picture: per block, the held-out difference in say-follows-Y against the frame input (paired by clip, at the frame's
  calibrated dose) has its interval inside +-0.15; otherwise "differs by input". Nothing temporal tested: the clip and the static
  video are the same by this test in every block.
- Ablation (name_steering R1, per question): D = accuracy with Y's direction removed minus with the own name's removed;
  "necessary" when D >= 0.25 with the interval above 0, "not needed" when D <= 0.10 with the interval inside (-0.15, 0.15),
  otherwise undetermined. R1_nats: log p of the correct first token with the own name removed minus with Y's removed;
  "necessary" at <= -0.5 with the interval below 0, "not needed" at >= -0.2 with the interval inside (-0.3, 0.3).
- Knockout: a cost is clean minus knocked-out log p(X's answer) - log p(Y's answer); the last reader is the largest L whose
  from-L-on knockout costs >= 1 nat with the interval above 0 (attention_knockout.py's rule).
- Intervals: 95% bootstrap over clips, 2000 resamples, seed 0.

Outputs in runs/<run>/<model>/: media_steer.csv (every row: input, question, clip, arm, the answer, correct, follows Y and Z,
first-token log-probabilities), summary.json (setup, gates, doses, steering tables, verdicts, knockout, ablation, sweeps, lens
readout, predictions), clips.json, stimuli.png, gpu.csv (nvidia-smi every 2 s), config.json, run.log; gate.csv for --stage gate.
"""

import hashlib
import json
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image

from jspace import batched, lens, media, stimuli
from jspace.analysis import boot_scene
from jspace.config import require_data
from jspace.model import Session
from jspace.runs import Run, parser

DOSES = [1.0, 2.0, 4.0, 8.0]
STEER_KEYS = ["Y", "Z", "random", "WU_Y", "PC"]
LAST_KEYS = ["Y", "WU_Y"]
ABLATIONS = ["own name", "other name", "random direction", "lens part", "random part"]
N_NEW = 4
GATE = {"say": 0.90, "animal": 0.85}


# ---- prompts


def prompt_text(lm, inp, task):
    med = media.MEDIUM[inp]
    q = media.QUESTIONS[(med, task)]
    part = {"video": {"type": "video"}, "picture": {"type": "image"}, "sound": {"type": "audio"}}[med]
    content = [{"type": "text", "text": q}, part] if med == "sound" else [part, {"type": "text", "text": q}]  # Gemma's card: audio after text
    return lm.chat_messages([{"role": "user", "content": content}])


def encode(lm, inp, task, items):
    """the processor on prompts of one input kind and question; items: [(frames, audio)]. Returns a dict of CPU tensors."""
    from transformers.video_utils import VideoMetadata

    med = media.MEDIUM[inp]
    text = [prompt_text(lm, inp, task)] * len(items)
    ms = [media.media_of(inp, f, a) for f, a in items]
    if med == "video":
        kw = dict(videos=ms, video_metadata=[VideoMetadata(**media.frame_metadata()) for _ in ms], do_sample_frames=False)
        if type(lm.processor).__name__.startswith("Qwen"):  # 224-pixel frames give 4 x 14 x 14 patches either way; False keeps 5.19's behaviour and its warning off
            kw["cap_pixels_per_frame"] = False
    elif med == "picture":
        kw = dict(images=[[m] for m in ms] if type(lm.processor).__name__.startswith("Gemma4") else ms)
    else:
        kw = dict(audio=ms)
    enc = lm.processor(text=text, return_tensors="pt", **kw)
    return {k: v for k, v in enc.items()}


def media_token(lm, inp):
    med = media.MEDIUM[inp]
    p = lm.processor
    tid = {"video": getattr(p, "video_token_id", None), "picture": getattr(p, "image_token_id", lm.image_token_id), "sound": getattr(p, "audio_token_id", None)}[med]
    assert tid is not None, f"{lm.name}: no token id for {med}"
    return int(tid)


def has_audio(lm):
    return getattr(lm.hf.config, "audio_config", None) is not None and getattr(lm.processor, "audio_token", None) is not None


# ---- the rows of a job


def make_row(inp, task, r, ID, **kw):
    tx, ty = media.truth(task, r["X"]), media.truth(task, r["Y"])
    row = dict(
        input=inp,
        task=task,
        scene=r["scene"],
        X=r["X"],
        Y=r["Y"],
        Z=r["Z"],
        clip=r["clip"],
        family="clean",
        arm="clean",
        site="",
        key="",
        set="",
        layers=(),
        dose=0.0,
        ablate="",
        ko=(),
        ko_kind="",
        ko_L=None,
    )
    row.update(kw)
    row["ids_x"], row["ids_y"] = ID[tx], ID[ty]
    return row


def phase_a(inputs, S, ID, BL, KO_STARTS, AL):
    rows = []
    for inp in inputs:
        for task in media.TASKS:
            for r in S:
                rows.append(make_row(inp, task, r, ID))
                for bn, layers in BL.items():
                    for dose in DOSES:
                        for key in STEER_KEYS:
                            rows.append(
                                make_row(inp, task, r, ID, family="steer", arm=f"steer {key} at the media tokens, {bn}, x{dose:g}", site="media", key=key, set=bn, layers=tuple(layers), dose=dose)
                            )
                        for key in LAST_KEYS:
                            rows.append(
                                make_row(inp, task, r, ID, family="steer", arm=f"steer {key} at the last token, {bn}, x{dose:g}", site="last", key=key, set=bn, layers=tuple(layers), dose=dose)
                            )
                for L in KO_STARTS:
                    rows.append(make_row(inp, task, r, ID, family="knockout", arm=f"block layers {L} on", ko=tuple(M for M in AL if M >= L), ko_kind="from", ko_L=L))
                    rows.append(make_row(inp, task, r, ID, family="knockout", arm=f"block layers up to {L}", ko=tuple(M for M in AL if M <= L), ko_kind="upto", ko_L=L))
                for a in ABLATIONS:
                    rows.append(make_row(inp, task, r, ID, family="ablate", arm=f"remove {a}", ablate=a))
    return rows


def phase_b(inputs, S, ID, starts, n_layers, dose_of):
    rows = []
    for inp in inputs:
        for task in media.TASKS:
            for r in S:
                for L in starts:
                    for kind, layers in (("prefix", range(-1, L + 1)), ("suffix", range(L, n_layers))):
                        for key in ("Y", "random"):
                            rows.append(make_row(inp, task, r, ID, family="sweep", arm=f"steer {key}, {kind} {L}", site="media", key=key, set=f"{kind} {L}", layers=tuple(layers), dose=dose_of[inp]))
    return rows


# ---- the summary (the rules above)


def pick(R, scenes, col, **eq):
    d = R
    for k, v in eq.items():
        d = d[d[k] == v]
    s = d.set_index("scene")[col].astype(float)
    assert not s.index.duplicated().any(), eq
    v = s.reindex(scenes).values
    assert not np.isnan(v).any(), f"missing rows for {eq}"
    return v


def summarize(R, S, inputs, BL, starts, AL, KO_STARTS, n_layers, n_cal, smoke):
    scenes_all = np.array([r["scene"] for r in S])
    cal = scenes_all[scenes_all < n_cal] if not smoke else scenes_all
    held = scenes_all[scenes_all >= n_cal] if not smoke else scenes_all
    boot = lambda v, sc: boot_scene(v, sc)  # noqa: E731
    out = dict(n=int(len(S)), n_held=int(len(held)), n_cal=int(len(cal)), gate={}, dose={}, steering={}, verdict={}, knockout={}, ablation={}, sweeps={}, compare={}, predictions={})
    if smoke:
        cnt = R.groupby(["input", "task", "arm"]).agg(n=("scene", "size"), correct=("correct", "sum"), follows_Y=("follows_Y", "sum"), follows_Z=("follows_Z", "sum")).reset_index()
        out["counts"] = cnt.to_dict(orient="records")
    for inp in inputs:
        Ri = R[R.input == inp]
        out["gate"][inp] = {
            t: dict(clean=float(pick(Ri, scenes_all, "correct", task=t, family="clean").mean()), passes=bool(pick(Ri, scenes_all, "correct", task=t, family="clean").mean() >= GATE[t]))
            for t in media.TASKS
        }
        say_ok, rule_ok = out["gate"][inp]["say"]["passes"], out["gate"][inp]["animal"]["passes"]

        def fy(task, sc, **eq):
            return pick(Ri, sc, "follows_Y", task=task, **eq)

        d_star = next((d for d in DOSES if fy("say", cal, family="steer", site="media", key="Y", set="middle", dose=d).mean() >= 0.5), None)
        out["dose"][inp] = dict(calibrated=d_star, used=d_star if d_star is not None else 8.0, note=None if d_star is not None else "no calibrated dose")
        tab = {}
        for bn in BL:
            for d in DOSES:
                e = {}
                for site, keys in (("media", STEER_KEYS), ("last", LAST_KEYS)):
                    for key in keys:
                        e[f"{site} {key} say follows Y"] = boot(fy("say", held, family="steer", site=site, key=key, set=bn, dose=d), held)
                        e[f"{site} {key} rule follows Y"] = boot(fy("animal", held, family="steer", site=site, key=key, set=bn, dose=d), held)
                e["media Z rule flips"] = boot(1 - pick(Ri, held, "correct", task="animal", family="steer", site="media", key="Z", set=bn, dose=d), held)
                e["media Z say follows Z"] = boot(pick(Ri, held, "follows_Z", task="say", family="steer", site="media", key="Z", set=bn, dose=d), held)
                tab[f"{bn} x{d:g}"] = e
        out["steering"][inp] = tab
        du = out["dose"][inp]["used"]
        V = {}
        if not say_ok:
            V = {"naming": "not perceived, untestable"}
        else:
            for bn in BL:
                y = fy("say", held, family="steer", site="media", key="Y", set=bn, dose=du)
                rnd = fy("say", held, family="steer", site="media", key="random", set=bn, dose=du)
                by, br = boot(y, held), boot(rnd, held)
                steers = bool(by[0] >= 0.5 and by[1] > br[2])
                last_y = fy("say", held, family="steer", site="last", key="Y", set=bn, dose=du).mean()
                last_wu = fy("say", held, family="steer", site="last", key="WU_Y", set=bn, dose=du).mean()
                wu = fy("say", held, family="steer", site="media", key="WU_Y", set=bn, dose=du).mean()
                e = dict(steers_naming=steers, say_follows_Y=by, random=br, J_specific=bool(wu <= 0.10), last_token_Y=float(last_y), last_token_WU=float(last_wu))
                if rule_ok:
                    dif = fy("animal", held, family="steer", site="media", key="Y", set=bn, dose=du) - fy("animal", held, family="steer", site="media", key="random", set=bn, dose=du)
                    bd = boot(dif, held)
                    zf = (1 - pick(Ri, held, "correct", task="animal", family="steer", site="media", key="Z", set=bn, dose=du)).mean()
                    e["rule_Y_minus_random"] = bd
                    e["rule"] = "seen" if bd[0] >= 0.30 and bd[1] > 0 and zf <= 0.10 else "only said" if bd[0] <= 0.10 else "undetermined"
                else:
                    e["rule"] = "rule not perceived, untestable"
                V[bn] = e
            passing = [bn for bn in BL if V[bn]["steers_naming"]]
            V["output push"] = (
                "no block steers naming"
                if not passing
                else "not told apart from an output push"
                if all(V[bn]["last_token_Y"] >= 0.5 and V[bn]["last_token_WU"] >= 0.5 for bn in passing)
                else "media-specific in " + ", ".join(bn for bn in passing if not (V[bn]["last_token_Y"] >= 0.5 and V[bn]["last_token_WU"] >= 0.5))
            )
        out["verdict"][inp] = V
        # sweeps
        sw = {}
        if say_ok:
            for kind in ("prefix", "suffix"):
                passes = []
                for L in starts:
                    y = boot(fy("say", held, family="sweep", key="Y", set=f"{kind} {L}"), held)
                    rnd = boot(fy("say", held, family="sweep", key="random", set=f"{kind} {L}"), held)
                    sw[f"{kind} {L}"] = dict(Y=y, random=rnd, steers=bool(y[0] >= 0.5 and y[1] > rnd[2]))
                    if sw[f"{kind} {L}"]["steers"]:
                        passes.append(L)
                sw[f"{kind} edge"] = (min(passes) if kind == "prefix" else max(passes)) if passes else None
        out["sweeps"][inp] = sw
        # knockout
        ko = {}
        for task in media.TASKS:
            if not out["gate"][inp][task]["passes"]:
                ko[task] = "not perceived, untestable"
                continue
            clean = pick(Ri, held, "ld", task=task, family="clean")
            e = {}
            for kind in ("from", "upto"):
                for L in KO_STARTS:
                    g = pick(Ri, held, "ld", task=task, family="knockout", ko_kind=kind, ko_L=L)
                    e[f"{kind} {L}"] = dict(cost=boot(clean - g, held), correct=boot(pick(Ri, held, "correct", task=task, family="knockout", ko_kind=kind, ko_L=L), held))
            last = max([L for L in KO_STARTS if e[f"from {L}"]["cost"][0] >= 1 and e[f"from {L}"]["cost"][1] > 0], default=None)
            e["last_reader"] = last
            e["last_reader_share"] = None if last is None else last / n_layers
            ko[task] = e
        out["knockout"][inp] = ko
        # ablation
        ab = {}
        for task in media.TASKS:
            c = pick(Ri, held, "correct", task=task, family="clean")
            ab[task] = {a: dict(cost=boot(c - pick(Ri, held, "correct", task=task, family="ablate", ablate=a), held)) for a in ABLATIONS}
            if out["gate"][inp][task]["passes"]:
                acc = {a: pick(Ri, held, "correct", task=task, family="ablate", ablate=a) for a in ("own name", "other name")}
                lp = {a: pick(Ri, held, "lp_X", task=task, family="ablate", ablate=a) for a in ("own name", "other name")}
                D, DN = boot(acc["other name"] - acc["own name"], held), boot(lp["own name"] - lp["other name"], held)
                ab[task]["D"], ab[task]["D_nats"] = D, DN
                ab[task]["R1"], ab[task]["R1_nats"] = r1(D), r1_nats(DN)
            else:
                ab[task]["R1"] = ab[task]["R1_nats"] = "not perceived, untestable"
        out["ablation"][inp] = ab
    # comparisons against the frame, at the frame's dose
    if "frame" in inputs:
        dref = out["dose"]["frame"]["used"]
        for inp in [i for i in inputs if i != "frame"]:
            cmp_ = {}
            for bn in BL:
                a = pick(R[R.input == inp], held, "follows_Y", task="say", family="steer", site="media", key="Y", set=bn, dose=dref)
                b = pick(R[R.input == "frame"], held, "follows_Y", task="say", family="steer", site="media", key="Y", set=bn, dose=dref)
                bd = boot(a - b, held)
                cmp_[bn] = dict(diff=bd, same=bool(bd[1] >= -0.15 and bd[2] <= 0.15))
            cmp_["verdict"] = "same as the picture" if all(cmp_[bn]["same"] for bn in BL) else "differs by input"
            out["compare"][f"{inp} vs frame"] = cmp_
    if "clip" in inputs and "static" in inputs:
        dref = out["dose"]["static"]["used"]
        cmp_ = {}
        for bn in BL:
            a = pick(R[R.input == "clip"], held, "follows_Y", task="say", family="steer", site="media", key="Y", set=bn, dose=dref)
            b = pick(R[R.input == "static"], held, "follows_Y", task="say", family="steer", site="media", key="Y", set=bn, dose=dref)
            bd = boot(a - b, held)
            cmp_[bn] = dict(diff=bd, same=bool(bd[1] >= -0.15 and bd[2] <= 0.15))
        cmp_["verdict"] = "nothing temporal tested" if all(cmp_[bn]["same"] for bn in BL) else "the clip differs from the static video"
        out["compare"]["clip vs static"] = cmp_
    if smoke:  # plan.md: the smoke check reports counts out of 5 and carries no verdict
        for k in ("verdict", "compare", "predictions", "knockout", "ablation", "sweeps", "steering"):
            out.pop(k, None)
        out["note"] = "smoke check: counts only, no verdict"
    else:
        out["predictions"] = predictions(out, R, inputs, BL, held, n_layers)
    return out


def r1(D):
    """name_steering's R1 on accuracies: D = other name removed minus own name removed"""
    m, lo, hi = D
    return "necessary" if m >= 0.25 and lo > 0 else "not needed" if m <= 0.10 and lo > -0.15 and hi < 0.15 else "undetermined"


def r1_nats(DN):
    """name_steering's R1_nats: log p with the own name removed minus with the other name removed"""
    m, lo, hi = DN
    return "necessary" if m <= -0.5 and hi < 0 else "not needed" if m >= -0.2 and lo > -0.3 and hi < 0.3 else "undetermined"


def suffix_edge_ok(edge, last_reader):
    """prediction 5's second half: the latest suffix start that steers naming is at or before the last naming read, within one
    step (4 layers) of it"""
    return edge is not None and last_reader - 4 <= edge <= last_reader


def reference_check(R):
    """plan.md step 5 on the middle frame: RECORD row 93 at 4 sigma, counts out of 5"""
    d = R[(R.input == "frame") & (R.task == "say") & (R.family == "steer") & (R.site == "media") & (R.dose == 4.0)]
    counts = {f"{key} {bn}": int(d[(d.key == key) & (d.set == bn)].follows_Y.sum()) for key in ("Y", "random") for bn in ("early", "middle")}
    n = int(R[(R.input == "frame") & (R.task == "say") & (R.family == "clean")].scene.nunique())
    ok = all(counts[f"Y {bn}"] >= n - 1 for bn in ("early", "middle")) and all(counts[f"random {bn}"] <= 1 for bn in ("early", "middle"))
    return dict(n=n, counts=counts, passes=bool(ok), rule="say follows Y on >= n - 1 clips from the early and the middle block at 4 sigma, random on <= 1")


def predictions(out, R, inputs, BL, held, n_layers):
    P = {}
    for inp in inputs:
        if not out["gate"][inp]["say"]["passes"]:
            P[inp] = {k: "untestable" for k in "12345678"}
            continue
        V, du, Ri = out["verdict"][inp], out["dose"][inp]["used"], R[R.input == inp]

        def m(task, **eq):
            return pick(Ri, held, "follows_Y", task=task, family="steer", site="media", dose=du, **eq).mean()

        p = {}
        p["1"] = "kept" if all(V[b]["say_follows_Y"][0] >= 0.8 for b in ("early", "middle")) else "missed"
        p["2"] = "kept" if V["late"]["say_follows_Y"][0] <= 0.1 else "missed"
        p["3"] = "kept" if max(m("say", key=k, set=b) for k in ("random", "WU_Y", "PC") for b in BL) <= 0.1 else "missed"
        if out["gate"][inp]["animal"]["passes"]:
            e, mi = m("animal", key="Y", set="early"), m("animal", key="Y", set="middle")
            p["4"] = "kept" if e > mi and e < 0.5 else "missed"
        else:
            p["4"] = "untestable"
        ko = out["knockout"][inp]["say"]
        edge = out["sweeps"][inp].get("suffix edge")
        if isinstance(ko, dict) and ko["last_reader"] is not None:
            share_ok = 0.67 <= ko["last_reader_share"] <= 0.75
            p["5"] = "kept" if share_ok and suffix_edge_ok(edge, ko["last_reader"]) else "missed"
        else:
            p["5"] = "missed" if isinstance(ko, dict) else "untestable"
        ab = out["ablation"][inp]["say"]
        p["6"] = "kept" if ab["own name"]["cost"][0] <= 0.10 and ab["lens part"]["cost"][0] <= 0.10 else "missed"
        if inp in ("clip", "static"):
            p["7"] = "kept" if out["compare"].get(f"{inp} vs frame", {}).get("verdict") == "same as the picture" else "missed"
        if inp == "sound":
            p["8"] = "kept" if out["compare"].get("sound vs frame", {}).get("verdict") == "same as the picture" else "missed"
        P[inp] = p
    return P


# ---- main


def main():
    ap = parser(__doc__, "media_steer")
    ap.add_argument("--stage", choices=["gate", "reference", "smoke", "main"], default="main")
    ap.add_argument("--n-per-class", type=int, default=8)
    ap.add_argument("--n-cal", type=int, default=16, help="calibration scenes (two per class): the dose is picked on them")
    ap.add_argument("--gate-models", default="", help="comma list of models whose gate.csv in this run must all name a clip (default: this model)")
    ap.add_argument("--inputs", default=",".join(media.INPUTS))
    ap.add_argument("--max-rows", type=int, default=512, help="largest batch, in prompts")
    ap.add_argument("--no-encoder-cache", action="store_true", help="run the vision encoder in every batch")
    args = ap.parse_args()
    if args.stage in ("smoke", "reference") and args.run == "media_steer":
        args.run = f"{args.stage}_media_steer"
    if args.stage == "reference":
        args.inputs = "frame"
    require_data(args.data_dir, "vggsound")
    man = media.manifest(args.data_dir)
    run = Run(args, f"media_steer {args.stage}")
    lm = Session(args.model)
    run.write_config(lm)
    t0 = time.time()
    mon = batched.GpuMonitor(run.path("gpu.csv"))
    inputs = [i for i in args.inputs.split(",") if i != "sound" or has_audio(lm)]
    if "sound" in args.inputs.split(",") and "sound" not in inputs:
        print(f"{lm.name} takes no audio: the sound input is left out")
    token = {inp: media_token(lm, inp) for inp in inputs}
    ID = {w: lm.single_ids([w, " " + w, w.capitalize(), " " + w.capitalize()]) for w in media.CLASSES}
    ID.update({d: lm.single_ids([d, " " + d]) for d in ("7", "3")})
    missing = [w for w, v in ID.items() if not v]
    assert not missing, f"no single-token form of {missing} in {lm.name}'s tokenizer"
    cand = {w: [c["id"] for c in man["clips"] if c["class"] == w] for w in media.CLASSES}
    cache = {}

    def clip_media(cid):
        if cid not in cache:
            cache[cid] = media.load_clip(args.data_dir, cid)
        return cache[cid]

    def gate_clips(classes):
        """the selection question (say, on the clip) for every candidate of the classes"""
        flat = [(w, c) for w in classes for c in cand[w]]
        encs = {("clip", "say", i): encode(lm, "clip", "say", [clip_media(c)]) for i, (_, c) in enumerate(flat)}
        eng = batched.Engine(lm, encs, {"clip": token["clip"]}, media.MEDIUM, batched.Tables(), lm.layers, args.max_rows)
        rows = [dict(input="clip", task="say", scene=i, site="", ablate="", ko=(), ids_x=ID[w], ids_y=ID[w]) for i, (w, _) in enumerate(flat)]
        res = eng.generate(rows, N_NEW, "gate")
        return pd.DataFrame([dict(clip=c, cls=w, generation=t, correct=media.answer_matches("say", t, w)) for (w, c), (t, _, _) in zip(flat, res)])

    if args.stage == "gate":
        G = gate_clips(media.CLASSES)
        run.save_csv(G, "gate.csv")
        print("gate (named / candidates): " + ", ".join(f"{w} {int(G[G.cls == w].correct.sum())}/{int((G.cls == w).sum())}" for w in media.CLASSES))
        run.save_json(dict(gpu=mon.stop(), seconds=time.time() - t0), "summary.json")
        return

    # ---- the clips
    smoke = args.stage in ("smoke", "reference")
    if smoke:
        G = gate_clips(media.SMOKE_CLASSES)
        ok = {w: [c for c in cand[w] if bool(G[(G.clip == c)].correct.iloc[0])] for w in media.SMOKE_CLASSES}
        empty = [w for w, v in ok.items() if not v]
        if empty:
            raise SystemExit(f"no candidate clip of {empty} is named by {lm.name}")
        S = media.assign_smoke(ok)
    else:
        models = [m for m in args.gate_models.split(",") if m] or [args.model]
        passing = None
        for m in models:
            path = args.runs_dir / args.run / m.split("/")[-1] / "gate.csv"
            if not path.exists():
                raise SystemExit(f"{path} not found: run `--stage gate --model {m} --run {args.run}` first")
            g = pd.read_csv(path)
            s = set(g[g.correct].clip)
            passing = s if passing is None else passing & s
        ok = {w: [c for c in cand[w] if c in passing] for w in media.CLASSES}
        short = {w: len(v) for w, v in ok.items() if len(v) < args.n_per_class}
        if short:
            raise SystemExit(f"fewer than {args.n_per_class} clips named by every gate model in {short}: draw more candidates (prepare_media.py --candidates)")
        S = media.assign(ok, args.n_per_class)
    run.save_json(
        dict(
            scenes=S,
            gate_models=None if smoke else models,
            manifest_revisions=man["revisions"],
            manifest_sha256=hashlib.sha256((args.data_dir / "vggsound/manifest.json").read_bytes()).hexdigest(),
            clips={c["id"]: c for c in man["clips"] if c["id"] in {r["clip"] for r in S}},
        ),
        "clips.json",
    )
    sheet = []
    for r in S[:2]:
        f, _ = clip_media(r["clip"])
        sheet += [(f"scene {r['scene']} {r['X']}: {media.FRAME_TIMES[j]:.1f} s", Image.fromarray(f[j])) for j in range(media.N_FRAMES)]
    stimuli.contact_sheet(sheet, media.N_FRAMES, run.path("stimuli.png"))

    # ---- every prompt through the processor once
    encs = {}
    for inp in inputs:
        for task in media.TASKS:
            for r in S:
                encs[(inp, task, r["scene"])] = encode(lm, inp, task, [clip_media(r["clip"])])
            lengths = {int(encs[(inp, task, r["scene"])]["input_ids"].shape[1]) for r in S}
            n_media = {int((encs[(inp, task, r["scene"])]["input_ids"] == token[inp]).sum()) for r in S}
            assert len(lengths) == 1 and len(n_media) == 1, f"{inp}/{task}: prompts differ in length {lengths} or media tokens {n_media}"
            print(f"{inp:6s} {task:6s}: {lengths.pop()} tokens, {n_media.pop()} media tokens")
    layout = {inp: dict(tokens=int(encs[(inp, "say", S[0]["scene"])]["input_ids"].shape[1]), media_tokens=int((encs[(inp, "say", S[0]["scene"])]["input_ids"] == token[inp]).sum())) for inp in inputs}
    print(f"prompts encoded  [{time.time() - t0:.0f}s]")

    # ---- the directions
    LF = lm.all_layers()
    n_layers, d = lm.final + 1, lm.d_model
    nearest = lambda L: L if (0 <= L < lm.final and L in lm.jac) else (0 if L < 0 else lm.final - 1)  # noqa: E731
    sc_n = max(r["scene"] for r in S) + 1
    dev = lm.device

    def per_scene(word_col, kind):
        out = {}
        for L in LF:
            V = np.zeros((sc_n, d), np.float32)
            V[[r["scene"] for r in S]] = lens.word_directions(lm, nearest(L), [ID[r[word_col]][0] for r in S], kind)[:, :, 0]
            out[L] = torch.as_tensor(V, device=dev)
        return out

    T = batched.Tables()
    T.dirs["X"] = per_scene("X", "J")
    T.dirs["Y"] = per_scene("Y", "J")
    T.dirs["Z"] = per_scene("Z", "J")
    T.dirs["WU_Y"] = per_scene("Y", "W_U")
    T.dirs["random"] = {L: torch.as_tensor(np.tile(lens.unit_rows(1, d, 700 + L)[:, :, 0], (sc_n, 1)), device=dev) for L in LF}  # one random direction per layer, seed 700 + L
    r_sub = int(round(0.1 * d))
    T.abl["own name"] = ("scene", {L: T.dirs["X"][L] for L in lm.layers})
    T.abl["other name"] = ("scene", {L: T.dirs["Y"][L] for L in lm.layers})
    T.abl["random direction"] = ("scene", {L: torch.as_tensor(np.tile(lens.unit_rows(1, d, 200 + L)[:, :, 0], (sc_n, 1)), device=dev) for L in lm.layers})
    T.abl["lens part"] = ("shared", {L: torch.as_tensor(lens.lens_subspace(lm, L, r_sub), device=dev) for L in lm.layers})
    T.abl["random part"] = ("shared", {L: torch.as_tensor(lens.random_subspace(d, r_sub, 100 + L), device=dev) for L in lm.layers})
    for L in [L for L in list(lm.jac) if L not in lm.layers]:  # the other maps are no longer needed: their memory goes to the batches
        del lm.jac[L]
    torch.cuda.empty_cache()
    print(f"directions for {len(LF)} layers, lens part of rank {r_sub}  [{time.time() - t0:.0f}s]")

    eng = batched.Engine(lm, encs, token, media.MEDIUM, T, lm.layers, args.max_rows)
    for inp in inputs:  # the batch of the processor's own output against the concatenation of single prompts
        eng.check_collate(inp, "say", [r["scene"] for r in S[:3]], lambda scs, inp=inp: encode(lm, inp, "say", [clip_media(next(x["clip"] for x in S if x["scene"] == s)) for s in scs]))
    cache_report = {}
    for inp in inputs:
        if not args.no_encoder_cache and eng.build_encoder_cache(inp, [r["scene"] for r in S]):
            cache_report[inp] = eng.check_encoder_cache(inp, [make_row(inp, "say", r, ID) for r in S[:4]], N_NEW)
        else:
            cache_report[inp] = dict(on=False)

    # ---- the clean states: dose unit, mean, principal direction, norms, and the lens readout at the read layers
    stats = {}
    for inp in inputs:
        st = dict(
            abs_proj={L: 0.0 for L in LF},
            n_tok={L: 0 for L in LF},
            sum={L: torch.zeros(d, device=dev) for L in LF},
            norm={L: 0.0 for L in LF},
            pc={L: torch.zeros(sc_n, d, device=dev) for L in LF},
            keep={L: {} for L in lm.layers},
        )
        stats[inp] = st

        def on_layer(L, rows, v, st=st):
            sc = torch.as_tensor([r["scene"] for r in rows], device=dev)
            st["abs_proj"][L] += float((v * T.dirs["X"][L][sc][:, None, :]).sum(-1).abs().sum())
            st["n_tok"][L] += v.shape[0] * v.shape[1]
            st["sum"][L] += v.sum((0, 1))
            st["norm"][L] += float(v.norm(dim=-1).sum())
            c = v - v.mean(1, keepdim=True)
            vh = torch.linalg.svd(c, full_matrices=False).Vh[:, 0]  # [b, d]
            sign = torch.sign((v.mean(1) * vh).sum(-1))
            st["pc"][L][sc] = vh * torch.where(sign == 0, torch.ones_like(sign), sign)[:, None]
            if L in st["keep"]:
                for j, r in enumerate(rows):
                    st["keep"][L][r["scene"]] = v[j].half().cpu()

        eng.capture([make_row(inp, "say", r, ID) for r in S], LF, on_layer)
        T.sigma[inp] = {L: st["abs_proj"][L] / st["n_tok"][L] for L in LF}
        T.mu[inp] = {L: st["sum"][L] / st["n_tok"][L] for L in lm.layers}
        T.dirs[(inp, "PC")] = st["pc"]
        print(f"{inp}: clean states captured; sigma " + ", ".join(f"L{L} {T.sigma[inp][L]:.2f}" for L in LF[:: max(1, len(LF) // 5)]) + f"  [{time.time() - t0:.0f}s]")
    readout = lens_readout(lm, stats, S, ID, inputs)
    norms = {inp: {L: dict(mean_norm=stats[inp]["norm"][L] / stats[inp]["n_tok"][L], sigma=T.sigma[inp][L]) for L in LF} for inp in inputs}
    for inp in inputs:
        stats[inp]["keep"] = None

    # ---- phase A: clean, steering over the blocks at four doses, knockout, ablation
    BL = media.blocks(n_layers)
    AL = sorted(eng.attn)
    KO_STARTS = [L for L in AL if (L - AL[0]) % 4 == 0] if len(AL) == n_layers else AL
    print(f"blocks {', '.join(f'{k} {v[0]}..{v[-1]}' for k, v in BL.items())}; attention layers {AL[:4]}...{AL[-1]} ({len(AL)}); knockout starts {KO_STARTS}")
    rows_a = phase_a(inputs, S, ID, BL, KO_STARTS, AL)
    print(f"phase A: {len(rows_a)} rows")
    res_a = eng.generate(rows_a, N_NEW, "phase A")
    R = to_frame(rows_a, res_a, "A")

    # ---- phase B: the sweeps at each input's calibrated dose
    scenes_all = np.array([r["scene"] for r in S])
    cal = scenes_all if smoke else scenes_all[scenes_all < args.n_cal]
    dose_of = {}
    for inp in inputs:
        ri = R[(R.input == inp) & (R.task == "say") & (R.family == "steer") & (R.site == "media") & (R.key == "Y") & (R.set == "middle")]
        dose_of[inp] = next((dd for dd in DOSES if ri[ri.dose == dd].set_index("scene").follows_Y.reindex(cal).astype(float).mean() >= 0.5), 8.0)
    starts = media.sweep_starts(n_layers)
    rows_b = phase_b(inputs, S, ID, starts, n_layers, dose_of)
    print(f"phase B: {len(rows_b)} rows at doses {dose_of}")
    res_b = eng.generate(rows_b, N_NEW, "phase B")
    R = pd.concat([R, to_frame(rows_b, res_b, "B")], ignore_index=True)
    run.save_csv(R.drop(columns=["ids_x", "ids_y", "ko"]).assign(layers=R.layers.map(lambda t: f"{t[0]}..{t[-1]}" if t else "")), "media_steer.csv")

    gpu = mon.stop()
    out = summarize(R, S, inputs, BL, starts, AL, KO_STARTS, n_layers, args.n_cal, smoke)
    out["setup"] = dict(
        inputs=inputs,
        layout=layout,
        blocks={k: [v[0], v[-1]] for k, v in BL.items()},
        attention_layers=AL,
        knockout_starts=KO_STARTS,
        sweep_starts=starts,
        lens_part_rank=r_sub,
        encoder_cache=cache_report,
        batch_rows={f"{inp}/{task}": b for (inp, task), b in eng.cap.items()},
        batches=eng.batches,
        rows=eng.rows_done,
        gpu=gpu,
        gpu_busy_share=eng.t_gpu / max(time.time() - t0, 1e-9),
        seconds=time.time() - t0,
    )
    out["readout"], out["norms"] = readout, norms
    if args.stage == "reference":
        out["reference"] = reference_check(R)
    run.save_json(out, "summary.json")
    for inp in inputs:
        print(
            f"{inp}: gate {json.dumps({t: round(v['clean'], 2) for t, v in out['gate'][inp].items()})}, dose {out['dose'][inp]}, "
            + ("; ".join(f"{b}: {v['steers_naming']} ({v['say_follows_Y'][0]:.2f})" for b, v in out.get("verdict", {}).get(inp, {}).items() if isinstance(v, dict)))
        )
    if out.get("predictions"):
        print("predictions: " + json.dumps(out["predictions"]))
    if gpu:
        print(f"GPU utilization: mean {gpu['mean_utilization']}%, median {gpu['median_utilization']}% over {gpu['samples']} samples")
    if args.stage == "reference":
        ref = out["reference"]
        print(f"reference (plan.md step 5, RECORD row 93): {'reproduced' if ref['passes'] else 'NOT reproduced'}: {ref['counts']} of {ref['n']}")
        if not ref["passes"]:
            raise SystemExit("the middle frame does not reproduce row 93: stop and find the error before any other run (plan.md step 5)")


def to_frame(rows, res, phase):
    recs = []
    for r, (text, lx, ly) in zip(rows, res):
        recs.append(
            dict(
                {k: v for k, v in r.items()},
                phase=phase,
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


@torch.no_grad()
def lens_readout(lm, stats, S, ID, inputs, chunk=64):
    """per input and read layer: the share of clips whose own name (and, as the null, Y's name) is in the J-lens and logit-lens
    top 10, read off the mean media state (pooled) and off the best single media token"""
    out = {}
    for inp in inputs:
        e = {}
        for L in lm.layers:
            hits = {k: [] for k in ("J pooled", "J token", "logit pooled", "logit token")}
            null = {k: [] for k in hits}
            for r in S:
                H = stats[inp]["keep"][L][r["scene"]].to(lm.device).float()
                for lens_kind, M in (("J", H @ lm.jac[L].float().T), ("logit", H)):
                    for word, store in ((r["X"], hits), (r["Y"], null)):
                        ids = torch.as_tensor(ID[word], device=lm.device)
                        x = lm.model.unembed(M.mean(0, keepdim=True)).float()
                        rk_pool = int(((x[:, None, :] > x[:, ids][:, :, None]).sum(-1) + 1).min())
                        rk_tok = 10**9
                        for s in range(0, M.shape[0], chunk):
                            x = lm.model.unembed(M[s : s + chunk]).float()
                            rk_tok = min(rk_tok, int(((x[:, None, :] > x[:, ids][:, :, None]).sum(-1) + 1).min()))
                        store[f"{lens_kind} pooled"].append(rk_pool <= lens.TOP)
                        store[f"{lens_kind} token"].append(rk_tok <= lens.TOP)
            e[L] = {k: dict(own=float(np.mean(hits[k])), null=float(np.mean(null[k]))) for k in hits}
        out[inp] = e
        print(f"{inp}: lens readout (own name in the top 10, J pooled / J token): " + ", ".join(f"L{L} {v['J pooled']['own']:.2f}/{v['J token']['own']:.2f}" for L, v in e.items()))
    return out


if __name__ == "__main__":
    main()
