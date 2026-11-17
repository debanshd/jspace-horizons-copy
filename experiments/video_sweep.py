"""Video sweep: depth_sweep's patching and media_steer's interventions on the matched video set (plan.md part 3).

Data. saxenan3/jspace-matched-video (src/jspace/matched.py): 64 scenes on the image experiment's schedule, each a donor video
(object X) and a receiver video (object Z of the other kind), the numbered slot q and the three names of the set question.
Every frame is drawn into slot q of the six-slot canvas, as depth_sweep draws a crop. Three inputs per clip:
  moving    the video, at the frames the model's own sampler picks (Qwen3.5: 2 per second; Gemma-4: 32 spread over the clip),
            the same count for donor and receiver (the receiver's native count, capped at both clips' lengths)
  frozen    the middle sampled frame repeated as often (the same tokens, no motion)
  still     that frame as a picture (the image experiment on a frame of the video)
Prompts are depth_sweep's: INTRO, the media, then one question about image slot q (say, animal, letter, set, ride).

Arms (all of them share batches, jspace.batched):
  patching, every question: clean; the receiver's own neutral-prompt states written back at every layer (the machinery check);
            the donor's at every layer (the positive control); and per window of 4 layers from layer -1 to the last, as in
            depth_sweep: the donor's states (transplant), the receiver's own name's J-lens direction at 4 sigma (inject), a
            random direction at 4 sigma (random), sigma being depth_sweep's: the mean size of the donor name's component on
            the receiver's media tokens. The donor's states come from its neutral prompt (INTRO and the media, no question).
  the other interventions, say and animal: media_steer's phase A and B with X the receiver's object, Y the donor's and Z a
            same-kind word (matched.same_kind_word): steering over the early, middle and late blocks at doses 1, 2, 4, 8 on the
            media tokens (Y, Z, random, Y's unembedding row, the clip's principal direction) and at the last token (Y, Y's
            unembedding row), the attention knockouts, the five ablations, and the sweeps at the calibrated dose.

The rule (written before the run):
- Patching follows depth_sweep and media_patch: gate on clean receiver naming (>= 0.90) and the rule (>= 0.85); machinery: the
  self-patch gives the clean answer on >= 0.95 of rows; positive control: the every-layer donor patch makes say follow the
  donor on >= 0.9; a question's horizon is the last window start at which it follows the donor on >= 0.5 of scenes; L_say is
  say's, L_use the last window start at which the animal rule follows the donor at least 0.3 more than under the random
  direction (depth_sweep's cut); verdict "too late" when L_use <= L_say - 8, "together", "never" or "nothing moves".
- The other interventions follow media_steer's frozen rule unchanged (calibration on scenes 0 to 15, verdicts on 16 to 63).
- Video against its controls: moving and frozen agree when their horizons are within one window (patching) and their
  say-follows-Y rates have a paired difference inside +-0.15 in every block (steering); the same for moving against still.

Predictions (written before the run; image horizons from docs/REPORT.md): media_patch's P1 to P3 for the moving video (two
horizons in that order; the rule's share in the family's image range, naming's in 0.67 to 0.75 or below the key-value sharing
limit, each widened by one window); P4 moving and frozen agree; P5 moving and still agree; and media_steer's predictions 1 to
6 for the steering arms.

Outputs in runs/<run>/<model>/: video_sweep.csv (every row), summary.json (patch: gates, controls, curves, horizons,
verdicts, predictions; interventions: media_steer's summary; compare; setup), clips.json (per scene the frame count, the
sampled indices of both clips and their sha256), stimuli.png, gpu.csv, config.json, run.log.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from depth_sweep import INTRO, SPECTRUM, TASKS  # noqa: E402
from media_patch import FAMILY_RULE_RANGE, NAME_RANGE, WINDOW, kv_limit, same_horizons, verdict, windows, within  # noqa: E402
from media_steer import DOSES, N_NEW, phase_a, phase_b  # noqa: E402
from media_steer import summarize as steer_summary  # noqa: E402

from jspace import batched, lens, matched, stimuli  # noqa: E402
from jspace import media as media_rules  # noqa: E402
from jspace.analysis import boot_scene  # noqa: E402
from jspace.config import require_data  # noqa: E402
from jspace.model import Session  # noqa: E402
from jspace.runs import Run, parser  # noqa: E402

QUESTIONS = {**TASKS, **SPECTRUM}  # say, animal, letter, set, ride, worded as in depth_sweep
STEER_TASKS = ["say", "animal"]
GATE = {"say": 0.90, "animal": 0.85}
N_CAL = 16


# ---- prompts and encodings


def prompt(lm, inp, task, r):
    part = {"video": {"type": "video"}, "picture": {"type": "image"}}[matched.MEDIUM[inp.split("@")[0]]]
    content = [{"type": "text", "text": INTRO}, part]
    if task != "neutral":
        content.append({"type": "text", "text": QUESTIONS[task].format(q=r["slot"], a=r["set"][0], b=r["set"][1], c=r["set"][2])})
    return lm.chat_messages([{"role": "user", "content": content}])


def encode(lm, inp, text, media_obj, meta):
    from transformers.video_utils import VideoMetadata

    if matched.MEDIUM[inp.split("@")[0]] == "video":
        kw = dict(videos=[media_obj], video_metadata=[VideoMetadata(**meta)], do_sample_frames=False)
        if type(lm.processor).__name__.startswith("Qwen"):
            kw["cap_pixels_per_frame"] = False  # transformers 5.19's behaviour, stated so its warning stays off
    else:
        kw = dict(images=[[media_obj]] if type(lm.processor).__name__.startswith("Gemma4") else [media_obj])
    enc = lm.processor(text=[text], return_tensors="pt", **kw)
    out = {}
    for k, v in enc.items():  # the pixels in the model's dtype: half the memory of float32, the dtype the forward casts to anyway
        out[k] = v.to(lm.dtype) if torch.is_tensor(v) and k in batched.FLOAT_KEYS else v
    return out


def row(inp, task, r, ID, **kw):
    """one prompt with its intervention; X is the receiver's object, Y the donor's, Z the same-kind word"""
    tx, ty = matched.truth(task, r["X"], r["set"]), matched.truth(task, r["Y"], r["set"])
    out = dict(
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
        donor=None,
    )
    out.update(kw)
    out["ids_x"], out["ids_y"] = ID[tx], ID[ty]
    return out


def patch_rows(inp, S, ID, n_layers):
    rows = []
    every = tuple(range(-1, n_layers))
    for task in QUESTIONS:
        for r in S:
            rows.append(row(inp, task, r, ID, family="pclean", arm="clean"))
            rows.append(row(inp, task, r, ID, family="self", arm="own states, every layer", site="patch", layers=every, donor=("self", r["scene"])))
            rows.append(row(inp, task, r, ID, family="full", arm="donor states, every layer", site="patch", layers=every, donor=("donor", r["scene"])))
            for w in windows(n_layers):
                rows.append(row(inp, task, r, ID, family="transplant", arm=f"transplant, layers {w[0]} to {w[-1]}", site="patch", layers=tuple(w), set=str(w[0]), donor=("donor", r["scene"])))
                rows.append(row(inp, task, r, ID, family="inject", arm=f"inject own name, layers {w[0]} to {w[-1]}", site="media", key="dsw_own", layers=tuple(w), set=str(w[0]), dose=4.0))
                rows.append(row(inp, task, r, ID, family="random", arm=f"inject random, layers {w[0]} to {w[-1]}", site="media", key="dsw_random", layers=tuple(w), set=str(w[0]), dose=4.0))
    return rows


def to_frame(rows, res):
    recs = []
    for r, (text, lx, ly) in zip(rows, res):
        rec = {k: v for k, v in r.items() if k not in ("ids_x", "ids_y", "ko", "layers", "donor")}
        mem = SETS[r["scene"]]
        rec.update(
            layers=f"{r['layers'][0]}..{r['layers'][-1]}" if r["layers"] else "",
            donor="" if r["donor"] is None else f"{r['donor'][0]}",
            generation=text,
            correct=matched.matches(r["task"], text, matched.truth(r["task"], r["X"], mem)),
            follows_Y=matched.matches(r["task"], text, matched.truth(r["task"], r["Y"], mem)),
            follows_Z=matched.matches(r["task"], text, matched.truth(r["task"], r["Z"], mem)),
            lp_X=lx,
            lp_Y=ly,
            ld=lx - ly,
        )
        recs.append(rec)
    return pd.DataFrame(recs)


SETS = {}  # scene -> the set question's three names (filled in main; to_frame reads it)


# ---- the patching summary


def curve(R, scenes, family, task, starts, col="follows_Y"):
    d = R[(R["family"] == family) & (R["task"] == task)]
    out = []
    for s in starts:
        v = d[d["set"] == str(s)].set_index("scene")[col].astype(float).reindex(scenes)
        assert not v.isna().any(), (family, task, s)
        out.append(boot_scene(v.to_numpy(), scenes))
    return out


def patch_summary(R, inputs, n_layers, family, kv):
    scenes = np.sort(R["scene"].unique())
    starts = [w[0] for w in windows(n_layers)]
    out = dict(windows=starts, gate={}, controls={}, curves={}, horizon={}, cut={}, verdict={}, share={}, predictions={})
    for inp in inputs:
        Ri = R[R["input"] == inp]

        def per_scene(fam, task, col):
            v = Ri[(Ri["family"] == fam) & (Ri["task"] == task)].set_index("scene")[col].reindex(scenes)
            assert not v.isna().any(), (inp, fam, task, col)
            return v.to_numpy()

        out["gate"][inp] = {t: dict(clean=float(per_scene("pclean", t, "correct").astype(float).mean())) for t in QUESTIONS}
        for t in QUESTIONS:
            out["gate"][inp][t]["passes"] = bool(out["gate"][inp][t]["clean"] >= GATE.get(t, 0.85))
        same = float(np.mean([(per_scene("self", t, "generation") == per_scene("pclean", t, "generation")).mean() for t in QUESTIONS]))
        pos = boot_scene(per_scene("full", "say", "follows_Y").astype(float), scenes)
        out["controls"][inp] = dict(self_patch_same=same, machinery_ok=bool(same >= 0.95), every_layer_say_follows_donor=pos, positive_ok=bool(pos[0] >= 0.9))
        cur = {f"{fam} {t}": curve(Ri, scenes, fam, t, starts) for fam in ("transplant", "inject", "random") for t in QUESTIONS}
        out["curves"][inp] = cur
        out["horizon"][inp] = {t: max([s for s, v in zip(starts, cur[f"transplant {t}"]) if v[0] >= 0.5], default=None) for t in QUESTIONS}
        ok = out["gate"][inp]["say"]["passes"] and out["controls"][inp]["machinery_ok"] and out["controls"][inp]["positive_ok"]
        if not ok:
            why = "not perceived, untestable" if not out["gate"][inp]["say"]["passes"] else "patch machinery failed" if not out["controls"][inp]["machinery_ok"] else "positive control failed"
            out["cut"][inp] = out["verdict"][inp] = why
            continue
        say_ok = [s for s, v in zip(starts, cur["transplant say"]) if v[0] >= 0.5]
        use_ok = [s for s, v, rv in zip(starts, cur["transplant animal"], cur["random animal"]) if v[0] >= rv[0] + 0.3] if out["gate"][inp]["animal"]["passes"] else []
        Ls, Lu = (max(say_ok) if say_ok else None), (max(use_ok) if use_ok else None)
        out["cut"][inp] = dict(L_say=Ls, L_use=Lu, say_windows=say_ok, use_windows=use_ok, rule_read=bool(out["gate"][inp]["animal"]["passes"]))
        out["verdict"][inp] = verdict(Ls, Lu)
        out["share"][inp] = dict(say=None if Ls is None else (Ls + WINDOW) / n_layers, use=None if Lu is None else (Lu + WINDOW) / n_layers)
    w = WINDOW / n_layers
    for inp in inputs:
        h = out["cut"][inp]
        if not isinstance(h, dict):
            out["predictions"][inp] = {k: "untestable" for k in ("P1", "P2", "P3", "P4", "P5")}
            continue
        sh, p = out["share"][inp], {}
        p["P1"] = ("kept" if out["verdict"][inp] == "too late" else "missed") if h["rule_read"] else "untestable"
        p["P2"] = ("kept" if within(sh["use"], *FAMILY_RULE_RANGE[family], w) else "missed") if h["rule_read"] else "untestable"
        name_ok = (sh["say"] is not None and sh["say"] <= kv + w) if kv else within(sh["say"], *NAME_RANGE, w)
        p["P3"] = "kept" if name_ok else "missed"
        if inp == "moving":
            for key, other in (("P4", "frozen"), ("P5", "still")):
                p[key] = ("kept" if same_horizons(h, out["cut"].get(other)) else "missed") if isinstance(out["cut"].get(other), dict) else "untestable"
        out["predictions"][inp] = p
    return out


def steer_compare(R, inputs, BL, dose_of):
    """moving against frozen and against still: paired say-follows-Y difference per block at the moving video's dose"""
    out = {}
    held = np.sort(R[R["scene"] >= N_CAL]["scene"].unique())
    for other in ("frozen", "still"):
        if "moving" not in inputs or other not in inputs:
            continue
        e = {}
        for bn in BL:
            v = {}
            for inp in ("moving", other):
                d = R[(R["input"] == inp) & (R["task"] == "say") & (R["family"] == "steer") & (R["site"] == "media") & (R["key"] == "Y") & (R["set"] == bn) & (R["dose"] == dose_of["moving"])]
                v[inp] = d.set_index("scene")["follows_Y"].astype(float).reindex(held).to_numpy()
            bd = boot_scene(v["moving"] - v[other], held)
            e[bn] = dict(diff=bd, same=bool(bd[1] >= -0.15 and bd[2] <= 0.15))
        e["verdict"] = "same" if all(e[bn]["same"] for bn in BL) else "differs"
        out[f"moving vs {other}"] = e
    return out


# ---- main


def main():
    ap = parser(__doc__, "video_sweep")
    ap.add_argument("--n", type=int, default=64, help="scenes (5 for a smoke check: counts only)")
    ap.add_argument("--inputs", default=",".join(matched.INPUTS))
    ap.add_argument("--parts", default="patch,steer", help="patch, steer, or both")
    ap.add_argument("--donor-gb", type=float, default=16.0, help="host memory for donor states at a time: scenes run in chunks that fit")
    ap.add_argument("--max-rows", type=int, default=512)
    ap.add_argument("--no-encoder-cache", action="store_true")
    args = ap.parse_args()
    require_data(args.data_dir, "matched")
    run = Run(args, "video_sweep")
    lm = Session(args.model)
    run.write_config(lm)
    t0 = time.time()
    mon = batched.GpuMonitor(run.path("gpu.csv"))
    parts, inputs = set(args.parts.split(",")), args.inputs.split(",")
    meta = matched.scenes(args.data_dir)[: args.n]
    root = args.data_dir / matched.DIR / "test"
    vp = lm.processor.video_processor

    # ---- per scene: the frame count and both clips' sampled frames, drawn into the slot
    S, clips, media = [], {}, {}
    for m in meta:
        info = {role: matched.video_info(root / m[f"{role}_file_name"]) for role in ("donor", "receiver")}
        fps = info["receiver"][1]
        n = matched.scene_frame_count(vp, info["receiver"][0], info["donor"][0], fps)
        idx = {role: matched.native_indices(vp, info[role][0], info[role][1], n) for role in ("donor", "receiver")}
        frames = {role: matched.canvas_frames(matched.decode(root / m[f"{role}_file_name"], idx[role]), m["slot"]) for role in ("donor", "receiver")}
        sc = m["scene_id"]
        r = dict(
            scene=sc,
            X=m["receiver_label"],
            Y=m["donor_label"],
            Z=matched.same_kind_word(m["receiver_label"], m["donor_label"], sc),
            clip=m["receiver_clip_id"],
            slot=m["slot"],
            set=list(m["membership"]),
        )
        S.append(r)
        SETS[sc] = r["set"]
        for role in ("donor", "receiver"):
            md = dict(total_num_frames=info[role][0], fps=info[role][1], frames_indices=idx[role], duration=info[role][0] / info[role][1], width=frames[role].shape[2], height=frames[role].shape[1])
            media[(role, sc)] = (frames[role], md)
        clips[sc] = dict(
            frames=n, donor=dict(clip=m["donor_clip_id"], indices=idx["donor"], total=info["donor"][0]), receiver=dict(clip=m["receiver_clip_id"], indices=idx["receiver"], total=info["receiver"][0])
        )
    run.save_json(dict(scenes=S, clips=clips, sampler=type(vp).__name__), "clips.json")
    stimuli.contact_sheet(
        [(f"scene {S[0]['scene']} {role} frame {j}", Image.fromarray(media[(role, S[0]["scene"])][0][j])) for role in ("receiver", "donor") for j in range(min(4, clips[S[0]["scene"]]["frames"]))],
        4,
        run.path("stimuli.png"),
    )
    print(f"{len(S)} scenes; frames per scene {sorted({c['frames'] for c in clips.values()})}  [{time.time() - t0:.0f}s]")

    ID = {w: lm.single_ids([w, " " + w, w.capitalize(), " " + w.capitalize()]) for w in matched.VOCAB}
    ID.update({c: lm.single_ids([c, " " + c]) for c in [*"7 3".split(), *sorted({w[0].upper() for w in matched.VOCAB})]})
    missing = [k for k, v in ID.items() if not v]
    assert not missing, f"no single-token form of {missing} in {lm.name}'s tokenizer"

    def enc_for(inp, task, r, role="receiver"):
        frames, md = media[(role, r["scene"])]
        return encode(lm, inp, prompt(lm, inp, task, r), matched.input_of(frames, inp.split("@")[0]), md)

    n_layers, d = lm.final + 1, lm.d_model
    LF = lm.all_layers()
    nearest = lambda L: L if (0 <= L < lm.final and L in lm.jac) else (0 if L < 0 else lm.final - 1)  # noqa: E731
    sc_n = max(r["scene"] for r in S) + 1
    dev = lm.device

    def per_scene_dirs(word_col, kind):
        out = {}
        for L in LF:
            V = np.zeros((sc_n, d), np.float32)
            V[[r["scene"] for r in S]] = lens.word_directions(lm, nearest(L), [ID[r[word_col]][0] for r in S], kind)[:, :, 0]
            out[L] = torch.as_tensor(V, device=dev)
        return out

    T = batched.Tables()
    for key, col, kind in (("X", "X", "J"), ("Y", "Y", "J"), ("Z", "Z", "J"), ("WU_Y", "Y", "W_U")):
        T.dirs[key] = per_scene_dirs(col, kind)
    T.dirs["random"] = {L: torch.as_tensor(np.tile(lens.unit_rows(1, d, 700 + L)[:, :, 0], (sc_n, 1)), device=dev) for L in LF}
    rnd_dsw = {L: torch.as_tensor(np.tile(lens.unit_rows(1, d, 900 + L)[:, :, 0], (sc_n, 1)), device=dev) for L in LF}  # depth_sweep's seeds
    r_sub = int(round(0.1 * d))
    T.abl["own name"] = ("scene", {L: T.dirs["X"][L] for L in lm.layers})
    T.abl["other name"] = ("scene", {L: T.dirs["Y"][L] for L in lm.layers})
    T.abl["random direction"] = ("scene", {L: torch.as_tensor(np.tile(lens.unit_rows(1, d, 200 + L)[:, :, 0], (sc_n, 1)), device=dev) for L in lm.layers})
    T.abl["lens part"] = ("shared", {L: torch.as_tensor(lens.lens_subspace(lm, L, r_sub), device=dev) for L in lm.layers})
    T.abl["random part"] = ("shared", {L: torch.as_tensor(lens.random_subspace(d, r_sub, 100 + L), device=dev) for L in lm.layers})
    for L in [L for L in list(lm.jac) if L not in lm.layers]:
        del lm.jac[L]
    torch.cuda.empty_cache()
    token = {}
    for inp in inputs:
        tid = lm.processor.video_token_id if matched.MEDIUM[inp] == "video" else getattr(lm.processor, "image_token_id", lm.image_token_id)
        for suffix in ("", "@donor", "@self"):
            token[inp + suffix] = int(tid)
    medium = {k: matched.MEDIUM[k.split("@")[0]] for k in token}

    def engine(encs):
        return batched.Engine(lm, encs, token, medium, T, lm.layers, args.max_rows)

    def chunks(gb_of):
        """scenes in order, cut where their donor states (both clips, every layer) would pass --donor-gb"""
        out, cur, used = [], [], 0.0
        for r in S:
            g = gb_of(r)
            if cur and used + g > args.donor_gb:
                out.append(cur)
                cur, used = [], 0.0
            cur.append(r)
            used += g
        return out + ([cur] if cur else [])

    R_parts, setup = [], dict(layout={}, cache={}, chunks={})
    BL = None
    for inp in inputs:
        # pass 1: the clean receiver states give the dose units, the means, the principal directions (nothing is kept)
        st = dict(ms={L: 0.0 for L in LF}, dsw={L: 0.0 for L in LF}, n={L: 0 for L in LF}, sum={L: torch.zeros(d, device=dev) for L in LF}, pc={L: torch.zeros(sc_n, d, device=dev) for L in LF})

        def stats(L, rows, v, st=st):
            sc = torch.as_tensor([r["scene"] for r in rows], device=dev)
            st["ms"][L] += float((v * T.dirs["X"][L][sc][:, None, :]).sum(-1).abs().sum())
            st["dsw"][L] += float((v * T.dirs["Y"][L][sc][:, None, :]).sum(-1).abs().sum())
            st["n"][L] += v.shape[0] * v.shape[1]
            st["sum"][L] += v.sum((0, 1))
            vh = torch.linalg.svd(v - v.mean(1, keepdim=True), full_matrices=False).Vh[:, 0]
            sign = torch.sign((v.mean(1) * vh).sum(-1))
            st["pc"][L][sc] = vh * torch.where(sign == 0, torch.ones_like(sign), sign)[:, None]

        for part in [S[i : i + 8] for i in range(0, len(S), 8)]:
            encs = {(inp, "say", r["scene"]): enc_for(inp, "say", r) for r in part}
            engine(encs).capture([row(inp, "say", r, ID) for r in part], LF, stats)
        T.sigma[inp] = {L: st["ms"][L] / st["n"][L] for L in LF}
        T.mu[inp] = {L: st["sum"][L] / st["n"][L] for L in lm.layers}
        T.dirs[(inp, "PC")] = st["pc"]
        ratio = {L: (st["dsw"][L] / st["n"][L]) / max(T.sigma[inp][L], 1e-12) for L in LF}
        # depth_sweep's dose unit differs from media_steer's; the inject directions carry the ratio so dose 4 is 4 of depth_sweep's sigma
        T.dirs[(inp, "dsw_own")] = {L: T.dirs["X"][L] * ratio[L] for L in LF}
        T.dirs[(inp, "dsw_random")] = {L: rnd_dsw[L] * ratio[L] for L in LF}
        print(f"{inp}: dose units ready  [{time.time() - t0:.0f}s]")

        # pass 2: per chunk of scenes, the donors' and receivers' neutral states, then every patching row and media_steer's phase A
        r0 = S[0]
        tok0 = int((enc_for(inp, "neutral", r0, "donor")["input_ids"] == token[inp]).sum())
        per_frame = tok0 / clips[r0["scene"]]["frames"] if matched.MEDIUM[inp] == "video" else None

        def gb_of(r):  # media tokens grow with the frame count for a video and are fixed for a picture
            toks = per_frame * clips[r["scene"]]["frames"] if per_frame else tok0
            return 2 * toks * d * len(LF) * 2 / 2**30

        groups = chunks(gb_of)
        setup["chunks"][inp] = [len(g) for g in groups]
        cache_decided = None
        rows_a_all = []
        for g in groups:
            encs = {}
            for r in g:
                for task in QUESTIONS:
                    encs[(inp, task, r["scene"])] = enc_for(inp, task, r)
                encs[(inp + "@donor", "neutral", r["scene"])] = enc_for(inp + "@donor", "neutral", r, "donor")
                encs[(inp + "@self", "neutral", r["scene"])] = enc_for(inp + "@self", "neutral", r, "receiver")
                setup["layout"].setdefault(inp, {})[r["scene"]] = int((encs[(inp, "say", r["scene"])]["input_ids"] == token[inp]).sum())
            eng = engine(encs)
            if not args.no_encoder_cache and eng.build_encoder_cache(inp, [r["scene"] for r in g]):
                if cache_decided is None:
                    setup["cache"][inp] = eng.check_encoder_cache(inp, [row(inp, "say", r, ID) for r in g[:4]], N_NEW)
                    cache_decided = setup["cache"][inp]["on"]
                eng.cache_on[inp] = cache_decided
            T.donor[inp] = {L: {} for L in LF}

            def keep(L, rows, v, inp=inp):
                for j, rr in enumerate(rows):
                    T.donor[inp][L][(rr["input"].split("@")[1], rr["scene"])] = v[j].half().cpu()

            eng.capture([row(inp + "@donor", "neutral", r, ID) for r in g] + [row(inp + "@self", "neutral", r, ID) for r in g], LF, keep)
            rows = []
            if "patch" in parts:
                rows += patch_rows(inp, g, ID, n_layers)
            if "steer" in parts:
                AL = sorted(eng.attn)
                KO = [L for L in AL if (L - AL[0]) % 4 == 0] if len(AL) == n_layers else AL
                BL = media_rules.blocks(n_layers)
                ra = [dict(x, donor=None) for x in phase_a([inp], g, ID, BL, KO, AL)]
                for x in ra:  # media_steer's rows, with the matched set's truth (the 16 classes)
                    x["ids_x"], x["ids_y"] = ID[matched.truth(x["task"], x["X"], [])], ID[matched.truth(x["task"], x["Y"], [])]
                rows += ra
                rows_a_all += ra
            res = eng.generate(rows, N_NEW, f"{inp}: scenes {g[0]['scene']}..{g[-1]['scene']}")
            R_parts.append(to_frame(rows, res))
            del T.donor[inp]
            torch.cuda.empty_cache()
        print(f"{inp}: patching and phase A done  [{time.time() - t0:.0f}s]")

    R = pd.concat(R_parts, ignore_index=True)
    out = dict(n=len(S), smoke=len(S) < 64)
    family, kv = ("gemma" if "gemma" in lm.name.lower() else "qwen"), kv_limit(lm)
    if "steer" in parts:
        # phase B: the sweeps at each input's calibrated dose
        cal = [r["scene"] for r in S if r["scene"] < N_CAL] if len(S) >= 64 else [r["scene"] for r in S]
        dose_of = {}
        for inp in inputs:
            ri = R[(R["input"] == inp) & (R["task"] == "say") & (R["family"] == "steer") & (R["site"] == "media") & (R["key"] == "Y") & (R["set"] == "middle")]
            dose_of[inp] = next((dd for dd in DOSES if ri[ri["dose"] == dd].set_index("scene")["follows_Y"].reindex(cal).astype(float).mean() >= 0.5), 8.0)
        starts = list(range(-1, n_layers, 4))
        for inp in inputs:
            for g in [S[i : i + 16] for i in range(0, len(S), 16)]:
                encs = {(inp, t, r["scene"]): enc_for(inp, t, r) for t in STEER_TASKS for r in g}
                eng = engine(encs)
                rb = [dict(x, donor=None) for x in phase_b([inp], g, ID, starts, n_layers, dose_of)]
                for x in rb:
                    x["ids_x"], x["ids_y"] = ID[matched.truth(x["task"], x["X"], [])], ID[matched.truth(x["task"], x["Y"], [])]
                R = pd.concat([R, to_frame(rb, eng.generate(rb, N_NEW, f"{inp} sweeps"))], ignore_index=True)
        Rs = R[R["family"].isin(["clean", "steer", "knockout", "ablate", "sweep"])]
        AL = sorted(batched.attention_layers(lm))
        KO = [L for L in AL if (L - AL[0]) % 4 == 0] if len(AL) == n_layers else AL
        out["interventions"] = steer_summary(Rs, S, inputs, BL, starts, AL, KO, n_layers, N_CAL if len(S) >= 64 else 0, len(S) < 64)
        out["interventions_compare"] = steer_compare(Rs, inputs, BL, dose_of)
    if "patch" in parts:
        out["patch"] = patch_summary(R[R["family"].isin(["pclean", "self", "full", "transplant", "inject", "random"])], inputs, n_layers, family, kv)
    run.save_csv(R, "video_sweep.csv")
    gpu = mon.stop()
    out["setup"] = dict(setup, inputs=inputs, parts=sorted(parts), sampler=type(vp).__name__, kv_limit=kv, family=family, lens_part_rank=r_sub, gpu=gpu, seconds=time.time() - t0)
    run.save_json(out, "summary.json")
    if "patch" in out:
        for inp in inputs:
            print(f"patch {inp}: controls {out['patch']['controls'][inp]}, cut {out['patch']['cut'][inp]}, verdict {out['patch']['verdict'][inp]}, predictions {out['patch']['predictions'][inp]}")
    if gpu:
        print(f"GPU utilization: mean {gpu['mean_utilization']}%, median {gpu['median_utilization']}% over {gpu['samples']} samples")


if __name__ == "__main__":
    main()
