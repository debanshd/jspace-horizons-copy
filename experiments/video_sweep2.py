"""Video sweep 2: video_sweep's patching and interventions, rerun on the matched clips each model names reliably, shown whole.

Why. video_sweep (PR #5) found every verdict untestable: no model named the matched clips well enough, drawn into one slot of
the six-slot canvas, to pass its 0.90 naming gate. clip_naming, clip_naming2 and clip_naming3 (PR #6, results in PR #7) then
showed whole clips with a COCO-category question; clip_naming3 found per model the clips it names as a moving clip, without
crediting a hedge, and the scenes whose receiver and donor it names (its "gated schedule"): Gemma-4-E2B 50 scenes, Qwen3.5-4B
59, Gemma-4-E4B 57. Qwen3.5-0.8B's gate is clip_naming4 (the same rule with clip_naming's original question; its earlier
answers to that question keep at least 52 scenes). This run asks video_sweep's questions on that schedule only. Changing the scenes, the display
and the questions makes it a new experiment with its own run name; video_sweep's results stand as they are.

What changes from video_sweep (everything else is video_sweep's, unchanged and imported from it):
- Scenes: the model's gated schedule, read from runs/<--gate-run>/<model>/summary.json (as the pod script passes it:
  clip_naming4 for Qwen3.5-0.8B, clip_naming3 for the others; its gated_scenes, the moving input's schedule). A missing gate
  file stops the run. --n N takes the first N gated scenes for a smoke check (counts only, no verdicts).
- Display: each sampled frame whole, no canvas: fitted inside FRAME (672 x 378, 16:9) on the canvas's grey and centred, so the
  donor and the receiver of a scene give the same number of media tokens (patching writes states position by position, and
  clips of different sizes would not line up). Frames come from the model's own sampler as in video_sweep (the scene's count is
  the receiver's native count, capped at both clips' lengths). The gate measured naming at each clip's own size; the clean
  gate below measures it again at FRAME.
- Prompts: INTRO2 ("Study the scene."), the media, then one question about "the main object shown" instead of "the object in
  image slot q". say is the COCO question the model's naming gate asked: clip_naming2's, except clip_naming's original one for
  Qwen3.5-0.8B (SAY_BY_MODEL; under clip_naming2's it answered "animal", "food" or "vehicle" far more often); animal, letter,
  set and ride are depth_sweep's rules reworded the same way (QUESTIONS2). The neutral prompt for the donor's and the
  receiver's own states is INTRO2 and the media.
- Scoring: say (correct, follows the donor, follows Z) with clip_naming3's scoring: an answer that names one thing is scored
  with clip_naming2's rule (the label, its head noun in a phrase, its plural, or a kind or spelling of it; generic words are
  wrong); an answer that names two or more different things is a hedge, and that row is run once more, with the same
  intervention, on the same conversation continued by clip_naming3's FOLLOWUP question; only that reply is scored, and a
  second hedge is wrong. With N_NEW = 4 new tokens every say answer may be cut off, so a last fragment that begins an earlier
  name is dropped (clip_naming3's cut rule). The first answer is kept (first_answer). The other questions are scored as
  video_sweep scores them (the first character). The first-token log-probabilities are the question's, before any follow-up,
  with video_sweep's single-token names, as before.
- Calibration and verdicts: video_sweep's split by scene number (scenes below 16 calibrate the steering dose, the rest carry
  the verdicts), applied to the gated scenes; a smoke check is set by --n, not by having fewer than 64 scenes.

The rule (written before the run) is video_sweep's, unchanged: the patching gate (clean say >= 0.90, the animal rule >= 0.85),
the machinery check (self-patch >= 0.95), the positive control (every-layer donor patch, say follows the donor >= 0.9), the
horizons (last window start following the donor on >= 0.5 of scenes), the cut and verdicts, media_steer's frozen rule for the
other interventions, the moving/frozen/still comparisons, and the predictions P1 to P5 and media_steer's 1 to 6. With every
clip in the schedule named reliably at its own size, the positive control is scored against the donor's label, as before.

Outputs in runs/<run>/<model>/: video_sweep2.csv (every row), summary.json (as video_sweep's, plus the gate it read),
clips.json, stimuli.png, gpu.csv, config.json, run.log.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import clip_naming as naming1  # noqa: E402  the original say question, for Qwen3.5-0.8B
import clip_naming2 as naming  # noqa: E402  the say question
import clip_naming3 as naming3  # noqa: E402  the say scoring, with hedges asked again
import video_sweep as vs  # noqa: E402  everything not listed in the docstring is video_sweep's
from media_patch import kv_limit  # noqa: E402
from media_steer import DOSES, N_NEW, phase_a, phase_b  # noqa: E402
from media_steer import summarize as steer_summary  # noqa: E402

from jspace import batched, lens, matched, stimuli  # noqa: E402
from jspace import media as media_rules  # noqa: E402
from jspace.config import require_data  # noqa: E402
from jspace.model import Session  # noqa: E402
from jspace.runs import Run, parser  # noqa: E402

FRAME = (672, 378)  # width, height: 16:9 at the canvas's width
INTRO2 = "Study the scene."
QUESTIONS2 = {
    "say": naming.QUESTION,
    "animal": "Rule: if the main object shown is an animal, answer 7; otherwise answer 3. Reply with the number only.",
    "letter": "What is the first letter of the name of the main object shown? Reply with the letter only.",
    "set": "Rule: if the main object shown is a {a}, a {b} or a {c}, answer 7; otherwise answer 3. Reply with the number only.",
    "ride": "Rule: if the main object shown is something people ride or drive, answer 7; otherwise answer 3. Reply with the number only.",
}
assert list(QUESTIONS2) == list(vs.QUESTIONS)  # the same questions in the same order, so video_sweep's rows and summary apply
# the say question a model is asked when it is not clip_naming2's: the one its naming gate asked (clip_naming4 for Qwen3.5-0.8B)
SAY_BY_MODEL = {"Qwen/Qwen3.5-0.8B": naming1.QUESTION}
SETS = {}  # scene -> the set question's three names (filled in main; to_frame reads it)


def fit(images, size=FRAME):
    """each frame fitted whole inside `size` (aspect kept, no crop) and centred on the canvas grey: uint8 [n, h, w, 3]"""
    w, h = size
    out = []
    for im in images:
        im = im.convert("RGB")
        k = min(w / im.width, h / im.height)
        im = im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))), Image.Resampling.LANCZOS)
        bg = Image.new("RGB", (w, h), stimuli.BG)
        bg.paste(im, ((w - im.width) // 2, (h - im.height) // 2))
        out.append(np.asarray(bg))
    return np.stack(out)


def gated_scenes(gate_summary, n=0):
    """the scenes of the gate run's schedule, in order; the first n of them for a smoke check"""
    sc = list(gate_summary["gated_scenes"])
    if not sc:
        raise SystemExit("the gate run left no scenes for this model")
    return sc[:n] if n else sc


def question(lm, task):
    """the question text for this model: QUESTIONS2's, except the say question of a model listed in SAY_BY_MODEL"""
    return SAY_BY_MODEL.get(getattr(lm, "name", ""), QUESTIONS2["say"]) if task == "say" else QUESTIONS2[task]


def prompt(lm, inp, task, r):
    part = {"video": {"type": "video"}, "picture": {"type": "image"}}[matched.MEDIUM[inp.split("@")[0]]]
    content = [{"type": "text", "text": INTRO2}, part]
    if task != "neutral":
        content.append({"type": "text", "text": question(lm, task).format(a=r["set"][0], b=r["set"][1], c=r["set"][2])})
    return lm.chat_messages([{"role": "user", "content": content}])


def judge(task, text, word, membership):
    """does the answer give `word`'s answer to `task`: clip_naming3's scoring for say (a hedge gives no one's answer; with
    N_NEW = 4 any answer may be cut off), video_sweep's for the rules and letter"""
    if task == "say":
        return naming3.score(text, word, cut=True)[0]
    return matched.matches(task, text, matched.truth(task, word, membership))


def followup_prompt(lm, inp, first_answer):
    """the say conversation continued: the model's own answer, then clip_naming3's follow-up question"""
    part = {"video": {"type": "video"}, "picture": {"type": "image"}}[matched.MEDIUM[inp.split("@")[0]]]
    return lm.chat_messages(
        [
            {"role": "user", "content": [{"type": "text", "text": INTRO2}, part, {"type": "text", "text": question(lm, "say")}]},
            {"role": "assistant", "content": [{"type": "text", "text": first_answer}]},
            {"role": "user", "content": [{"type": "text", "text": naming3.FOLLOWUP}]},
        ]
    )


def ask_again(rows, res, encode_followup, run_rows, label):
    """each say row whose answer hedged, run once more with its own intervention on the follow-up conversation; the reply
    replaces the answer (the first answer is kept in the row as first_answer) and the question's log-probabilities stay.
    encode_followup(row, first_answer) gives the follow-up prompt's encoding; run_rows(encs, rows, label) generates."""
    idx = [i for i, (r, x) in enumerate(zip(rows, res)) if r["task"] == "say" and naming3.hedged(x[0], cut=True)]
    if not idx:
        return res
    encs, again = {}, []
    for k, i in enumerate(idx):
        r = dict(rows[i], task=f"say~{k}")  # its own prompt: the follow-up holds this row's answer
        encs[(r["input"], r["task"], r["scene"])] = encode_followup(rows[i], res[i][0])
        again.append(r)
    out = list(res)
    for i, x in zip(idx, run_rows(encs, again, f"{label}: {len(idx)} hedged answers asked again")):
        rows[i]["first_answer"] = res[i][0]
        out[i] = (x[0], res[i][1], res[i][2])
    return out


def to_frame(rows, res):
    recs = []
    for r, (text, lx, ly) in zip(rows, res):
        rec = {k: v for k, v in r.items() if k not in ("ids_x", "ids_y", "ko", "layers", "donor")}
        mem = SETS[r["scene"]]
        rec.update(
            layers=f"{r['layers'][0]}..{r['layers'][-1]}" if r["layers"] else "",
            donor="" if r["donor"] is None else f"{r['donor'][0]}",
            first_answer=r.get("first_answer", ""),
            generation=text,
            correct=judge(r["task"], text, r["X"], mem),
            follows_Y=judge(r["task"], text, r["Y"], mem),
            follows_Z=judge(r["task"], text, r["Z"], mem),
            lp_X=lx,
            lp_Y=ly,
            ld=lx - ly,
        )
        recs.append(rec)
    return pd.DataFrame(recs)


def summaries(R, S, inputs, parts, n_layers, AL, family, kv, BL, starts, dose_of, smoke):
    """video_sweep's summaries, with the smoke check set by --n rather than by having fewer than 64 scenes"""
    out = {}
    if "steer" in parts:
        Rs = R[R["family"].isin(["clean", "steer", "knockout", "ablate", "sweep"])]
        KO = [L for L in AL if (L - AL[0]) % 4 == 0] if len(AL) == n_layers else AL
        out["interventions"] = steer_summary(Rs, S, inputs, BL, starts, AL, KO, n_layers, 0 if smoke else vs.N_CAL, smoke)
        out["interventions_compare"] = vs.steer_compare(Rs, inputs, BL, dose_of, smoke)
    if "patch" in parts:
        out["patch"] = vs.patch_summary(R[R["family"].isin(["pclean", "self", "full", "transplant", "inject", "random"])], inputs, n_layers, family, kv)
    return out


def main():
    ap = parser(__doc__, "video_sweep2")
    ap.add_argument("--gate-run", required=True, help="the naming run whose gated schedule this model runs on (clip_naming3)")
    ap.add_argument("--n", type=int, default=0, help="the first N gated scenes, for a smoke check (counts only); 0 for all")
    ap.add_argument("--inputs", default=",".join(matched.INPUTS))
    ap.add_argument("--parts", default="patch,steer", help="patch, steer, or both")
    ap.add_argument("--donor-gb", type=float, default=16.0, help="host memory for donor states at a time: scenes run in chunks that fit")
    ap.add_argument("--max-rows", type=int, default=512)
    ap.add_argument("--no-encoder-cache", action="store_true")
    args = ap.parse_args()
    require_data(args.data_dir, "matched")
    gate_file = args.runs_dir / args.gate_run / args.model.split("/")[-1] / "summary.json"
    if not gate_file.exists():
        raise SystemExit(f"{gate_file} not found: run the naming gate ({args.gate_run}) for {args.model} first")
    gate = json.loads(gate_file.read_text())
    wanted = gated_scenes(gate, args.n)
    smoke = args.n > 0
    run = Run(args, "video_sweep2")
    lm = Session(args.model)
    run.write_config(lm)
    t0 = time.time()
    mon = batched.GpuMonitor(run.path("gpu.csv"))
    parts, inputs = set(args.parts.split(",")), args.inputs.split(",")
    by_id = {m["scene_id"]: m for m in matched.scenes(args.data_dir)}
    meta = [by_id[s] for s in wanted]
    root = args.data_dir / matched.DIR / "test"
    vp = lm.processor.video_processor

    # ---- per scene: the frame count and both clips' sampled frames, each fitted whole inside FRAME
    S, clips, media = [], {}, {}
    for m in meta:
        info = {role: matched.video_info(root / m[f"{role}_file_name"]) for role in ("donor", "receiver")}
        fps = info["receiver"][1]
        n = matched.scene_frame_count(vp, info["receiver"][0], info["donor"][0], fps)
        idx = {role: matched.native_indices(vp, info[role][0], info[role][1], n) for role in ("donor", "receiver")}
        frames = {role: fit(matched.decode(root / m[f"{role}_file_name"], idx[role])) for role in ("donor", "receiver")}
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
    run.save_json(dict(scenes=S, clips=clips, sampler=type(vp).__name__, frame=FRAME, gate=dict(run=args.gate_run, file=str(gate_file), scenes=wanted)), "clips.json")
    stimuli.contact_sheet(
        [(f"scene {S[0]['scene']} {role} frame {j}", Image.fromarray(media[(role, S[0]["scene"])][0][j])) for role in ("receiver", "donor") for j in range(min(4, clips[S[0]["scene"]]["frames"]))],
        4,
        run.path("stimuli.png"),
    )
    print(f"{len(S)} scenes from {args.gate_run}; frames per scene {sorted({c['frames'] for c in clips.values()})}  [{time.time() - t0:.0f}s]")

    ID = {w: lm.single_ids([w, " " + w, w.capitalize(), " " + w.capitalize()]) for w in matched.VOCAB}
    ID.update({c: lm.single_ids([c, " " + c]) for c in [*"7 3".split(), *sorted({w[0].upper() for w in matched.VOCAB})]})
    missing = [k for k, v in ID.items() if not v]
    assert not missing, f"no single-token form of {missing} in {lm.name}'s tokenizer"

    def enc_for(inp, task, r, role="receiver"):
        frames, md = media[(role, r["scene"])]
        return vs.encode(lm, inp, prompt(lm, inp, task, r), matched.input_of(frames, inp.split("@")[0]), md)

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

    budget = {}

    def engine(encs):
        # one token budget for the whole run, as in video_sweep (a budget measured again per engine moved the batch sizes)
        e = batched.Engine(lm, encs, token, medium, T, lm.layers, args.max_rows, budget_tokens=budget.get("tokens"))
        e.budget = budget.setdefault("tokens", max(2048, e.budget // 2048 * 2048))
        return e

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

    def encode_followup(r, first):
        frames, md = media[("receiver", r["scene"])]
        return vs.encode(lm, r["input"], followup_prompt(lm, r["input"], first), matched.input_of(frames, r["input"]), md)

    def run_rows(encs, rows, label):  # the follow-ups run in an engine of their own: each has its own prompt
        return engine(encs).generate(rows, N_NEW, label)

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
            engine(encs).capture([vs.row(inp, "say", r, ID) for r in part], LF, stats)
        T.sigma[inp] = {L: st["ms"][L] / st["n"][L] for L in LF}
        T.mu[inp] = {L: st["sum"][L] / st["n"][L] for L in lm.layers}
        T.dirs[(inp, "PC")] = st["pc"]
        ratio = {L: (st["dsw"][L] / st["n"][L]) / max(T.sigma[inp][L], 1e-12) for L in LF}
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
        for g in groups:
            encs = {}
            for r in g:
                for task in QUESTIONS2:
                    encs[(inp, task, r["scene"])] = enc_for(inp, task, r)
                encs[(inp + "@donor", "neutral", r["scene"])] = enc_for(inp + "@donor", "neutral", r, "donor")
                encs[(inp + "@self", "neutral", r["scene"])] = enc_for(inp + "@self", "neutral", r, "receiver")
                setup["layout"].setdefault(inp, {})[r["scene"]] = int((encs[(inp, "say", r["scene"])]["input_ids"] == token[inp]).sum())
            eng = engine(encs)
            if not args.no_encoder_cache and eng.build_encoder_cache(inp, [r["scene"] for r in g]):
                if cache_decided is None:
                    setup["cache"][inp] = eng.check_encoder_cache(inp, [vs.row(inp, "say", r, ID) for r in g[:4]], N_NEW)
                    cache_decided = setup["cache"][inp]["on"]
                eng.cache_on[inp] = cache_decided
            T.donor[inp] = {L: {} for L in LF}

            def keep(L, rows, v, inp=inp):
                for j, rr in enumerate(rows):
                    T.donor[inp][L][(rr["input"].split("@")[1], rr["scene"])] = v[j].half().cpu()

            eng.capture([vs.row(inp + "@donor", "neutral", r, ID) for r in g] + [vs.row(inp + "@self", "neutral", r, ID) for r in g], LF, keep)
            rows = []
            if "patch" in parts:
                rows += vs.patch_rows(inp, g, ID, n_layers)
            if "steer" in parts:
                AL = sorted(eng.attn)
                KO = [L for L in AL if (L - AL[0]) % 4 == 0] if len(AL) == n_layers else AL
                BL = media_rules.blocks(n_layers)
                ra = [dict(x, donor=None) for x in phase_a([inp], g, ID, BL, KO, AL)]
                for x in ra:  # media_steer's rows, with the matched set's truth (the 16 classes)
                    x["ids_x"], x["ids_y"] = ID[matched.truth(x["task"], x["X"], [])], ID[matched.truth(x["task"], x["Y"], [])]
                rows += ra
            label = f"{inp}: scenes {g[0]['scene']}..{g[-1]['scene']}"
            res = ask_again(rows, eng.generate(rows, N_NEW, label), encode_followup, run_rows, label)  # before the donor states go
            R_parts.append(to_frame(rows, res))
            del T.donor[inp]
            torch.cuda.empty_cache()
        print(f"{inp}: patching and phase A done  [{time.time() - t0:.0f}s]")

    R = pd.concat(R_parts, ignore_index=True)
    out = dict(n=len(S), smoke=smoke, gate=dict(run=args.gate_run, scenes=wanted))
    family, kv = ("gemma" if "gemma" in lm.name.lower() else "qwen"), kv_limit(lm)
    if "steer" in parts:
        # phase B: the sweeps at each input's calibrated dose (calibration: the gated scenes below 16, or every scene in a smoke check)
        cal = [r["scene"] for r in S] if smoke else [r["scene"] for r in S if r["scene"] < vs.N_CAL]
        if not cal:
            raise SystemExit(f"no gated scene below {vs.N_CAL} to calibrate the steering dose on")
        dose_of = {}
        for inp in inputs:
            ri = R[(R["input"] == inp) & (R["task"] == "say") & (R["family"] == "steer") & (R["site"] == "media") & (R["key"] == "Y") & (R["set"] == "middle")]
            dose_of[inp] = next((dd for dd in DOSES if ri[ri["dose"] == dd].set_index("scene")["follows_Y"].reindex(cal).astype(float).mean() >= 0.5), 8.0)
        starts = list(range(-1, n_layers, 4))
        for inp in inputs:
            for g in [S[i : i + 16] for i in range(0, len(S), 16)]:
                encs = {(inp, t, r["scene"]): enc_for(inp, t, r) for t in vs.STEER_TASKS for r in g}
                eng = engine(encs)
                rb = [dict(x, donor=None) for x in phase_b([inp], g, ID, starts, n_layers, dose_of)]
                for x in rb:
                    x["ids_x"], x["ids_y"] = ID[matched.truth(x["task"], x["X"], [])], ID[matched.truth(x["task"], x["Y"], [])]
                res = ask_again(rb, eng.generate(rb, N_NEW, f"{inp} sweeps"), encode_followup, run_rows, f"{inp} sweeps")
                R = pd.concat([R, to_frame(rb, res)], ignore_index=True)
    AL = sorted(batched.attention_layers(lm))
    out.update(summaries(R, S, inputs, parts, n_layers, AL, family, kv, BL, starts if "steer" in parts else None, dose_of if "steer" in parts else None, smoke))
    run.save_csv(R, "video_sweep2.csv")
    gpu = mon.stop()
    out["setup"] = dict(setup, inputs=inputs, parts=sorted(parts), sampler=type(vp).__name__, frame=FRAME, kv_limit=kv, family=family, lens_part_rank=r_sub, gpu=gpu, seconds=time.time() - t0)
    run.save_json(out, "summary.json")
    if "patch" in out:
        for inp in inputs:
            print(f"patch {inp}: controls {out['patch']['controls'][inp]}, cut {out['patch']['cut'][inp]}, verdict {out['patch']['verdict'][inp]}, predictions {out['patch']['predictions'][inp]}")
    if gpu:
        print(f"GPU utilization: mean {gpu['mean_utilization']}%, median {gpu['median_utilization']}% over {gpu['samples']} samples")


if __name__ == "__main__":
    main()
