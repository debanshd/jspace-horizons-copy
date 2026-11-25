"""Audio sweep: video_sweep2's patching and interventions on the matched spoken-audio set, on the scenes each model names.

Why. video_sweep2 (PR #8, results in PR #9) measured, per model, the layer by which a video's object stops being used ("is it
an animal?") and stops being nameable, and found the same depths for a moving video, a frozen frame and a still picture.
This run asks the same question of the spoken object names in saxenan3/jspace-matched-spoken-audio (src/jspace/spoken.py):
does the naming/rule separation carry over to spoken input, and does it follow the input's position in the prompt? Changing
the input to audio makes it a new experiment with its own run name; video_sweep2's results stand as they are.

What is video_sweep2's, unchanged and imported: the 64-scene schedule of the matched set (scene ids, receiver X, donor Y,
same-kind word Z, the set question's names), the patching arms (clean; the receiver's own states written back at every layer,
the machinery check; the donor's at every layer, the positive control; per 4-layer window from layer -1, the donor's states,
the receiver's own name's J-lens direction at 4 sigma, a random direction at 4 sigma), media_steer's phase A and B
interventions on say and animal, the batching, the rule, and the summaries.

What changes:
- Inputs: the two prompt orders of spoken.py. q_audio: the question, then the audio (Gemma-4's card's order). audio_q: the
  audio, then the question (the video experiments' order). The dataset asks for the order to be an explicit factor.
- Questions: the dataset's protocol.json wording (spoken.QUESTIONS), in place of "the main object shown".
- The donor's and the receiver's own states are captured under the receiver's exact question, per question (the dataset's
  protocol): with the question before the audio the audio's states can depend on it. Before any patching the run checks, per
  scene, question and input, that the donor's and the self prompt's audio tokens sit at exactly the receiver's positions; a
  mismatch stops the run (no state is interpolated).
- Scoring: src/jspace/strict.py, whole answer, first answer only (no hedge follow-up): say is correct when the answer is the
  spoken word itself; letter the one letter; set, animal, ride the one digit. An answer that used all N_NEW = 4 new tokens
  without ending is neither correct nor valid; every row records hit_limit, synonym (say: a plural, kind or spelling) and
  valid. audio_clean asks the same prompts with the same decoding, so its gate holds under these settings.
- Scenes: the model's gated schedule from audio_clean (--gate-run), the scenes whose receiver and donor recordings it names in
  both orders. The run stops unless audio_clean's summary is ready (its accuracy gates pass). --n N takes the first N for a
  pilot (counts only, no verdicts).
- Added rows: dclean, the donor's clean answer under the receiver's exact question (protocol.json's required baseline).
- controls_pass in summary.json: in every input, the clean say and animal gates on the pclean rows, the self patch (>= 0.95
  identical to clean) and the full donor patch (say follows the donor >= 0.9). The launcher runs the full sweep only after a
  5-scene patch-only pilot (--parts patch --n 5) whose controls pass, run twice with identical answers, and inspected.
- Recordings recur across scenes (15 recordings, one voice, each used in 2 to 18 of the 64 scenes): the intervals are
  bootstrapped over scenes, so they describe these scene assignments, not independent utterances, and are narrower than
  intervals over recordings would be. Any write-up of these results states this next to the intervals.

The rule (written before the run) is video_sweep's, applied to the whole-answer scores above: the patching gate (clean say >= 0.90, the animal rule >= 0.85),
the machinery check (self-patch >= 0.95), the positive control (every-layer donor patch, say follows the donor >= 0.9), the
horizons (the last window start following the donor on >= 0.5 of scenes), the cut and verdicts, P1 to P3, media_steer's frozen
rule and predictions 1 to 6 for the other interventions. P4 and P5 (moving against frozen and still) do not apply. Added: the
two orders agree when their horizons are within one window (media_patch.same_horizons), reported as order_compare.

Outputs in runs/<run>/<model>/: audio_sweep.csv (every row), summary.json (video_sweep2's, plus order_compare, controls_pass,
donor_clean, the share of answers at the token limit, the gate it read and the audio-token layout), scenes.json, stimuli.png (the first scene's two waveforms), gpu.csv, config.json, run.log.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import video_sweep as vs  # noqa: E402  rows, patch summary, steering comparison
import video_sweep2 as vs2  # noqa: E402  the summaries
from media_patch import kv_limit, same_horizons, windows  # noqa: E402
from media_steer import DOSES, N_NEW, has_audio, phase_a, phase_b  # noqa: E402

from jspace import batched, lens, matched, spoken, strict  # noqa: E402
from jspace import media as media_rules  # noqa: E402
from jspace.analysis import boot_scene  # noqa: E402
from jspace.config import require_data  # noqa: E402
from jspace.model import Session  # noqa: E402
from jspace.runs import Run, parser  # noqa: E402

QUESTIONS_A = dict(spoken.QUESTIONS)
assert list(QUESTIONS_A) == list(vs.QUESTIONS)  # the same questions in the same order, so video_sweep's rows and summary apply


# audio_sweep_static's inputs beside q_audio, the counterparts of video_sweep2's frozen video and still picture: the recording
# with its spoken span frozen (spoken.frozen), and the word typed as a transcript; both with the question first
STATIC = {"q_frozen": "frozen", "q_typed": "typed"}
INPUTS_ALL = spoken.INPUTS + list(STATIC)
TYPED = "Transcript of the audio: {word}"


def prompt_messages(inp, task, r, word=None):
    """the user turn of an input (or of a captured state of it: "q_typed@donor~say"); the typed input needs the word it types"""
    base = inp.split("@")[0]
    q = QUESTIONS_A[task].format(a=r["set"][0], b=r["set"][1], c=r["set"][2])
    if base == "q_typed":
        assert word, "the typed input types a word"
        return [{"role": "user", "content": [{"type": "text", "text": q}, {"type": "text", "text": TYPED.format(word=word)}]}]
    return [{"role": "user", "content": spoken.content("q_audio" if base in STATIC else base, q)}]


def audio_for(inp, wav):
    """what an input plays: the recording, its frozen version, or nothing (typed)"""
    base = inp.split("@")[0]
    return None if base == "q_typed" else spoken.frozen(wav) if base == "q_frozen" else wav


def typed_span(enc_receiver, enc_donor):
    """the typed word's place, from the end of the prompt: the one position where the receiver's and the donor's typed prompts
    differ (each word one token); anything else stops the run"""
    a, b = enc_receiver["input_ids"][0], enc_donor["input_ids"][0]
    if a.shape != b.shape:
        raise SystemExit(f"the typed prompts differ in length ({a.shape[0]} and {b.shape[0]}): a word is not one token")
    d = torch.nonzero(a != b).flatten().tolist()
    if len(d) != 1:
        raise SystemExit(f"the typed prompts differ at {d}, not at one position")
    return batched.Span([d[0] - a.shape[0]])


def capture_input(inp, role, task):
    """the input name of a captured state: its role and the question it was captured under ("q_audio@donor~say")"""
    return f"{inp}@{role}~{task}"


def patch_rows(inp, S, ID, n_layers):
    """video_sweep's patching rows, the donor's and the self states taken under each row's own question"""
    rows = []
    every = tuple(range(-1, n_layers))
    for task in QUESTIONS_A:
        for r in S:
            own, don = (f"self~{task}", r["scene"]), (f"donor~{task}", r["scene"])
            rows.append(vs.row(inp, task, r, ID, family="pclean", arm="clean"))
            rows.append(vs.row(capture_input(inp, "donor", task), task, r, ID, family="dclean", arm="donor clean, receiver's question"))
            rows.append(vs.row(inp, task, r, ID, family="self", arm="own states, every layer", site="patch", layers=every, donor=own))
            rows.append(vs.row(inp, task, r, ID, family="full", arm="donor states, every layer", site="patch", layers=every, donor=don))
            for w in windows(n_layers):
                rows.append(vs.row(inp, task, r, ID, family="transplant", arm=f"transplant, layers {w[0]} to {w[-1]}", site="patch", layers=tuple(w), set=str(w[0]), donor=don))
                rows.append(vs.row(inp, task, r, ID, family="inject", arm=f"inject own name, layers {w[0]} to {w[-1]}", site="media", key="dsw_own", layers=tuple(w), set=str(w[0]), dose=4.0))
                rows.append(vs.row(inp, task, r, ID, family="random", arm=f"inject random, layers {w[0]} to {w[-1]}", site="media", key="dsw_random", layers=tuple(w), set=str(w[0]), dose=4.0))
    return rows


def to_frame(rows, res):
    """one record per row, scored whole-answer on the first answer (strict.py), with hit_limit"""
    recs = []
    for r, (text, lx, ly, hit) in zip(rows, res):
        rec = {k: v for k, v in r.items() if k not in ("ids_x", "ids_y", "ko", "layers", "donor")}
        mem = vs2.SETS[r["scene"]]
        sc = strict.score(r["task"], text, r["X"], mem, hit)
        rec.update(
            layers=f"{r['layers'][0]}..{r['layers'][-1]}" if r["layers"] else "",
            donor="" if r["donor"] is None else f"{r['donor'][0]}",
            generation=text,
            hit_limit=hit,
            correct=sc["correct"],
            synonym=sc["synonym"],
            valid=sc["valid"],
            follows_Y=strict.matches(r["task"], text, r["Y"], mem, hit),
            follows_Z=strict.matches(r["task"], text, r["Z"], mem, hit),
            lp_X=lx,
            lp_Y=ly,
            ld=lx - ly,
        )
        recs.append(rec)
    return pd.DataFrame(recs)


def controls_pass(patch, inputs):
    """every input: the clean say and animal gates, the self patch and the full donor patch"""
    ok = lambda i: all(patch["gate"][i][t]["passes"] for t in vs.GATE) and patch["controls"][i]["machinery_ok"] and patch["controls"][i]["positive_ok"]  # noqa: E731
    return {i: bool(ok(i)) for i in inputs} | {"all": bool(all(ok(i) for i in inputs))}


def donor_clean(R, inputs):
    """per input and question: the share of dclean rows giving the donor's answer"""
    out = {}
    for inp in inputs:
        d = R[(R["family"] == "dclean") & (R["input"].str.startswith(f"{inp}@donor~"))]
        out[inp] = {t: float(d[d["task"] == t]["follows_Y"].astype(float).mean()) for t in QUESTIONS_A}
    return out


def audio_positions(enc, token_id):
    """the media positions of a prompt: its audio tokens, or a Span's places"""
    return batched.media_positions(enc["input_ids"][0].tolist(), token_id)


def check_alignment(encs, inp, scenes, token_id):
    """per scene and question: the donor's and the self prompt's audio tokens at exactly the receiver's positions. Returns the
    audio-token count per question; stops the run on any mismatch"""
    counts = {}
    for task in QUESTIONS_A:
        for s in scenes:
            p = audio_positions(encs[(inp, task, s)], token_id)
            for role in ("donor", "self"):
                e = encs[(capture_input(inp, role, task), task, s)]
                q = audio_positions(e, token_id)
                if p != q or not p:
                    raise SystemExit(f"{inp} scene {s} {task}: the {role} prompt's audio tokens ({len(q)}) are not at the receiver's positions ({len(p)})")
                if isinstance(token_id, batched.Span):  # a typed word: the same prompt but for the word, which sits at the span
                    a, b = encs[(inp, task, s)]["input_ids"][0], e["input_ids"][0]
                    d = torch.nonzero(a != b).flatten().tolist() if a.shape == b.shape else None
                    if d is None or not set(d) <= set(p):
                        raise SystemExit(f"{inp} scene {s} {task}: the {role} prompt differs from the receiver's outside the typed word")
            counts.setdefault(task, set()).add(len(p))
    return {t: sorted(c) for t, c in counts.items()}


def order_compare(patch):
    """the two prompt orders agree when their horizons are within one window (media_patch.same_horizons)"""
    cut = patch["cut"]
    if not all(isinstance(cut.get(i), dict) for i in spoken.INPUTS):
        return "untestable"
    return dict(same=bool(same_horizons(cut["q_audio"], cut["audio_q"])), q_audio=cut["q_audio"], audio_q=cut["audio_q"])


def static_compare(patch, inputs):
    """video_sweep's P4 and P5 for the audio counterparts: the spoken word against its frozen version (P4) and against the typed
    word (P5), kept when their horizons are within one window (media_patch.same_horizons); written into q_audio's predictions"""
    out = {}
    for key, other in (("P4", "q_frozen"), ("P5", "q_typed")):
        if "q_audio" not in inputs or other not in inputs:
            continue
        a, b = patch["cut"].get("q_audio"), patch["cut"].get(other)
        out[key] = dict(other=other, verdict=("kept" if same_horizons(a, b) else "missed") if isinstance(a, dict) and isinstance(b, dict) else "untestable")
        if isinstance(patch["predictions"].get("q_audio"), dict):
            patch["predictions"]["q_audio"][key] = out[key]["verdict"]
    return out


def steer_static_compare(R, inputs, BL, dose_of, smoke):
    """video_sweep.steer_compare with q_audio for the moving video: the paired say-follows-Y difference per block at q_audio's
    dose, on the held-out scenes (>= N_CAL), "same" when the 95% interval lies within +-0.15 in every block"""
    if smoke:
        return "smoke check: counts only, no comparison"
    if "q_audio" not in inputs:
        return {}
    held = np.sort(R[R["scene"] >= vs.N_CAL]["scene"].unique())
    if not len(held):
        raise ValueError(f"no held-out scenes (scene >= {vs.N_CAL}) to compare on")
    out = {}
    for other in [i for i in STATIC if i in inputs]:
        e = {}
        for bn in BL:
            v = {}
            for inp in ("q_audio", other):
                d = R[(R["input"] == inp) & (R["task"] == "say") & (R["family"] == "steer") & (R["site"] == "media") & (R["key"] == "Y") & (R["set"] == bn) & (R["dose"] == dose_of["q_audio"])]
                v[inp] = d.set_index("scene")["follows_Y"].astype(float).reindex(held).to_numpy()
            bd = boot_scene(v["q_audio"] - v[other], held)
            e[bn] = dict(diff=bd, same=bool(bd[1] >= -0.15 and bd[2] <= 0.15))
        e["verdict"] = "same" if all(e[bn]["same"] for bn in BL) else "differs"
        out[f"q_audio vs {other}"] = e
    return out


def waveform_png(wavs, labels, path, w=1200, h=160):
    """the waveforms, one strip each, labelled: what the model heard"""
    im = Image.new("RGB", (w, h * len(wavs)), (247, 246, 242))
    d = ImageDraw.Draw(im)
    for k, (x, lab) in enumerate(zip(wavs, labels)):
        y0 = k * h + h // 2
        step = max(1, len(x) // w)
        env = np.abs(x[: step * w]).reshape(w, step).max(1)
        for i, a in enumerate(env):
            d.line([(i, y0 - a * (h // 2 - 8)), (i, y0 + a * (h // 2 - 8))], fill=(42, 120, 214))
        d.text((8, k * h + 6), lab, fill=(20, 33, 61))
    im.save(path)


def both_orders(args, gate, gate_dir):
    """audio_sweep's schedule: audio_clean's ready (both orders) and its gated scenes; nothing added to the gate record"""
    if not gate.get("ready"):
        raise SystemExit(f"{gate_dir / 'summary.json'}: the clean answers do not pass audio_clean's gates (ready is false); nothing to patch")
    return vs2.gated_scenes(gate, args.n), {}


BOTH = ",".join(spoken.INPUTS)


def main(experiment="audio_sweep", doc=__doc__, schedule=both_orders, default_inputs=BOTH):
    """the sweep; audio_sweep_qfirst gives its own name, docstring, schedule (one order's gate) and inputs"""
    ap = parser(doc, experiment)
    ap.add_argument("--gate-run", default="audio_clean", help="the audio_clean run whose gated schedule this model runs on")
    ap.add_argument("--n", type=int, default=0, help="the first N gated scenes, for a pilot (counts only); 0 for all")
    ap.add_argument("--inputs", default=default_inputs)
    ap.add_argument("--parts", default="patch,steer", help="patch, steer, or both")
    ap.add_argument("--donor-gb", type=float, default=16.0, help="host memory for donor states at a time: scenes run in chunks that fit")
    ap.add_argument("--max-rows", type=int, default=512)
    args = ap.parse_args()
    require_data(args.data_dir, "spoken")
    gate_file = args.runs_dir / args.gate_run / args.model.split("/")[-1] / "summary.json"
    if not gate_file.exists():
        raise SystemExit(f"{gate_file} not found: run audio_clean for {args.model} first")
    gate = json.loads(gate_file.read_text())
    wanted, gate_extra = schedule(args, gate, gate_file.parent)
    smoke = args.n > 0
    run = Run(args, experiment)
    lm = Session(args.model)
    if not has_audio(lm):
        raise SystemExit(f"{lm.name} takes no audio")
    run.write_config(lm)
    t0 = time.time()
    mon = batched.GpuMonitor(run.path("gpu.csv"))
    parts, inputs = set(args.parts.split(",")), args.inputs.split(",")
    assert set(inputs) <= set(INPUTS_ALL), inputs
    root = args.data_dir / spoken.DIR
    by_id = {m["scene_id"]: m for m in spoken.scenes(args.data_dir)}
    meta = [by_id[s] for s in wanted]

    # ---- per scene: both recordings, as the processor receives them
    S, media, used = [], {}, {}
    for m in meta:
        sc = m["scene_id"]
        r = dict(scene=sc, X=m["receiver_label"], Y=m["donor_label"], Z=matched.same_kind_word(m["receiver_label"], m["donor_label"], sc), clip=m["receiver_audio_id"], set=list(m["membership"]))
        S.append(r)
        vs2.SETS[sc] = r["set"]
        for role in ("donor", "receiver"):
            media[(role, sc)] = spoken.load_wav(root / m[f"{role}_audio"])
            used[m[f"{role}_audio_id"]] = m[f"{role}_sha256"]
    run.save_json(dict(scenes=S, recordings=used, gate=dict(run=args.gate_run, file=str(gate_file), scenes=wanted, **gate_extra)), "scenes.json")
    shown = [media[("receiver", S[0]["scene"])], media[("donor", S[0]["scene"])]]
    names = [f"scene {S[0]['scene']} receiver: {S[0]['X']}", f"donor: {S[0]['Y']}"]
    if "q_frozen" in inputs:  # what the frozen input plays
        shown, names = shown + [spoken.frozen(shown[0])], names + [f"receiver, frozen: {S[0]['X']}"]
    waveform_png(shown, names, run.path("stimuli.png"))
    print(f"{len(S)} scenes from {args.gate_run}; {len(used)} recordings  [{time.time() - t0:.0f}s]")

    ID = {w: lm.single_ids([w, " " + w, w.capitalize(), " " + w.capitalize()]) for w in matched.VOCAB}
    ID.update({c: lm.single_ids([c, " " + c]) for c in [*"7 3".split(), *sorted({w[0].upper() for w in matched.VOCAB})]})
    missing = [k for k, v in ID.items() if not v]
    assert not missing, f"no single-token form of {missing} in {lm.name}'s tokenizer"

    def encode(messages, wav):
        text = lm.chat_messages(messages)
        enc = lm.processor(text=[text], audio=[wav], return_tensors="pt") if wav is not None else lm.processor(text=[text], return_tensors="pt")
        return {k: (v.to(lm.dtype) if torch.is_tensor(v) and k in batched.FLOAT_KEYS else v) for k, v in enc.items()}

    def enc_for(inp, task, r, role="receiver"):
        word = r["Y"] if role == "donor" else r["X"]  # a self prompt is the receiver's own
        return encode(prompt_messages(inp, task, r, word), audio_for(inp, media[(role, r["scene"])]))

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
    tid = lm.audio_token_id
    token, medium = {}, {}
    for inp in inputs:
        # the typed input has no audio token: its media position is the typed word's place (a batched.Span)
        tok = typed_span(enc_for(inp, "say", S[0]), enc_for(inp, "say", S[0], "donor")) if inp == "q_typed" else tid
        med = "text" if inp == "q_typed" else "sound"
        token[inp], medium[inp] = tok, med
        for role in ("donor", "self"):
            for task in QUESTIONS_A:
                token[capture_input(inp, role, task)], medium[capture_input(inp, role, task)] = tok, med

    budget = {}

    def engine(encs):
        # one token budget for the whole run, as in video_sweep (a budget measured again per engine moved the batch sizes)
        e = batched.Engine(lm, encs, token, medium, T, lm.layers, args.max_rows, budget_tokens=budget.get("tokens"))
        e.budget = budget.setdefault("tokens", max(2048, e.budget // 2048 * 2048))
        return e

    R_parts, setup = [], dict(layout={}, chunks={})
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

        # pass 2: per chunk of scenes, the donor's and self states under every question, then every patching row and phase A
        tok0 = len(audio_positions(enc_for(inp, "say", S[0]), token[inp]))
        gb = 2 * len(QUESTIONS_A) * tok0 * d * len(LF) * 2 / 2**30  # both roles, every question, every layer, fp16

        def chunks():
            per = max(1, int(args.donor_gb // gb))
            return [S[i : i + per] for i in range(0, len(S), per)]

        groups = chunks()
        setup["chunks"][inp] = [len(g) for g in groups]
        for g in groups:
            encs = {}
            for r in g:
                for task in QUESTIONS_A:
                    encs[(inp, task, r["scene"])] = enc_for(inp, task, r)
                    encs[(capture_input(inp, "donor", task), task, r["scene"])] = enc_for(capture_input(inp, "donor", task), task, r, "donor")
                    encs[(capture_input(inp, "self", task), task, r["scene"])] = enc_for(capture_input(inp, "self", task), task, r, "receiver")
            for task, c in check_alignment(encs, inp, [r["scene"] for r in g], token[inp]).items():
                setup["layout"].setdefault(inp, {}).setdefault(task, set()).update(c)
            eng = engine(encs)
            T.donor[inp] = {L: {} for L in LF}

            def keep(L, rows, v, inp=inp):
                for j, rr in enumerate(rows):
                    T.donor[inp][L][(rr["input"].split("@")[1], rr["scene"])] = v[j].half().cpu()

            eng.capture([vs.row(capture_input(inp, role, task), task, r, ID) for role in ("donor", "self") for task in QUESTIONS_A for r in g], LF, keep)
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
            label = f"{inp}: scenes {g[0]['scene']}..{g[-1]['scene']}"
            R_parts.append(to_frame(rows, eng.generate(rows, N_NEW, label, hit_limit=True)))  # before the donor states go
            del T.donor[inp]
            torch.cuda.empty_cache()
        print(f"{inp}: patching and phase A done  [{time.time() - t0:.0f}s]")

    R = pd.concat(R_parts, ignore_index=True)
    out = dict(n=len(S), smoke=smoke, gate=dict(run=args.gate_run, scenes=wanted, **gate_extra))
    family, kv = ("gemma" if "gemma" in lm.name.lower() else "qwen"), kv_limit(lm)
    starts = dose_of = None
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
                R = pd.concat([R, to_frame(rb, eng.generate(rb, N_NEW, f"{inp} sweeps", hit_limit=True))], ignore_index=True)
    AL = sorted(batched.attention_layers(lm))
    out.update(vs2.summaries(R, S, inputs, parts, n_layers, AL, family, kv, BL, starts, dose_of, smoke))
    if "patch" in out:
        out["order_compare"] = order_compare(out["patch"])
        out["static_compare"] = static_compare(out["patch"], inputs)
        out["controls_pass"] = controls_pass(out["patch"], inputs)
        out["donor_clean"] = donor_clean(R, inputs)
    if "steer" in parts:
        out["steer_static_compare"] = steer_static_compare(R[R["family"].isin(["clean", "steer", "knockout", "ablate", "sweep"])], inputs, BL, dose_of, smoke)
    out["hit_limit"] = float(R["hit_limit"].astype(float).mean())
    run.save_csv(R, "audio_sweep.csv")
    gpu = mon.stop()
    layout = {i: {t: sorted(c) for t, c in v.items()} for i, v in setup["layout"].items()}
    out["setup"] = dict(chunks=setup["chunks"], audio_tokens=layout, inputs=inputs, parts=sorted(parts), kv_limit=kv, family=family, lens_part_rank=r_sub, gpu=gpu, seconds=time.time() - t0)
    run.save_json(out, "summary.json")
    if "patch" in out:
        for inp in inputs:
            print(f"patch {inp}: controls {out['patch']['controls'][inp]}, cut {out['patch']['cut'][inp]}, verdict {out['patch']['verdict'][inp]}, predictions {out['patch']['predictions'][inp]}")
        print(f"orders: {out['order_compare']}; donor clean {out['donor_clean']}; controls pass {out['controls_pass']}")
    if gpu:
        print(f"GPU utilization: mean {gpu['mean_utilization']}%, median {gpu['median_utilization']}% over {gpu['samples']} samples")


if __name__ == "__main__":
    main()
