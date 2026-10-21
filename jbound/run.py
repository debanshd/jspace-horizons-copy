#!/usr/bin/env python
"""Run the articulation-bound cells (jspace_bound_cells.py, copied to cells_runpod.py) as a script on one model, with progress bars.

    jbound/jb --model Qwen/Qwen3.5-4B --cells M0,B1,B2,B3,B4,B5,B6 --tag run1
    jbound/jb --model Qwen/Qwen3.5-4B --tag smoke --smoke                       # tiny: 3 scenes per k, ks {1, 2, 8}
    jbound/jb --model google/gemma-4-E4B-it --cells B7 --tag run1
    jbound/jb --model Qwen/Qwen3.5-4B --cells B5,B6 --tag run1 --aggregate-only --other-summaries '/workspace/nayan/runs/run1/*/eB5_summary_*.json'

The setup (jbound/setup_runpod.py: the notebook's model, lens and setup cells) runs first, then each requested cell in order in the
same namespace, the way the notebook ran them. Output goes to /workspace/nayan/runs/<tag>/<model>/ with run.log, config.json,
verdicts.json and the cells' own e*.json and figures/. A cell that needs another cell's definitions (B7 uses M0, B1 and B3 helpers)
gets them without that cell's forwards. --aggregate-only loads this model's saved eB1..eB4 results instead of running them, so B5 and
B6 can draw across every model in the tag.
"""
import argparse, datetime, glob, hashlib, io, json, os, re, sys, time, traceback

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

HERE = os.path.dirname(os.path.abspath(__file__))
CELLS = os.path.join(HERE, "cells_runpod.py")
CELLS_E = os.path.join(HERE, "cells_e.py")
SETUP = os.path.join(HERE, "setup_runpod.py")
DEPS = {"B1": ["M0"], "B2": ["M0", "B1"], "B3": ["M0", "B1"], "B4": ["M0", "B1", "B3"], "B5": ["M0", "B1", "B3", "B4"], "B6": ["M0", "B1", "B3", "B4", "B5"], "B7": ["M0", "B1", "B3"], "B3s": ["M0", "B1", "B3"], "E0": ["M0", "B1", "B3"], "E1": ["M0", "B1", "B3", "E0"], "E2": ["M0", "B1", "B3", "E0", "E1"], "E2P": ["M0", "B1", "B3", "E0", "E1", "E2"], "E2u": ["M0", "B1", "B3", "E0", "E1", "E2"], "N1": ["M0", "B1", "B2", "B3", "E0", "E1", "E2P"], "T1": ["M0", "B1", "B2", "B3", "E0", "E1", "E2", "E2P", "N1"], "R1": ["M0", "B1", "B2", "B3", "E0", "E1", "E2P", "N1"], "E2Pd": ["M0", "B1", "B3", "E0", "E1", "E2", "E2P"], "L1": ["M0", "B1", "B2", "B3", "E0", "E1", "E2", "E2P", "N1", "T1"], "A1": ["M0", "B1", "B2", "B3", "E0", "E1", "E2", "E2P", "N1", "T1"], "A2": ["M0", "B1", "B2", "B3", "E0", "E1", "E2P", "N1"], "F1": ["M0", "B1", "B2", "B3", "E0", "E1", "E2", "E2P", "N1"], "F2": ["M0", "B1", "B2", "B3", "E0", "E1", "E2", "E2P", "N1", "T1"], "G1": ["M0", "B1", "B2", "B3", "E0", "E1", "E2", "E2P", "N1", "T1"], "F3": ["M0", "B1", "B2", "B3", "E0", "E1", "E2", "E2P", "N1", "T1"], "G1b": ["M0", "B1", "B2", "B3", "E0", "E1", "E2", "E2P", "N1", "T1"], "F4": ["M0", "B1", "B2", "B3", "E0", "E1", "E2", "E2P", "N1", "T1"], "D1": ["M0", "B1", "B2", "B3", "E0", "E1", "E2", "E2P", "N1", "T1", "G1"], "K1": ["M0", "B1", "B2", "B3", "E0", "E1", "E2", "E2P", "N1", "T1", "G1"], "S1": ["M0", "B1", "B2", "B3", "E0", "E1", "E2", "E2P", "N1", "T1", "G1"], "D2": ["M0", "B1", "B2", "B3", "E0", "E1", "E2", "E2P", "N1", "T1", "G1"], "O1": ["M0", "B1", "B2", "B3", "E0", "E1", "E2", "E2P", "N1", "T1", "G1"]}
TITLES = {"M0": "picture helpers and the vision preflight", "B1": "the bound: items in J-space at the report, by modality and load",
          "B2": "the collapse: where the dropped items are, and the additive ceiling", "B3": "carried: what the model uses from a scene it can no longer see",
          "B3s": "carried from the scene state: the carry test on the scene positions, with the lens / random / complement split", "B4": "photographs: the same curve and bound on COCO under POPE", "B5": "the slope: the same numbers across model sizes",
          "B6": "the frontier: items said against items used, and the articulation ratio", "B7": "omni: one vocabulary through text, speech, picture, video and sound", "E0": "stimuli: COCO crops in numbered slots and the same scene as text", "E1": "relevance: one position / any position / joint, image and text", "E2": "cross-modal competition: image grid + text list, six arms, loads 1-6", "E2P": "causal interchange: one slot's state swapped (full / J part / random / complement / non-target)", "E2u": "cross-modal competition with an unindexed text source (membership probes)", "N1": "necessity of the J-readable image-token component on GQA (ablation arms, judge-scored)", "T1": "transport cost by modality: the J part of the evidence tokens ablated, picture vs text, the same scenes", "R1": "reportability: does the model's own description state the facts it uses (judge-scored entailment)", "E2Pd": "interchange machinery diagnostic: self-swap exactness, donor swap generations", "L1": "lens calibration: true downstream gain of lens-selected directions on image vs text tokens", "A1": "label vs object: one word direction decides whether the model reads the printed label or sees the object", "A2": "SCAM: real photos with a hand-written note; one direction decides what the model reports; injection dose curve", "F1": "fact reuse: a donor's lens part transplanted into a receiver that never saw the picture and gets a new rule", "F2": "two facts seen in a picture, used by a rule chosen after encoding; picture and printed donors; slices at the read layers and at every layer", "G1": "the name as the handle: one direction removed or injected on the slot tokens or at the answer; say, letter, animal, ride", "O1": "where the lens first reads the object name, against the fact horizon", "D2": "the reverse by timing: donor before the use horizon, receiver restored in the band; four-frame video donor", "S1": "a rule over names against a rule over the referent: which follows the moved or injected name", "K1": "the load law: the name direction's necessity as the scene fills, k = 1 to 6", "D1": "the depth law: in which layers a name is still said and still used; plus the word animal injected", "F4": "the name without the fact on fresh crops: placebo donor, joint question, r = 0.1 d, every-layer control", "G1b": "exploratory: the printed word's rules phrased for a word instead of an object", "F3": "fact reuse with a competing receiver: the receiver sees its own object; lens, random (three draws), complement and full transplants"}

# --smoke: the same cells on a few scenes, so every assertion, shape and figure runs before the real run. Never written to a real run folder
SMOKE_SUBS = {
    "B1": [("B_KS, B_PER_K = [1, 2, 3, 4, 6, 8], 24", "B_KS, B_PER_K = [1, 2, 8], 3")],
    "B2": [('            assert len(dk) >= 6, f"too few k = {k} scenes with single-item vectors for every shape ({len(dk)})"',
            '            if len(dk) == 0:\n                print(f"smoke: no k = {k} scenes with single-item vectors for every shape, additive skipped at this k")\n'
            '                additive[rend][k] = dict(sum=(np.nan,) * 3, sum_absent=(np.nan,) * 3, measured=(np.nan,) * 3, n=0)\n                continue')],
    "B3": [('B_R, B_SPLIT_KS, B_SPLIT_RENDS = 256, [4, 8], ["text", "picture"]', 'B_R, B_SPLIT_KS, B_SPLIT_RENDS = 256, [8], ["text", "picture"]')],
    "B4": [("B4_N_PER_K, B4_KS = 20, [1, 2, 3, 4, 6, 8]", "B4_N_PER_K, B4_KS = 3, [1, 2, 8]")],
    "B7": [("B7_KS, B7_PER_K = [1, 2, 3, 4, 6, 8], 16", "B7_KS, B7_PER_K = [1, 2, 8], 2")],
    "E0": [("E_N_SCENES = 64", "E_N_SCENES = 4")],
    "E2": [("E2_KS = [1, 2, 4, 6]", "E2_KS = [1, 4]")],
    "E2P": [("E2P_N_PAIRS = 32", "E2P_N_PAIRS = 4")],
    "N1": [('N1_N_IMAGES = int(os.environ.get("N1_N_IMAGES", 16))', "N1_N_IMAGES = 3")],
    "T1": [('T1_N = int(os.environ.get("T1_N", E_N_SCENES))', "T1_N = 4")],
    "A1": [('A1_N = int(os.environ.get("A1_N", E_N_SCENES))', "A1_N = 4")],
    "A2": [('A2_N = int(os.environ.get("A2_N", 64))', "A2_N = 4")],
    "F1": [('F1_N = int(os.environ.get("F1_N", E_N_SCENES))', "F1_N = 4")],
    "F2": [('F2_N = int(os.environ.get("F2_N", E_N_SCENES))', "F2_N = 4")],
    "G1": [('G1_N = int(os.environ.get("G1_N", E_N_SCENES))', "G1_N = 4"), ('G1_CAL = int(os.environ.get("G1_CAL", 16))', "G1_CAL = 2")],
    "F3": [('F3_N = int(os.environ.get("F3_N", E_N_SCENES))', "F3_N = 4")],
    "G1b": [('G1B_N = int(os.environ.get("G1B_N", E_N_SCENES))', "G1B_N = 4")],
    "F4": [('F4_N = int(os.environ.get("F4_N", E_N_SCENES))', "F4_N = 4")],
    "D1": [('D1_N = int(os.environ.get("D1_N", E_N_SCENES))', "D1_N = 4")],
    "K1": [('K1_N = int(os.environ.get("K1_N", E_N_SCENES))', "K1_N = 4")],
    "S1": [('S1_N = int(os.environ.get("S1_N", E_N_SCENES))', "S1_N = 4")],
    "D2": [('D2_N = int(os.environ.get("D2_N", E_N_SCENES))', "D2_N = 4")],
    "O1": [('O1_N = int(os.environ.get("O1_N", E_N_SCENES))', "O1_N = 4")],
}


class Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()
        return len(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def split_cells(src):
    """{name: source} for every '# %% ---- NAME: ...' cell, in file order"""
    out, name, buf = {}, None, []
    for line in src.splitlines(keepends=True):
        m = re.match(r"# %% -+ (\w+)[:\s]", line)
        if m:
            if name:
                out[name] = "".join(buf)
            name, buf = m.group(1), [line]
        elif name:
            buf.append(line)
    if name:
        out[name] = "".join(buf)
    return out


def definitions(src):
    """the part of a cell before its '# --- reads' marker: constants and functions, no forwards"""
    return src.split("\n# --- reads")[0] + "\n"


def intkeys(o):
    """JSON back to the cells' dicts: keys that are integers become int again"""
    if isinstance(o, dict):
        return {(int(k) if isinstance(k, str) and re.fullmatch(r"-?\d+", k) else k): intkeys(v) for k, v in o.items()}
    if isinstance(o, list):
        return [intkeys(v) for v in o]
    return o


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:12]


def fmt(t):
    try:
        return f"{t[0]:.2f} [{t[1]:.2f}, {t[2]:.2f}]"
    except Exception:
        return str(t)


def verdicts(ns):
    """one row per finished cell, from the live results"""
    rows = []
    r1, r2, r3, r4, r5, r6, r7 = (ns.get(k) or {} for k in ("resB1", "resB2", "resB3", "resB4", "resB5", "resB6", "resB7"))
    if r1.get("verdict"):
        rows.append(("B1", r1["verdict"], "K: " + ", ".join(f"{r} {v:.1f}" for r, v in r1["K"].items()) + "; " + r1["copying"] + "; seen: " + ", ".join(f"{r} {'ok' if row['seen'] else 'NO'}" for r, row in r1["by_rendering"].items())))
    if r2.get("where_verdict"):
        rows.append(("B2", r2["where_verdict"], "additive: " + ", ".join(f"{r} {v['verdict'].split(':')[0]}" for r, v in r2["additive"].items())))
    r3s = ns.get("resB3s") or {}
    if r3s.get("verdict"):
        rows.append(("B3s", r3s["verdict"]["bound"], "controls: " + ", ".join(f"{r} {'ok' if row['control_ok'] else 'FAILED'}" for r, row in r3s["by_rendering"].items()) +
                     "; picture silent share k=8 " + fmt(r3s["verdict"]["picture_silent_k8"]) + "; carried content " + ", ".join(f"{r} {v}" for r, v in r3s["verdict"]["split"].items())))
    if r3.get("verdict"):
        rows.append(("B3", r3["verdict"]["bound"], "controls: " + ", ".join(f"{r} {'ok' if row['control_ok'] else 'FAILED'}" for r, row in r3["by_rendering"].items()) +
                     "; picture silent share k=8 " + fmt(r3["verdict"]["picture_silent_k8"]) + "; carried content " + ", ".join(f"{r} {v.get('verdict', '')}" for r, v in r3.get("split", {}).items())))
    if r4.get("verdict"):
        rows.append(("B4", r4["verdict"], f"seen {'ok' if r4['seen'] else 'NO'}, control {'ok' if r4['control_ok'] else 'FAILED'}, K_lens {fmt(r4['capacity']['K'])}"))
    if r5.get("models"):
        rows.append(("B5", r5.get("verdict", "one model so far"), "models: " + ", ".join(r5["models"])))
    if r6.get("fits"):
        m = r6["models"][-1]
        rows.append(("B6", "articulation ratio " + ", ".join(f"{r} {f['ratio']:.2f}" for r, f in r6["fits"][m].items()), f"on {m}: " + ", ".join(f"{r} said K {f['K_said']:.1f} / used K {f['K_used']:.1f}" for r, f in r6["fits"][m].items())))
    for key, label in (("resE1", "E1"), ("resE2", "E2"), ("resE2P", "E2P")):
        rr = ns.get(key) or {}
        if key == "resE1" and rr.get("by_rendering"):
            rows.append(("E1", "; ".join(f"{r} {v['verdict']}" for r, v in rr["by_rendering"].items()), "; ".join(f"{r}: other items B-A {fmt(v['P1_other_B_minus_A'])}, target B-A {fmt(v['P2_target_B_minus_A'])}, acc A {v['conditions']['A']['accuracy'][0]:.2f}" for r, v in rr["by_rendering"].items())))
        if key == "resE2" and rr.get("verdict"):
            rows.append(("E2", rr["verdict"], "; ".join(f"k={k}: I text->vision {fmt(v['I_text_on_vision'])}, I vision->text {fmt(v['I_vision_on_text'])}" for k, v in rr["by_k"].items())))
        if key == "resE2P" and rr.get("by_k"):
            rows.append(("E2P", "; ".join(f"k={k}: {v['verdict']}" for k, v in rr["by_k"].items()), "; ".join(f"k={k}: full {v['effects']['full, every layer']['dp_donor'][0]:.2f}, J {v['effects']['J part, read layers']['dp_donor'][0]:.2f}, random {v['effects']['random part, read layers']['dp_donor'][0]:.2f}, complement {v['effects']['complement, read layers']['dp_donor'][0]:.2f}" for k, v in rr["by_k"].items())))
    if r7.get("verdict"):
        rows.append(("B7", r7["verdict"], "silent share k=8: " + ", ".join(f"{r} {fmt(v)}" for r, v in r7["silent_k8"].items()) + "; " +
                     ", ".join(f"{r} seen {'ok' if row['seen'] else 'NO'}/control {'ok' if row['control_ok'] else 'FAILED'}" for r, row in r7["by_rendering"].items())))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--cells", default=None, help="comma list; default M0,B1,B2,B3,B4,B5,B6 (B7 for Gemma)")
    ap.add_argument("--tag", default=datetime.datetime.now().strftime("%Y%m%d_%H%M"))
    ap.add_argument("--smoke", action="store_true", help="tiny sizes, written to smoke_<tag>")
    ap.add_argument("--batch", type=int, default=0, help="override BATCH")
    ap.add_argument("--lens-revision", default=None)
    ap.add_argument("--other-summaries", default="", help="glob of eB5_summary_*.json from other models, for B5 and B6")
    ap.add_argument("--aggregate-only", action="store_true", help="load this model's saved eB1..eB4 results and run only B5/B6")
    ap.add_argument("--run-root", default="/workspace/nayan/runs")
    ap.add_argument("--cells-file", default=None, help="an extra cells file (e.g. jbound/cells_e.py) whose cells are added to the catalogue")
    args = ap.parse_args()
    if args.cells is None:
        args.cells = "B7" if "gemma" in args.model.lower() else "M0,B1,B2,B3,B4,B5,B6"
    tag = ("smoke_" if args.smoke and not args.tag.startswith("smoke") else "") + args.tag
    model_tag = args.model.split("/")[-1]
    run_dir = f"{args.run_root}/{tag}/{model_tag}"
    os.makedirs(run_dir, exist_ok=True)
    log = open(f"{run_dir}/run.log", "a")
    sys.stdout, sys.stderr = Tee(sys.__stdout__, log), Tee(sys.__stderr__, log)
    con = Console(file=sys.__stdout__, highlight=False)
    lcon = Console(file=log, no_color=True, force_terminal=False, width=160)

    def say(*a, **k):
        con.print(*a, **k)
        lcon.print(*a, **k)

    wanted = [c.strip() for c in args.cells.split(",") if c.strip()]
    say(Panel(f"[bold]{args.model}[/bold]  cells {', '.join(wanted)}  tag [bold]{tag}[/bold]{'  [yellow]SMOKE[/yellow]' if args.smoke else ''}\n"
              f"cells sha {sha(CELLS)}  setup sha {sha(SETUP)}  {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n{run_dir}", title="articulation bound", border_style="blue"))

    ns = {"__name__": "__jbound__", "MODEL_NAME": args.model, "RUN_TAG": tag, "RUN_ROOT": args.run_root, "LENS_REVISION": args.lens_revision,
          "BATCH_OVERRIDE": args.batch, "B_OTHER_PATHS": sorted(glob.glob(args.other_summaries)) if args.other_summaries else [], "SMOKE": args.smoke}
    t0 = time.time()
    with con.status("[bold blue]loading model, lens and the notebook setup (batch check, fonts)...", spinner="dots"):
        exec(compile(open(SETUP).read(), SETUP, "exec"), ns)
    say(f"[green]setup done in {time.time() - t0:.0f}s[/green]  batch check: max |dp| {ns['BATCH_CHECK']['max_dp']:.4f}, mean |dlog10 rank| {ns['BATCH_CHECK']['mean_dlog10rank']:.3f}, BATCH {ns['BATCH']}")
    import torch, transformers, jlens
    cfg = dict(model=args.model, model_revision=getattr(ns["hf_model"].config, "_commit_hash", None), model_class=type(ns["hf_model"]).__name__, lens_file=ns["LENS_FILE"],
               lens_path=ns["LENS_PATH"], jpp_file=ns.get("JPP_FILE"), jpp_loaded=ns.get("JPP") is not None, lens_revision=args.lens_revision, layers=ns["LAYERS"], final=ns["FINAL"], batch=ns["BATCH"], batch_tokens=ns["BATCH_TOKENS"],
               batch_check=ns["BATCH_CHECK"], gpu=torch.cuda.get_device_name(0), torch=torch.__version__, transformers=transformers.__version__,
               jlens=getattr(jlens, "__version__", "?"), python=sys.version.split()[0], cells_sha=sha(CELLS), setup_sha=sha(SETUP), smoke=args.smoke, tag=tag,
               smoke_substitutions={n: [o for o, _ in SMOKE_SUBS[n]] for n in SMOKE_SUBS if n in wanted} if args.smoke else {},
               date=datetime.datetime.now().isoformat(timespec="seconds"), argv=sys.argv[1:])
    json.dump(cfg, open(f"{run_dir}/config.json", "w"), indent=2)

    cells = split_cells(open(CELLS).read())
    if args.cells_file:
        cells.update(split_cells(open(args.cells_file).read()))
    unknown = [c for c in wanted if c not in cells]
    assert not unknown, f"unknown cells {unknown}; the file has {list(cells)}"
    if args.aggregate_only:
        for n in ("B1", "B2", "B3", "B3s", "B4"):
            path = f"{run_dir}/e{n}_results.json"
            ns[f"res{n}"] = intkeys(json.load(open(path))) if os.path.exists(path) else {}
            say(f"loaded {path}" if os.path.exists(path) else f"[yellow]no {path}: res{n} = {{}}[/yellow]")
        ns["B_OK_RUN"] = bool(ns["resB1"])
    ran, timing = set(), {}
    for name in wanted:
        for dep in DEPS.get(name, []):                                        # definitions of earlier cells this one uses, no forwards
            if dep in ("E0", "E2") and name in ("E1", "E2", "E2u", "E2P", "E2Pd", "T1", "L1", "A1", "F1", "F2", "G1", "F3", "G1b", "F4", "D1", "K1", "S1", "D2", "O1") and (dep == "E0" or name in ("E1", "E2")) and dep not in ran and dep not in wanted[:wanted.index(name)]:
                raise SystemExit(f"{name} needs {dep} to have run in this session (its stimuli live in memory): add {dep} to --cells")
            if dep not in ran and dep not in wanted[:wanted.index(name)]:
                keep_save = ns.get("save")
                ns["save"] = lambda *a, **k: None
                dsrc = definitions(cells[dep])
                if args.smoke:
                    for old, new in SMOKE_SUBS.get(dep, []):
                        dsrc = dsrc.replace(old, new)
                try:
                    exec(compile(dsrc, f"{CELLS}:{dep}:definitions", "exec"), ns)
                finally:
                    ns["save"] = keep_save
                if dep == "M0" and "MM_PROC" not in ns and args.aggregate_only:          # B7 loads its own processor when MM_PROC is undefined
                    ns.update(MM_PROC=None, MM_VISION_PATH="base", MM_N_IMAGE=ns["MM_IMG_TOKENS"] - 8)
                ran.add(dep)
                say(f"[dim]{dep}: definitions loaded for {name}[/dim]")
        src = cells[name]
        if args.smoke:
            for old, new in SMOKE_SUBS.get(name, []):
                assert src.count(old) == 1, f"smoke substitution not found once in {name}: {old!r}"
                src = src.replace(old, new)
        if args.aggregate_only and name in ("M0", "B1", "B2", "B3", "B3s", "B4"):
            src = definitions(src)
        say(Panel(f"[bold]{name}[/bold]  {TITLES.get(name, '')}", border_style="cyan", subtitle=f"{datetime.datetime.now():%H:%M:%S}", subtitle_align="right"))
        t0 = time.time()
        try:
            exec(compile(src, f"{CELLS}:{name}", "exec"), ns)
        except Exception:
            traceback.print_exc()
            say(f"[bold red]{name} FAILED after {time.time() - t0:.0f}s; stopping. Log: {run_dir}/run.log[/bold red]")
            sys.exit(1)
        finally:
            try:
                ns["plt"].close("all")
            except Exception:
                pass
            torch.cuda.empty_cache()
        ran.add(name)
        timing[name] = round(time.time() - t0)
        say(f"[green]{name} done in {timing[name]}s[/green]  GPU peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
        json.dump(dict(cells=timing, verdicts=verdicts(ns)), open(f"{run_dir}/verdicts.json", "w"), indent=2)
    rows = verdicts(ns)
    if rows:
        tb = Table(title=f"verdicts · {model_tag} · {tag}", show_lines=True, expand=True)
        tb.add_column("cell", style="bold", width=5)
        tb.add_column("verdict", ratio=2)
        tb.add_column("numbers", ratio=3)
        for r in rows:
            tb.add_row(*r)
        say(tb)
    say(Panel(f"all done in {sum(timing.values())}s · " + ", ".join(f"{k} {v}s" for k, v in timing.items()) + f"\nresults {run_dir}\nfigures {run_dir}/figures", border_style="green"))


if __name__ == "__main__":
    main()
