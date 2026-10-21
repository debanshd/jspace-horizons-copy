"""Reproduce a figure cell on saved results (no forwards) and, if saving fails, list the artists with absurd extents.
    DIAG_TAG=smoke2 DIAG_CELL=B6 venv/bin/python jbound/diag_fig.py
"""
import glob, json, os, sys, traceback
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run as R

tag, cell, model = os.environ.get("DIAG_TAG", "smoke2"), os.environ.get("DIAG_CELL", "B6"), os.environ.get("DIAG_MODEL", "Qwen/Qwen3.5-4B")
ns = {"__name__": "__diag__", "MODEL_NAME": model, "RUN_TAG": tag, "RUN_ROOT": "/workspace/nayan/runs", "LENS_REVISION": None, "BATCH_OVERRIDE": 0, "B_OTHER_PATHS": [], "SMOKE": True}
exec(compile(open(R.SETUP).read(), R.SETUP, "exec"), ns)
cells = R.split_cells(open(R.CELLS).read())
run_dir = ns["RUN_DIR"]
for n in ("B1", "B2", "B3", "B4"):
    p = f"{run_dir}/e{n}_results.json"
    ns[f"res{n}"] = R.intkeys(json.load(open(p))) if os.path.exists(p) else {}
ns["B_OK_RUN"] = True
for dep in R.DEPS.get(cell, []):
    ns["save"] = lambda *a, **k: None
    dsrc = R.definitions(cells[dep])
    for old, new in R.SMOKE_SUBS.get(dep, []):
        dsrc = dsrc.replace(old, new)
    exec(compile(dsrc, R.CELLS + ":" + dep, "exec"), ns)
    if dep == "M0" and "MM_PROC" not in ns:
        ns.update(MM_PROC=None, MM_VISION_PATH="base", MM_N_IMAGE=ns["MM_IMG_TOKENS"] - 8)
src = cells[cell]
for old, new in R.SMOKE_SUBS.get(cell, []):
    src = src.replace(old, new)
# keep the figure: make ws_finish store it before saving
ns["_FIGS"] = []
src = src.replace("    ws_finish(fig, ", "    _FIGS.append(fig); ws_finish(fig, ")
try:
    exec(compile(src, R.CELLS + ":" + cell, "exec"), ns)
    print("figure saved fine")
except Exception:
    traceback.print_exc(limit=3)
    fig = ns["_FIGS"][-1]
    fig.canvas.draw_idle()
    r = fig.canvas.get_renderer() if hasattr(fig.canvas, "get_renderer") else None
    import matplotlib
    from matplotlib.text import Text
    print("figure size in", fig.get_size_inches(), "dpi", fig.dpi)
    for a in fig.findobj():
        try:
            bb = a.get_window_extent(r) if r is not None else a.get_window_extent()
        except Exception:
            continue
        if bb is not None and (abs(bb.y0) > 1e5 or abs(bb.y1) > 1e5 or abs(bb.x0) > 1e5 or abs(bb.x1) > 1e5):
            print("HUGE:", type(a).__name__, getattr(a, "get_text", lambda: "")(), bb, "pos", getattr(a, "get_position", lambda: None)())
