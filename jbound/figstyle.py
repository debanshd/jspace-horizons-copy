"""The suite's figure style for the standalone figure scripts: the same font, Okabe-Ito palette and rc as setup_runpod.py."""
import glob, logging
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager as fm
import seaborn as sns

OKABE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]   # J-lens 0, keep-only 1, J++ 2, PCA 3, random 4, text/other 5


def apply(titlesize=9.5, labelsize=9):
    for f in glob.glob("/usr/share/**/*[Hh]eros*.[ot]tf", recursive=True):
        fm.fontManager.addfont(f)
    font = next((f for f in ["TeX Gyre Heros", "Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"] if f in {x.name for x in fm.fontManager.ttflist}), "DejaVu Sans")
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    sns.set_theme(context="paper", style="ticks", palette=OKABE, rc={
        "font.family": [font], "font.size": 9, "axes.titlesize": titlesize, "axes.labelsize": labelsize, "legend.fontsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": 0.8, "xtick.major.width": 0.8,
        "ytick.major.width": 0.8, "legend.frameon": False, "axes.titlelocation": "left",
        "axes.titleweight": "normal", "figure.dpi": 130, "savefig.dpi": 300, "savefig.bbox": "tight"})
    return font


def spread(items, gap, lo, hi):
    """items = [(y, payload)]: nudge the y's apart by at least gap, keep their order, keep the stack inside [lo, hi]"""
    items = sorted(items, key=lambda t: t[0])
    ys = [t[0] for t in items]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + gap)
    if ys and ys[-1] > hi:
        ys = [y - (ys[-1] - hi) for y in ys]
    if ys and ys[0] < lo:
        ys = [y + (lo - ys[0]) for y in ys]
    return [(y, t[1]) for y, t in zip(ys, items)]


def finish(fig, out):
    sns.despine(fig=fig)
    fig.savefig(out)
    fig.savefig(out.replace(".png", ".pdf"))
    print("wrote", out)
