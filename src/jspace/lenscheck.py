"""What validates a lens fitted here (config.LENS_CHECKS): the maps of two lenses compared layer by layer, and the concept readout
on the J-lens paper's lens-quality prompts (src/jspace/lens_eval). The scripts that load models and lenses use these:
scripts/compare_lenses.py, scripts/lens_readout.py, scripts/validate_lens.py."""

import hashlib
import json

import numpy as np

from . import config

EVAL_DIR = config.ROOT / "src" / "jspace" / "lens_eval"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def checked_layers(layers, n_layers, min_depth=config.LENS_CHECKS["min_depth"]):
    """the layers a check covers: those the experiments read, from int(min_depth * n_layers) on (Session.layers)"""
    return [L for L in layers if L >= int(min_depth * n_layers)]


def cosines(ja, jb):
    """per shared layer: the cosine of the two maps (flattened), and of their departures from the identity (the logit lens's map),
    which the identity part does not inflate"""
    import torch

    out = {}
    for L in sorted(set(ja) & set(jb)):
        a, b = ja[L].float().flatten(), jb[L].float().flatten()
        eye = torch.eye(ja[L].shape[0]).flatten()
        da, db = a - eye, b - eye
        out[L] = (float(a @ b / (a.norm() * b.norm())), float(da @ db / (da.norm() * db.norm())))
    return out


def agree(cos, n_layers, min_cosine=config.LENS_CHECKS["min_cosine"]):
    """checks 1 and 2: the maps' cosine is at least min_cosine at every checked layer. (passes, the checked layers, the lowest)"""
    layers = checked_layers(sorted(cos), n_layers)
    if not layers:
        raise SystemExit("the two lenses share no checked layer")
    low = min(cos[L][0] for L in layers)
    return bool(low >= min_cosine), layers, low


def eval_sets():
    """the five lens-quality sets, each checked against its sha256: {name: items}"""
    out = {}
    for name, expected in config.LENS_EVAL_FILES.items():
        f = EVAL_DIR / f"lens-eval-{name}.json"
        got = sha256(f)
        if got != expected:
            raise SystemExit(f"{f}: sha256 {got}, pinned {expected}")
        out[name] = json.loads(f.read_text())["items"]
    return out


def readout_offset(ids, set_name, newline_ids):
    """the readout position, counted from the end of the prompt's tokens: the last token for every set but poetry, whose readout
    is the last newline token (the end of the couplet's first line), as the sets' README defines"""
    if set_name != "poetry":
        return -1
    at = [i for i, t in enumerate(ids) if t in newline_ids]
    if not at:
        raise SystemExit("a poetry prompt without a newline token")
    return at[-1] - len(ids)


def word_ids(encode, word):
    """a word's candidate tokens: the first token of "word" and of " word" (its rank is the better of the two)"""
    return sorted({encode(word)[0], encode(" " + word)[0]})


def pass_at_k(item_ranks, k):
    """the paper's pass@k for one set: over items, the mean share of intermediates whose best rank over layers is at most k"""
    return float(np.mean([np.mean([r <= k for r in ranks]) for ranks in item_ranks]))


def margin(summary, k=config.LENS_CHECKS["k"]):
    """a lens_readout summary's margin: the J-lens's pass@k averaged over the sets minus the logit lens's (the identity map's)"""
    return summary[f"jlens_pass@{k}"]["mean"] - summary[f"logit_pass@{k}"]["mean"]


def readout_bar(calibration_margins):
    """check 3's bar, the lowest margin among the calibration lenses, and whether it is informative (above 0, so the identity
    map, whose margin is 0, fails it)"""
    bar = min(calibration_margins.values())
    return bar, bool(bar > 0)
