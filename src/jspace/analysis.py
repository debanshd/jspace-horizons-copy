"""Scoring answers and the scene bootstrap every summary uses."""

import re

import numpy as np


def noun_matches(outs, targets, words):
    """Whether each answer's first vocabulary word is its target (a trailing 's' is dropped when that names a vocabulary word)."""
    ok = []
    for o, t in zip(outs, targets):
        toks = re.findall(r"[a-z]+", o.lower())
        first = next((w for w in toks if w in words or w.rstrip("s") in words), None)
        first = first.rstrip("s") if first and first not in words else first
        ok.append(bool(first == t))
    return np.array(ok)


def boot_scene(values, scenes, n=2000, seed=0):
    """Mean with a 95% bootstrap interval over scenes: rows that share a scene are resampled together."""
    values, scenes = np.asarray(values, float), np.asarray(scenes)
    u = np.unique(scenes)
    by = {g: values[scenes == g] for g in u}
    rng = np.random.default_rng(seed)
    bs = [np.concatenate([by[g] for g in rng.choice(u, len(u))]).mean() for _ in range(n)]
    return float(values.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def fmt(t):
    """(mean, lo, hi) as 'mean [lo, hi]'."""
    return f"{t[0]:.2f} [{t[1]:.2f}, {t[2]:.2f}]"


def rate(verdicts):
    """Share of True among the verdicts that are not None (nan when there are none)."""
    vals = [float(v) for v in verdicts if v is not None]
    return float(np.mean(vals)) if vals else float("nan")
