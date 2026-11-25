"""check_lens's Qwen rule on the CPU (no model, no network): the bar is the weakest calibration lens at the nearest depth, so each
calibration lens passes it, and a lens below it at any read layer from half depth fails."""

import importlib.util
from pathlib import Path

from jspace import config

ROOT = Path(__file__).resolve().parents[1]


def load():
    spec = importlib.util.spec_from_file_location("check_lens", ROOT / "scripts/check_lens.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def summary(n, jl):
    return dict(n_layers=n, jlens={str(L): v for L, v in jl.items()})


def test_every_calibration_lens_meets_the_bar_at_its_own_layers():
    cl = load()
    cal = [summary(24, {12: 0.10, 16: 0.20, 20: 0.30}), summary(32, {16: 0.05, 24: 0.40, 28: 0.25}), summary(64, {32: 0.15, 48: 0.10, 60: 0.20})]
    for s in cal:
        layers = sorted(int(L) for L in s["jlens"])
        bar = cl.calibration_bar(cal, [L / s["n_layers"] for L in layers])
        assert all(s["jlens"][str(L)] >= b for L, b in zip(layers, bar))


def test_the_bar_is_the_lowest_at_the_nearest_depth():
    cl = load()
    cal = [summary(10, {5: 0.5, 9: 0.6}), summary(20, {10: 0.3, 19: 0.8})]
    assert cl.calibration_bar(cal, [0.5, 0.9, 0.65]) == [0.3, 0.6, 0.3]  # 0.65 is nearest 0.5 in both


def test_the_calibration_set_is_the_four_neuronpedia_qwen_lenses():
    assert config.QWEN_CALIBRATION == ("Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-27B")
    assert all(not config.MODELS[m].lens_repo for m in config.QWEN_CALIBRATION)  # Neuronpedia's own pins
