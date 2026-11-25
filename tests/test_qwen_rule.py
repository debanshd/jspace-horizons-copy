"""check_lens's Qwen rule on the CPU (no model, no network): the bar is the worst calibration rank at the nearest depth, so each
calibration lens meets it; the rule is informative only if the bar rejects every calibration model's logit lens; and lenses seen
before the rule was written are listed so it refuses them."""

import importlib.util
from pathlib import Path

from jspace import config

ROOT = Path(__file__).resolve().parents[1]


def load():
    spec = importlib.util.spec_from_file_location("check_lens", ROOT / "scripts/check_lens.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def summary(n, jl, lg):
    return dict(n_layers=n, rank_jlens={str(L): v for L, v in jl.items()}, rank_logit={str(L): v for L, v in lg.items()})


CAL = {
    "a": summary(24, {12: 2.0, 16: 1.5, 20: 0.5}, {12: 4.0, 16: 3.0, 20: 0.4}),
    "b": summary(32, {16: 2.5, 24: 1.0, 28: 0.3}, {16: 4.5, 24: 2.0, 28: 0.2}),
}


def test_every_calibration_lens_meets_the_bar_at_its_own_layers():
    cl = load()
    for s in CAL.values():
        layers = sorted(int(L) for L in s["rank_jlens"])
        bar = cl.calibration_bar(list(CAL.values()), [L / s["n_layers"] for L in layers])
        assert all(s["rank_jlens"][str(L)] <= b for L, b in zip(layers, bar))


def test_the_bar_is_the_worst_rank_at_the_nearest_depth():
    cl = load()
    assert cl.calibration_bar(list(CAL.values()), [0.5, 0.85]) == [2.5, 0.5]


def test_the_rule_is_void_when_a_logit_lens_meets_the_bar():
    cl = load()
    assert cl.informative(CAL) == {"a": True, "b": True}  # both logit lenses are above the bar at mid depth
    weak = dict(CAL, c=summary(20, {10: 9.0, 15: 9.0, 18: 9.0}, {10: 4.0, 15: 3.0, 18: 1.0}))  # a lens so poor the bar rejects nothing
    assert not all(cl.informative(weak).values())


def test_the_calibration_set_and_the_lenses_already_seen():
    assert config.QWEN_CALIBRATION == ("Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-27B")
    assert all(not config.MODELS[m].lens_repo for m in config.QWEN_CALIBRATION)  # Neuronpedia's own pins
    seen = {config.lens_for("Qwen/Qwen3.5-9B", "jlens")[3]} | {config.MODELS[m].lens_sha256 for m in config.FIT_MODELS if m.startswith("Qwen/")}
    assert seen == set(config.SEEN_LENSES)
