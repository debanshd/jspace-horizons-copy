"""The lens checks of config.LENS_CHECKS on the CPU (no model, no network): the map comparison, the readout's positions, tokens and
scoring, the prompt sets' pins, and the halves the 500-prompt extension fits."""

import importlib.util
from pathlib import Path

import pytest
import torch

from jspace import config, lenscheck

ROOT = Path(__file__).resolve().parents[1]


def test_identical_maps_agree_and_a_different_one_does_not():
    g = torch.Generator().manual_seed(0)
    d, n = 16, 10
    ja = {L: torch.eye(d) + 0.3 * torch.randn(d, d, generator=g) for L in range(n - 1)}
    same = lenscheck.cosines(ja, {L: j.clone() for L, j in ja.items()})
    assert all(abs(c - 1) < 1e-6 and abs(c2 - 1) < 1e-6 for c, c2 in same.values())
    assert lenscheck.agree(same, n)[0]
    other = {L: torch.eye(d) + 0.3 * torch.randn(d, d, generator=g) for L in range(n - 1)}
    cos = lenscheck.cosines(ja, other)
    passes, layers, low = lenscheck.agree(cos, n)
    assert not passes and layers == [L for L in range(n - 1) if L >= int(0.3 * n)]
    assert all(c2 < c for c, c2 in cos.values())  # the identity inflates the plain cosine; the departures show the difference


def test_the_checked_layers_are_the_read_layers():
    assert lenscheck.checked_layers(range(63), 64) == list(range(19, 63))
    assert lenscheck.checked_layers(range(23), 24) == list(range(7, 23))


def test_readout_positions_tokens_and_pass_at_k():
    assert lenscheck.readout_offset([5, 6, 7], "multihop", {9}) == -1
    assert lenscheck.readout_offset([5, 9, 6, 9, 7, 8], "poetry", {9}) == -3  # the last newline token
    with pytest.raises(SystemExit):
        lenscheck.readout_offset([5, 6], "poetry", {9})
    enc = {"Brazil": [11, 12], " Brazil": [13]}.__getitem__
    assert lenscheck.word_ids(enc, "Brazil") == [11, 13]
    assert lenscheck.pass_at_k([[1, 50], [3]], 10) == pytest.approx((0.5 + 1.0) / 2)
    assert lenscheck.readout_bar({"a": 0.04, "b": 0.03, "c": 0.05}) == (0.03, True)
    assert lenscheck.readout_bar({"a": 0.04, "b": -0.05}) == (-0.05, False)  # a calibration lens below its logit lens: void
    s = {"jlens_pass@10": {"mean": 0.30}, "logit_pass@10": {"mean": 0.25}}
    assert lenscheck.margin(s) == pytest.approx(0.05)
    identity = {"jlens_pass@10": {"mean": 0.25}, "logit_pass@10": {"mean": 0.25}}
    bar, ok = lenscheck.readout_bar({"a": 0.03})
    assert ok and not lenscheck.margin(identity) >= bar  # the identity map's margin is 0: it fails an informative bar


def test_the_prompt_sets_match_their_pins():
    sets = lenscheck.eval_sets()
    assert sorted(sets) == ["association", "multihop", "multilingual", "poetry", "typo"]
    assert all(all(it["prompt"] and it["intermediates"] for it in items) for items in sets.values())
    assert all("\n" in it["prompt"] for it in sets["poetry"])


def test_the_extension_halves_cover_the_lens():
    spec = importlib.util.spec_from_file_location("fit_lens", ROOT / "scripts/fit_lens.py")
    fl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fl)
    lo, hi = config.LENS_CHECKS["halves"]
    assert (lo, hi) == (250, 500) and config.LENS_CHECKS["readout_decides"] is False  # the change of 25 Nov
    base = dict(n_requested=lo, start=0, shard=None)  # the 250-prompt fit: the first half
    ext = dict(n_requested=hi, start=lo, shard=None)  # prompts 250 to 499: the second half
    assert fl.check_cover([base, ext]) == hi  # together, the 500-prompt lens
    assert fl.covered(hi, lo) == list(range(lo, hi))


def test_the_reproduction_and_calibration_models_are_pinned():
    c = config.LENS_CHECKS
    assert c["reproduce"] in config.FIT_MODELS and config.FIT_MODELS[c["reproduce"]].revision == config.MODELS[c["reproduce"]].revision
    assert all(m in config.MODELS and not config.MODELS[m].lens_repo for m in config.READOUT_CALIBRATION)


def test_each_family_has_its_check_1_and_calibration():
    for fam, f in config.LENS_FAMILIES.items():
        assert f["reproduce"] in config.FIT_MODELS and f["reproduce_n"] > 0
        pinned = config.MODELS[config.REPRO_PINNED.get(f["reproduce"], f["reproduce"])]
        assert not pinned.lens_repo  # a Neuronpedia lens
        assert all(m in config.MODELS and not config.MODELS[m].lens_repo and config.lens_family(m) == fam for m in f["calibration"])
    # Gemma's check 1 refits the base E2B, the model Neuronpedia's E2B lens was fitted on, which MODELS pins for the instruct E2B
    assert config.MODELS["google/gemma-4-E2B-it"].lens_file.startswith("gemma-4-e2b/jlens/")
    assert config.lens_family("google/gemma-4-12B-it") == "google" and config.lens_family("Qwen/Qwen3.5-9B") == "Qwen"
    with pytest.raises(SystemExit):
        config.lens_family("meta-llama/Llama-3.1-8B")
