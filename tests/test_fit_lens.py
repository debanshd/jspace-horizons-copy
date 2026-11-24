"""scripts/fit_lens.py's corpus rule, prompt selection and shards, and the pins it reads, on the CPU (no model, no network)."""

import importlib.util
from pathlib import Path

import pytest

from jspace import config

ROOT = Path(__file__).resolve().parents[1]


def load():
    spec = importlib.util.spec_from_file_location("fit_lens", ROOT / "scripts/fit_lens.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_the_corpus_rule():
    fl = load()
    long = lambda c: c * 700  # noqa: E731
    texts = [" = Heading = " + "x" * 700, long("a"), "short", long("b"), long("a"), "  " + long("c") + "  \n", None]
    p = fl.paragraphs(texts)
    assert sorted(p) == sorted([long("a"), long("b"), long("c")])  # no heading, no short one, no repeat, stripped
    assert p == fl.paragraphs(texts)  # the seed-0 order is fixed
    assert fl.paragraphs(texts, seed=1) != p or len(p) < 3


def test_selection_and_shards_cover_the_same_prompts():
    fl = load()
    prompts = [f"p{i}" for i in range(20)]
    first = fl.select(prompts, 10)
    assert first == prompts[:10]
    shards = [fl.select(prompts, 10, (i, 3)) for i in range(3)]
    assert sorted(x for s in shards for x in s) == sorted(first) and not set(shards[0]) & set(shards[1])
    assert fl.parse_shard("1/4") == (1, 4) and fl.parse_shard(None) is None
    with pytest.raises(SystemExit):
        fl.select(prompts, 30)
    with pytest.raises(AssertionError):
        fl.select(prompts, 10, (3, 3))


def test_pins():
    assert set(config.FIT_MODELS) == {"Qwen/Qwen2.5-Omni-3B", "Qwen/Qwen2.5-Omni-7B", "Qwen/Qwen3-Omni-30B-A3B-Instruct"}
    for s in config.FIT_MODELS.values():
        assert len(s.revision) == 40 and s.loader.endswith("ThinkerForConditionalGeneration") and s.n_layers > 0 and s.d_model > 0
    assert len(config.WIKITEXT_REVISION) == 40 and all(len(h) == 64 for _, h in config.WIKITEXT_FILES.values())


def test_a_lens_extends_by_fitting_the_rest():
    fl = load()
    prompts = [f"p{i}" for i in range(30)]
    assert fl.select(prompts, 20, start=8) == prompts[8:20]
    assert fl.select(prompts, 8) + fl.select(prompts, 20, start=8) == prompts[:20]  # the first 8 of 20 are the first 8
    assert fl.select(prompts, 20, (1, 2), start=8) == prompts[9:20:2]
    m = lambda n, start=0, shard=None: dict(n_requested=n, start=start, shard=shard)  # noqa: E731
    assert fl.check_cover([dict(n_requested=8, shard=None), m(20, 8)]) == 20  # an old meta has no start
    assert fl.check_cover([m(8), m(20, 8, [0, 2]), m(20, 8, [1, 2])]) == 20
    assert fl.check_cover([m(20, 0, [0, 2]), m(20, 0, [1, 2])]) == 20
    for bad in ([m(8), m(20, 6)], [m(8), m(20, 10)], [m(8), m(20, 8, [0, 2])]):  # overlap, gap, half a range
        with pytest.raises(SystemExit):
            fl.check_cover(bad)
