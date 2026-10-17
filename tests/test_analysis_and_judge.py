"""Answer scoring, the scene bootstrap, and the judge's cache (no network)."""

import hashlib
import json

import numpy as np
import pytest

from jspace.analysis import boot_scene, noun_matches, rate
from jspace.judge import JUDGE_MODEL, RUBRIC_NAME, Judge

WORDS = {"dog", "cat", "train"}


def test_noun_matches_reads_the_first_vocabulary_word():
    assert noun_matches(["A dog.", "dogs", "the cat and the dog", "Train", "a horse"], ["dog", "dog", "cat", "train", "horse"], WORDS).tolist() == [True, True, True, True, False]
    assert noun_matches(["cat"], ["dog"], WORDS).tolist() == [False]


def test_boot_scene_resamples_scenes_together():
    m, lo, hi = boot_scene([1, 1, 0, 0], [0, 0, 1, 1])
    assert m == 0.5 and 0 <= lo <= m <= hi <= 1
    assert boot_scene([1, 1, 1], [0, 1, 2]) == (1.0, 1.0, 1.0)


def test_rate_ignores_missing_verdicts():
    assert rate([True, None, False]) == 0.5
    assert np.isnan(rate([None]))


def test_parse():
    assert Judge.parse('{"verdict": "YES"}') is True
    assert Judge.parse('```json\n{"verdict":"no"}\n```') is False
    assert Judge.parse("YES") is True
    assert Judge.parse("yes or no") is None
    assert Judge.parse("") is None


def key(answer, target):
    return hashlib.sha1((JUDGE_MODEL + RUBRIC_NAME.format(target=target, answer=answer)).encode()).hexdigest()


def test_recorded_verdicts_need_no_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    ref = tmp_path / "reference.json"
    ref.write_text(json.dumps({key("a dog", "dog"): True}))
    run = tmp_path / "run"
    run.mkdir()
    assert Judge(run, [ref]).name_match(["a dog"], ["dog"]) == [True]
    assert json.load(open(run / "judge_cache.json")) == {key("a dog", "dog"): True}  # the run keeps the verdicts it used


def test_a_new_answer_without_a_key_stops_the_run(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="no recorded verdict"):
        Judge(tmp_path).name_match(["a cat"], ["dog"])
