"""clip_naming3's first-name scoring and its per-input rule on the CPU."""

import importlib
import sys
import types
from pathlib import Path

import pandas as pd

from jspace import matched

ROOT = Path(__file__).resolve().parents[1]


def load():
    """experiments/clip_naming3.py, with a stand-in for jspace.model where its dependencies (jlens) are missing"""
    sys.path.insert(0, str(ROOT / "experiments"))
    stub = None
    try:
        importlib.import_module("jspace.model")
    except ImportError:
        stub = types.ModuleType("jspace.model")
        stub.Session = None
        sys.modules["jspace.model"] = stub
    try:
        return importlib.import_module("clip_naming3")
    finally:
        sys.path.remove(str(ROOT / "experiments"))
        if stub is not None:
            del sys.modules["jspace.model"]


def test_the_first_name_of_a_list_is_scored():
    cn = load()
    assert cn.first_name("Horse, cow, horse, hor") == "Horse" and cn.score("Horse, cow, horse, hor", "horse")[0]
    assert cn.score("laptop;computer;screen", "laptop")[0] and cn.score("cat or dog", "cat")[0]
    assert not cn.score("cat or dog", "dog")[0]  # only the first name counts
    assert cn.score("race car", "car")[0] and cn.score("Sailboat.", "boat")[0]  # clip_naming2's scoring on that name
    assert not cn.score("animal, horse", "horse")[0] and not cn.score("dog", "horse")[0]
    assert cn.first_name("Orange") == "Orange" and cn.first_name("Landrover") == "Landrover"  # no cut inside a word
    assert cn.first_name("") == ""


def answers(wrong):
    """four clips, all named on every input except the (clip, input) pairs in `wrong`"""
    rows = [dict(clip=c, label="cat", input=i, correct=(c, i) not in wrong, coco=True) for c in "abcd" for i in matched.INPUTS]
    return pd.DataFrame(rows)


def test_reliable_and_gated_per_input():
    cn = load()
    schedule = [(0, "a", "b"), (1, "c", "d"), (2, "a", "c")]
    out = cn.summarize(answers({("c", "frozen"), ("d", "moving")}), schedule)
    assert out["reliable_by_input"]["moving"] == ["a", "b", "c"] and out["gated_by_input"]["moving"] == [0, 2]
    assert out["gated_by_input"]["frozen"] == [0] and out["gated_by_input"]["still"] == [0, 1, 2]
    assert out["gated_input"] == "moving" and out["gated_scenes"] == [0, 2] and out["reliable"] == ["a", "b", "c"]
    assert out["reliable_all"] == ["a", "b"] and out["gated_all"] == [0]  # clip_naming's all-three-inputs rule, for comparison
    assert set(out["share"]) == set(matched.INPUTS) and "unreliable" not in out


def test_question_is_clip_naming2s():
    cn = load()
    assert cn.QUESTION == cn.n2.QUESTION and cn.N_NEW == 24
