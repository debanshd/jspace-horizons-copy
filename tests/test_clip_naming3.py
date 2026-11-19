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


def test_one_name_is_scored_and_a_hedge_is_not():
    cn = load()
    assert cn.score("race car", "car")[0] and cn.score("Sailboat.", "boat")[0]  # clip_naming2's scoring on a single name
    assert cn.score("Horse, horse", "horse")[0] and not cn.hedged("Horse, horse")  # the same name repeated is one name
    assert cn.hedged("cat or dog") and cn.score("cat or dog", "cat") == (False, False)  # a hedge earns nothing
    assert cn.hedged("Horse, cow, horse, hor") and cn.names("Horse, cow, horse, hor") == ["horse", "cow"]  # "hor" is cut off
    assert cn.hedged("laptop;computer;screen") and cn.hedged("Horse\nCow")
    assert cn.names("car, carrot") == ["car", "carrot"]  # only the last name can be a cut-off fragment
    assert not cn.hedged("Orange") and not cn.hedged("Landrover") and not cn.hedged("motorboat")  # no split inside a word
    assert not cn.score("animal", "horse")[0] and not cn.score("dog", "horse")[0]
    assert cn.names("") == [] and cn.score("", "cat") == (False, False)


def test_the_follow_up_continues_the_same_conversation():
    cn = load()
    seen = []
    lm = types.SimpleNamespace(chat_messages=lambda msgs: seen.append(msgs) or "")
    cn.prompt(lm, "moving")
    cn.prompt(lm, "still", "cat or dog")
    assert len(seen[0]) == 1 and [m["role"] for m in seen[1]] == ["user", "assistant", "user"]
    assert seen[1][0]["content"][0] == {"type": "image"} and seen[1][1]["content"][0]["text"] == "cat or dog"
    assert seen[1][2]["content"][0]["text"] == cn.FOLLOWUP and "one category" in cn.FOLLOWUP


def answers(wrong):
    """four clips, all named on every input except the (clip, input) pairs in `wrong`"""
    rows = [dict(clip=c, label="cat", input=i, correct=(c, i) not in wrong, coco=True, hedged=False) for c in "abcd" for i in matched.INPUTS]
    return pd.DataFrame(rows)


def test_reliable_and_gated_per_input():
    cn = load()
    schedule = [(0, "a", "b"), (1, "c", "d"), (2, "a", "c")]
    out = cn.summarize(answers({("c", "frozen"), ("d", "moving")}), schedule)
    assert out["reliable_by_input"]["moving"] == ["a", "b", "c"] and out["gated_by_input"]["moving"] == [0, 2]
    assert out["gated_by_input"]["frozen"] == [0] and out["gated_by_input"]["still"] == [0, 1, 2]
    assert out["gated_input"] == "moving" and out["gated_scenes"] == [0, 2] and out["reliable"] == ["a", "b", "c"]
    assert out["reliable_all"] == ["a", "b"] and out["gated_all"] == [0]  # clip_naming's all-three-inputs rule, for comparison
    assert set(out["share"]) == set(matched.INPUTS) and "unreliable" not in out and out["hedged"]["moving"] == 0.0


def test_question_is_clip_naming2s():
    cn = load()
    assert cn.QUESTION == cn.n2.QUESTION and cn.N_NEW == 24
