"""video_sweep2's pieces on the CPU: the whole-frame fitting, the gated scenes, the prompts and the scoring."""

import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from jspace import stimuli

ROOT = Path(__file__).resolve().parents[1]


def load():
    """experiments/video_sweep2.py, with a stand-in for jspace.model where its dependencies (jlens) are missing"""
    sys.path.insert(0, str(ROOT / "experiments"))
    stub = None
    try:
        importlib.import_module("jspace.model")
    except ImportError:
        stub = types.ModuleType("jspace.model")
        stub.Session = None
        sys.modules["jspace.model"] = stub
    try:
        return importlib.import_module("video_sweep2")
    finally:
        sys.path.remove(str(ROOT / "experiments"))
        if stub is not None:
            del sys.modules["jspace.model"]


def test_frames_of_any_size_fit_one_frame_whole():
    vs2 = load()
    wide = Image.new("RGB", (1920, 1080), (255, 0, 0))
    tall = Image.new("RGB", (600, 1080), (0, 0, 255))
    a, b = vs2.fit([wide]), vs2.fit([tall])
    assert a.shape == b.shape == (1, 378, 672, 3)  # the same size, so donor and receiver give the same number of tokens
    assert (a[0] == (255, 0, 0)).all()  # 16:9 fills the frame
    w = round(600 * 378 / 1080)  # the tall frame keeps its aspect, centred on the canvas grey
    left = (672 - w) // 2
    assert (b[0, :, left + 2] == (0, 0, 255)).all() and (b[0, :, : left - 1] == stimuli.BG).all() and (b[0, :, left + w + 1 :] == stimuli.BG).all()


def test_gated_scenes():
    vs2 = load()
    gate = {"gated_scenes": [3, 11, 14, 15, 25]}
    assert vs2.gated_scenes(gate) == [3, 11, 14, 15, 25]
    assert vs2.gated_scenes(gate, 2) == [3, 11]
    with pytest.raises(SystemExit):
        vs2.gated_scenes({"gated_scenes": []})


def test_prompts_ask_about_the_main_object_not_a_slot():
    vs2 = load()
    assert list(vs2.QUESTIONS2) == ["say", "animal", "letter", "set", "ride"]
    assert all("slot" not in q for q in vs2.QUESTIONS2.values())
    seen = []
    lm = types.SimpleNamespace(chat_messages=lambda msgs: seen.append(msgs) or "")
    r = dict(set=["cat", "bus", "clock"])
    vs2.prompt(lm, "moving", "set", r)
    vs2.prompt(lm, "still@donor", "neutral", r)
    c = seen[0][0]["content"]
    assert [p["type"] for p in c] == ["text", "video", "text"] and c[0]["text"] == vs2.INTRO2
    assert c[2]["text"] == "Rule: if the main object shown is a cat, a bus or a clock, answer 7; otherwise answer 3. Reply with the number only."
    assert [p["type"] for p in seen[1][0]["content"]] == ["text", "image"]  # the neutral prompt: INTRO2 and the media only


def test_say_is_scored_as_clip_naming3_and_the_rules_as_video_sweep():
    vs2 = load()
    assert vs2.judge("say", "race car", "car", []) and not vs2.judge("say", "subway car", "train", [])
    assert vs2.judge("say", "Sailboat", "boat", []) and not vs2.judge("say", "animal", "horse", [])
    assert vs2.judge("animal", "7", "horse", []) and not vs2.judge("animal", "3", "horse", [])
    assert vs2.judge("letter", "C", "car", []) and vs2.judge("set", "7", "cat", ["cat", "bus", "clock"])
    assert vs2.judge("ride", "7", "bicycle", []) and not vs2.judge("ride", "7", "cat", [])


def test_to_frame_scores_each_row():
    vs2 = load()
    vs2.SETS[0] = ["cat", "bus", "clock"]
    rows = [dict(input="moving", task="say", scene=0, X="cat", Y="car", Z="dog", layers=(), donor=None, ids_x=[1], ids_y=[2], ko=())]
    R = vs2.to_frame(rows, [("race car", -1.0, -0.5)])
    assert not R.loc[0, "correct"] and R.loc[0, "follows_Y"] and not R.loc[0, "follows_Z"] and R.loc[0, "ld"] == pytest.approx(-0.5)
    assert np.isclose(R.loc[0, "lp_X"], -1.0)


def test_a_hedge_scores_for_no_one():
    vs2 = load()
    assert not vs2.judge("say", "cat or dog", "cat", []) and not vs2.judge("say", "cat or dog", "dog", [])
    assert vs2.judge("say", "Horse, horse, hor", "horse", [])  # with 4 new tokens a cut-off fragment is dropped


def test_hedged_say_rows_are_asked_again_with_their_own_intervention():
    vs2 = load()
    rows = [
        dict(input="moving", task="say", scene=3, family="transplant", layers=(7, 8, 9, 10)),
        dict(input="moving", task="animal", scene=3, family="transplant", layers=(7, 8, 9, 10)),
        dict(input="moving", task="say", scene=4, family="clean", layers=()),
    ]
    res = [("cat or dog", -1.0, -2.0), ("7", -0.1, -0.2), ("horse", -0.3, -0.4)]
    seen = {}

    def encode_followup(r, first):
        return dict(first=first, scene=r["scene"])

    def run_rows(encs, again, label):
        seen.update(encs=encs, again=again)
        return [("dog", -9.0, -9.0, None)]

    out = vs2.ask_again(rows, res, encode_followup, run_rows, "test")
    assert list(seen["encs"]) == [("moving", "say~0", 3)] and seen["encs"][("moving", "say~0", 3)]["first"] == "cat or dog"
    assert seen["again"][0]["family"] == "transplant" and seen["again"][0]["layers"] == (7, 8, 9, 10)  # the same intervention
    assert out[0] == ("dog", -1.0, -2.0) and rows[0]["first_answer"] == "cat or dog"  # the reply, the question's log-probabilities
    assert out[1:] == res[1:] and "first_answer" not in rows[2] and rows[0]["task"] == "say"
    assert vs2.ask_again(rows[2:], res[2:], encode_followup, lambda *a: 1 / 0, "none") == res[2:]  # no hedge: nothing is run


def test_followup_prompt_continues_the_say_conversation():
    vs2 = load()
    seen = []
    lm = types.SimpleNamespace(chat_messages=lambda msgs: seen.append(msgs) or "")
    vs2.followup_prompt(lm, "frozen", "cat or dog")
    m = seen[0]
    assert [x["role"] for x in m] == ["user", "assistant", "user"] and [p["type"] for p in m[0]["content"]] == ["text", "video", "text"]
    assert m[0]["content"][2]["text"] == vs2.QUESTIONS2["say"] and m[1]["content"][0]["text"] == "cat or dog"
    assert m[2]["content"][0]["text"] == vs2.naming3.FOLLOWUP
