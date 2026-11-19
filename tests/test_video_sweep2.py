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


def test_say_is_scored_as_clip_naming2_and_the_rules_as_video_sweep():
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
