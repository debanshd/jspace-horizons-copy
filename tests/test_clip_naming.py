"""clip_naming's pieces on the CPU: the scoring, the summary's rule on made-up answers, the frame count and the prompt."""

import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from jspace import matched

ROOT = Path(__file__).resolve().parents[1]


def load():
    """experiments/clip_naming.py, with a stand-in for jspace.model where its dependencies (jlens) are missing"""
    sys.path.insert(0, str(ROOT / "experiments"))
    stub = None
    try:
        importlib.import_module("jspace.model")
    except ImportError:
        stub = types.ModuleType("jspace.model")
        stub.Session = None
        sys.modules["jspace.model"] = stub
    try:
        return importlib.import_module("clip_naming")
    finally:
        sys.path.remove(str(ROOT / "experiments"))
        if stub is not None:
            del sys.modules["jspace.model"]


def test_scoring_is_exact_after_normalizing():
    cn = load()
    assert cn.score("Airplane.", "airplane") == (True, True)
    assert cn.score("  airplane\n", "airplane") == (True, True)
    assert cn.score("plane", "airplane") == (False, False)
    assert cn.score("Aeroplane, Airplane, Air", "airplane") == (False, False)  # a list of names is not the name
    assert cn.score("Sailboat", "boat") == (False, False)
    assert cn.score("Dining table", "dining table") == (True, True)
    assert cn.score("dog", "cat") == (False, True)  # wrong, but a COCO name
    assert len(cn.COCO_NAMES) == 80 and len(set(cn.COCO_NAMES)) == 80
    assert set(matched.VOCAB) <= set(cn.COCO_NAMES)  # every label of the set is a COCO name, so the question can be answered


def made_up(clips, wrong):
    """answers: every clip right on every input, except (clip, input) pairs in `wrong`"""
    rows = []
    for c in clips:
        for inp in matched.INPUTS:
            ok = (c, inp) not in wrong
            rows.append(dict(clip=c, label="cat", input=inp, answer="cat" if ok else "dog", correct=ok, coco=True))
    return pd.DataFrame(rows)


def test_summary_rule_on_made_up_answers():
    cn = load()
    clips = [f"c{i:02d}" for i in range(20)]
    R = made_up(clips, {("c00", "moving"), ("c01", "still"), ("c02", "moving"), ("c02", "frozen")})
    schedule = [(0, "c00", "c05"), (1, "c05", "c06"), (2, "c07", "c01"), (3, "c08", "c09")]
    out = cn.summarize(R, schedule)
    assert out["share"]["moving"][0] == pytest.approx(18 / 20) and out["gate"]["moving"]  # 0.90 passes
    assert out["share"]["frozen"][0] == pytest.approx(19 / 20) and out["gate"]["frozen"]
    assert out["unreliable"] == ["c00", "c01", "c02"] and len(out["reliable"]) == 17
    assert out["gated_scenes"] == [1, 3]  # scene 0 has an unreliable receiver, scene 2 an unreliable donor
    lo, hi = out["share"]["moving"][1:]
    assert lo <= 0.9 <= hi
    out2 = cn.summarize(made_up(clips, {(c, "moving") for c in clips[:3]}), schedule)
    assert not out2["gate"]["moving"] and out2["gate"]["still"]  # 0.85 fails


def test_summary_stops_on_a_missing_answer():
    cn = load()
    R = made_up(["a", "b"], set())
    with pytest.raises(AssertionError):
        cn.summarize(R[~((R["clip"] == "b") & (R["input"] == "still"))], [])


def test_frame_count_is_capped_at_the_clip_length():
    vp = types.SimpleNamespace(
        sample_frames=lambda md, num_frames=None: (
            np.linspace(0, md.total_num_frames - 1, num_frames or 32).astype(int) if (num_frames or 32) <= md.total_num_frames else (_ for _ in ()).throw(ValueError("too many"))
        )
    )
    assert matched.scene_frame_count(vp, 13, 13, 24.0) == 13  # Gemma's 32 on a 13-frame clip: the clip's length
    assert matched.scene_frame_count(vp, 96, 96, 24.0) == 32


def test_prompt_puts_the_media_before_the_question():
    cn = load()
    seen = []
    lm = types.SimpleNamespace(chat_messages=lambda msgs: seen.append(msgs) or "")
    cn.prompt(lm, "moving")
    cn.prompt(lm, "still")
    assert [p["type"] for p in seen[0][0]["content"]] == ["video", "text"] and seen[0][0]["content"][1]["text"] == cn.QUESTION
    assert seen[1][0]["content"][0] == {"type": "image"}
