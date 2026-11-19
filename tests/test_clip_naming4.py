"""clip_naming4 on the CPU: clip_naming's question, with clip_naming3's follow-up and scoring."""

import importlib
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load():
    """experiments/clip_naming4.py, with a stand-in for jspace.model where its dependencies (jlens) are missing"""
    sys.path.insert(0, str(ROOT / "experiments"))
    stub = None
    try:
        importlib.import_module("jspace.model")
    except ImportError:
        stub = types.ModuleType("jspace.model")
        stub.Session = None
        sys.modules["jspace.model"] = stub
    try:
        return importlib.import_module("clip_naming4")
    finally:
        sys.path.remove(str(ROOT / "experiments"))
        if stub is not None:
            del sys.modules["jspace.model"]


def test_the_question_is_clip_namings_and_the_rest_clip_naming3s():
    cn = load()
    c1, c3 = importlib.import_module("clip_naming"), importlib.import_module("clip_naming3")
    assert cn.QUESTION == c1.QUESTION and "general word" not in cn.QUESTION
    assert cn.score is c3.score and cn.hedged is c3.hedged and cn.summarize is c3.summarize and cn.FOLLOWUP == c3.FOLLOWUP
    assert cn.N_NEW == 24 and cn.GATED_INPUT == "moving"


def test_the_prompt_and_its_follow_up():
    cn = load()
    seen = []
    lm = types.SimpleNamespace(chat_messages=lambda msgs: seen.append(msgs) or "")
    cn.prompt(lm, "moving")
    cn.prompt(lm, "still", "horse, cow")
    assert len(seen[0]) == 1 and [p["type"] for p in seen[0][0]["content"]] == ["video", "text"] and seen[0][0]["content"][1]["text"] == cn.QUESTION
    assert [m["role"] for m in seen[1]] == ["user", "assistant", "user"] and seen[1][0]["content"][0] == {"type": "image"}
    assert seen[1][1]["content"][0]["text"] == "horse, cow" and seen[1][2]["content"][0]["text"] == cn.FOLLOWUP
