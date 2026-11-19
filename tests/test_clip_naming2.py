"""clip_naming2's question and scoring on the CPU, including the answers that motivated it (PR #7's Gemma runs)."""

import importlib
import sys
import types
from pathlib import Path

from jspace import matched

ROOT = Path(__file__).resolve().parents[1]


def load():
    """experiments/clip_naming2.py, with a stand-in for jspace.model where its dependencies (jlens) are missing"""
    sys.path.insert(0, str(ROOT / "experiments"))
    stub = None
    try:
        importlib.import_module("jspace.model")
    except ImportError:
        stub = types.ModuleType("jspace.model")
        stub.Session = None
        sys.modules["jspace.model"] = stub
    try:
        return importlib.import_module("clip_naming2")
    finally:
        sys.path.remove(str(ROOT / "experiments"))
        if stub is not None:
            del sys.modules["jspace.model"]


def test_head_noun_and_kinds_count():
    cn = load()
    for text, label in [
        ("race car", "car"),
        ("Sports cars.", "car"),
        ("coffee cup", "cup"),
        ("office chair", "chair"),
        ("Pond bird", "bird"),
        ("military aircraft", "airplane"),
        ("electric bicycle", "bicycle"),
        ("duck", "bird"),
        ("Sailboat", "boat"),
        ("a wooden dining table", "dining table"),
        ("cat", "cat"),
    ]:
        assert cn.score(text, label)[0], (text, label)


def test_what_stays_wrong():
    cn = load()
    for text, label in [
        ("subway car", "train"),  # its head noun is car, not train
        ("animal", "horse"),
        ("food", "pizza"),
        ("Vehicle", "car"),
        ("dog", "horse"),
        ("coffee", "cup"),
        ("hot dog", "dog"),  # a COCO name of its own, not a kind of dog
        ("teddy bear", "bear"),
        ("Horse, cow, horse", "cow"),  # a list: no head noun is read
        ("laptop;computer", "computer"),
        ("cat or dog", "dog"),
        ("", "cat"),
    ]:
        assert not cn.score(text, label)[0], (text, label)


def test_only_the_first_line_is_read():
    cn = load()
    assert cn.score("dog\ncat", "dog")[0] and not cn.score("dog\ncat", "cat")[0]


def test_extra_names_are_kinds_of_coco_categories():
    cn = load()
    assert set(cn.EXTRA_NAMES2.values()) <= set(cn.base.COCO_NAMES) and not set(cn.EXTRA_NAMES2) & set(cn.base.COCO_NAMES)
    assert set(cn.base.EXTRA_NAMES.items()) <= set(cn.EXTRA_NAMES2.items())


def test_question_names_no_label():
    cn = load()
    assert cn.QUESTION.startswith(cn.base.QUESTION.split(" Return only")[0])
    assert cn.QUESTION.endswith("Return only the category name, nothing else.")
    added = "Name the object itself, not a general word such as animal, food, vehicle or furniture."
    assert added in cn.QUESTION and not any(w in added.lower().split() for w in matched.VOCAB)
