"""strict.py's whole-answer scoring: the answers the inherited scoring credits wrongly, and the ones it should keep."""

import pytest

from jspace import matched, strict

MEM = ["bus", "clock", "dog"]


@pytest.mark.parametrize(
    "task,text,word",
    [("say", "not a cat", "cat"), ("animal", "7 or 3", "cat"), ("letter", "Cannot determine", "car"), ("say", "cat or dog", "cat"), ("set", "7, because", "bus"), ("say", "a cat and a dog", "cat")],
)
def test_partial_answers_are_wrong_here_and_were_credited_before(task, text, word):
    assert matched.matches(task, text, matched.truth(task, word, MEM))  # the inherited scoring credits them
    assert not strict.matches(task, text, word, MEM) and not strict.valid(task, text)


@pytest.mark.parametrize(
    "task,text,word",
    [
        ("say", "Cat", "cat"),
        ("say", "**Cat.**", "cat"),
        ("say", "the bus", "bus"),
        ("say", " airplane\n", "airplane"),
        ("letter", "C.", "car"),
        ("letter", "**c**", "car"),
        ("animal", "7", "dog"),
        ("ride", "3", "cup"),
        ("set", "7.", "clock"),
    ],
)
def test_whole_answers_are_correct(task, text, word):
    assert strict.matches(task, text, word, MEM) and strict.valid(task, text)


def test_exact_and_synonym_are_apart():
    assert strict.say("cats", "cat") == (False, True)
    assert strict.say("aeroplane", "airplane") == (False, True)
    assert strict.say("Kittens", "cat") == (False, True)
    assert strict.say("airplane", "airplane") == (True, True)
    assert strict.say("dog", "cat") == (False, False)
    s = strict.score("say", "puppy", "dog", MEM)
    assert s == dict(correct=False, synonym=True, valid=True)


def test_an_answer_at_the_token_limit_is_not_whole():
    assert strict.score("say", "cat", "cat", MEM, hit_limit=True) == dict(correct=False, synonym=False, valid=False)
    assert not strict.matches("animal", "7", "cat", MEM, hit_limit=True)


def test_truth_is_matcheds():
    for w in matched.VOCAB:
        for t in ("say", "letter", "set", "animal", "ride"):
            assert strict.truth(t, w, MEM) == matched.truth(t, w, MEM)


def test_synonyms_are_clip_naming2s():
    import importlib
    import sys
    import types
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    stub = None
    try:
        importlib.import_module("jspace.model")
    except ImportError:  # no jlens here: a stand-in, as test_audio's load() uses
        stub = sys.modules["jspace.model"] = types.ModuleType("jspace.model")
        stub.Session = None
    try:
        n2 = importlib.import_module("clip_naming2")
    finally:
        sys.path.pop(0)
        if stub is not None:
            del sys.modules["jspace.model"]
    assert strict.SYNONYMS == n2.EXTRA_NAMES2


def test_phrase_level_takes_a_modifier_not_a_list_or_a_negation():
    for text, word in [("sports car", "car"), ("Wall clock.", "clock"), ("a coffee cup", "cup"), ("military aircraft", "airplane"), ("cat", "cat")]:
        assert strict.phrase(text, word), text
    for text, word in [("not a cat", "cat"), ("cat or dog", "dog"), ("laptop;computer", "computer"), ("bicycle, motor", "motor"), ("a small grey house cat", "cat"), ("car door", "car")]:
        assert not strict.phrase(text, word), text
    assert not strict.say("sports car", "car")[1]  # the synonym level, the primary one, does not take it


def test_rescore_uses_the_first_answer_and_keeps_the_saved_scores():
    import importlib.util
    import sys
    import types

    import pandas as pd

    stub = None
    if "jspace.model" not in sys.modules:
        try:
            importlib.import_module("jspace.model")
        except ImportError:
            stub = sys.modules["jspace.model"] = types.ModuleType("jspace.model")
            stub.Session = None
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("rescore_strict", root / "scripts/rescore_strict.py")
    rs = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(rs)
    finally:
        if stub is not None:
            del sys.modules["jspace.model"]
    R = pd.DataFrame(
        [
            dict(family="pclean", task="say", scene=0, X="car", Y="dog", Z="bus", first_answer="car or dog", generation="car", correct=True, follows_Y=False, follows_Z=False),
            dict(family="pclean", task="say", scene=0, X="car", Y="dog", Z="bus", first_answer=float("nan"), generation="sports car", correct=True, follows_Y=False, follows_Z=False),
            dict(family="pclean", task="letter", scene=0, X="car", Y="dog", Z="bus", first_answer=float("nan"), generation="Cannot determine", correct=True, follows_Y=False, follows_Z=False),
            dict(family="steer", task="say", scene=0, X="car", Y="dog", Z="bus", first_answer="", generation="car", correct=True, follows_Y=False, follows_Z=False),
        ]
    )
    out = rs.rescore(R, {0: ["bus", "clock", "dog"]})
    assert len(out) == 3 and out["scored"].tolist() == ["car or dog", "sports car", "Cannot determine"]
    assert out["correct"].tolist() == [False, False, False] and out["old_correct"].all()
    assert rs.rescore(R, {0: ["bus", "clock", "dog"]}, "phrase")["correct"].tolist() == [False, True, False]
