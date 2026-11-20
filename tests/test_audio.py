"""The audio experiments' pieces on the CPU: the recordings, the prompt orders, the naming rule, and the sweep's per-question
donor states and audio-token alignment check."""

import importlib
import sys
import types
import wave
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from jspace import matched, spoken

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    """a script from experiments/, with a stand-in for jspace.model where its dependencies (jlens) are missing"""
    sys.path.insert(0, str(ROOT / "experiments"))
    stub = None
    try:
        importlib.import_module("jspace.model")
    except ImportError:
        stub = types.ModuleType("jspace.model")
        stub.Session = None
        sys.modules["jspace.model"] = stub
    try:
        return importlib.import_module(name)
    finally:
        sys.path.remove(str(ROOT / "experiments"))
        if stub is not None:
            del sys.modules["jspace.model"]


def write_wav(path, x, rate=16000):
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes((np.clip(x, -1, 1 - 2**-15) * 32768).astype("<i2").tobytes())


def test_load_wav_reads_pcm16_and_refuses_another_rate_or_length(tmp_path):
    x = np.sin(np.linspace(0, 200, spoken.N_SAMPLES)).astype(np.float32) * 0.5
    write_wav(tmp_path / "a.wav", x)
    y = spoken.load_wav(tmp_path / "a.wav")
    assert y.dtype == np.float32 and y.shape == (spoken.N_SAMPLES,) and np.abs(y - x).max() < 1e-4
    write_wav(tmp_path / "b.wav", x, rate=22050)
    with pytest.raises(AssertionError):
        spoken.load_wav(tmp_path / "b.wav")
    write_wav(tmp_path / "c.wav", x[:1000])
    with pytest.raises(AssertionError):
        spoken.load_wav(tmp_path / "c.wav")


def test_controls_and_questions():
    assert not spoken.silence().any() and spoken.silence().shape == (spoken.N_SAMPLES,)
    n = spoken.noise()
    assert np.array_equal(n, spoken.noise()) and 0.015 < float(n.std()) < 0.025  # seeded, speech-like level
    assert list(spoken.QUESTIONS) == ["say", "animal", "letter", "set", "ride"]
    assert "{a}" in spoken.QUESTIONS["set"] and all("audio" in q for q in spoken.QUESTIONS.values())


def test_the_two_prompt_orders():
    q = spoken.content("q_audio", "Q?")
    a = spoken.content("audio_q", "Q?")
    assert [p["type"] for p in q] == ["text", "audio"] and [p["type"] for p in a] == ["audio", "text"]
    with pytest.raises(ValueError):
        spoken.content("moving", "Q?")


def test_recordings_are_consistent_across_scenes():
    rows = [
        dict(
            row_id="r0",
            donor_label="bird",
            donor_audio="audio/w2.wav",
            donor_sha256="b",
            donor_audio_id="w2",
            receiver_label="cat",
            receiver_audio="audio/w6.wav",
            receiver_sha256="c",
            receiver_audio_id="w6",
        ),
        dict(
            row_id="r1",
            donor_label="cat",
            donor_audio="audio/w6.wav",
            donor_sha256="c",
            donor_audio_id="w6",
            receiver_label="bird",
            receiver_audio="audio/w2.wav",
            receiver_sha256="b",
            receiver_audio_id="w2",
        ),
    ]
    assert spoken.recordings(rows) == {"w2": ("bird", "audio/w2.wav", "b"), "w6": ("cat", "audio/w6.wav", "c")}
    rows[1]["receiver_sha256"] = "x"  # the same recording id with another hash
    with pytest.raises(AssertionError):
        spoken.recordings(rows)


def test_naming_messages_and_follow_up():
    an = load("audio_naming")
    m = an.messages("q_audio")
    assert [p["type"] for p in m[0]["content"]] == ["text", "audio"] and m[0]["content"][0]["text"] == spoken.QUESTIONS["say"]
    t = an.messages("typed", "horse")
    assert [p["type"] for p in t[0]["content"]] == ["text", "text"] and t[0]["content"][1]["text"] == "Transcript of the audio: horse"
    f = an.messages("audio_q", None, "cat or dog")
    assert [x["role"] for x in f] == ["user", "assistant", "user"] and f[2]["content"][0]["text"] == an.c3.FOLLOWUP
    assert an.names_object("a dog", False, ["cat", "dog"]) and not an.names_object("silence", False, ["cat", "dog"])


def test_naming_rule_on_made_up_answers():
    an = load("audio_naming")
    recs = {"w0": ("cat", "", ""), "w1": ("dog", "", ""), "w2": ("horse", "", "")}
    wrong = {("w2", "q_audio")}  # horse missed only with the question first
    rows = [dict(item=k, label=lab, input=i, correct=(k, i) not in wrong, hedged=False, answer=lab, names_object=True) for k, (lab, _, _) in recs.items() for i in an.INPUTS]
    rows += [dict(item=c, label="", input=i, correct=False, hedged=False, answer="nothing", names_object=False) for c in an.CONTROLS for i in spoken.INPUTS]
    schedule = [(0, "w0", "w1"), (1, "w2", "w0"), (2, "w1", "w0")]
    out = an.summarize(pd.DataFrame(rows), schedule, recs)
    assert out["share"]["q_audio"][0] == pytest.approx(2 / 3) and not out["gate"]["q_audio"] and out["gate"]["audio_q"]
    assert out["gated_by_input"]["q_audio"] == [0, 2] and out["gated_by_input"]["audio_q"] == [0, 1, 2]
    assert out["gated_scenes"] == [0, 2]  # named in both orders
    assert out["named"]["q_audio"] == 2 and out["scenes_lost"]["q_audio"] == {"w2": 1} and out["scenes_lost"]["audio_q"] == {}
    assert out["share_by_scene"]["q_audio"] == pytest.approx(2 / 3) and out["controls"]["audio_q"]["silence"]["names_object"] is False
    assert "typed" not in out["gated_by_input"] and out["gate"]["typed"]


def test_sweep_rows_take_the_donor_state_of_their_own_question():
    sw = load("audio_sweep")
    r = dict(scene=3, X="cat", Y="car", Z="dog", clip="w6", set=["cat", "bus", "clock"])
    ID = {w: [1] for w in matched.VOCAB} | {c: [2] for c in ["7", "3", *{w[0].upper() for w in matched.VOCAB}]}
    rows = sw.patch_rows("q_audio", [r], ID, 32)
    for x in rows:
        if x["family"] in ("full", "transplant"):
            assert x["donor"] == (f"donor~{x['task']}", 3)
        if x["family"] == "self":
            assert x["donor"] == (f"self~{x['task']}", 3)
    assert {x["task"] for x in rows} == set(sw.QUESTIONS_A) and sw.capture_input("q_audio", "donor", "say") == "q_audio@donor~say"
    assert len(rows) == len(sw.QUESTIONS_A) * (3 + 3 * len(sw.windows(32)))


def test_alignment_check_stops_on_a_mismatch():
    sw = load("audio_sweep")
    A = 9  # the audio token
    ids = lambda pre, n, post=2: {"input_ids": torch.tensor([[5] * pre + [A] * n + [6] * post])}  # noqa: E731
    encs = {}
    for task in sw.QUESTIONS_A:
        encs[("q_audio", task, 0)] = ids(7, 4)
        encs[(sw.capture_input("q_audio", "donor", task), task, 0)] = ids(7, 4)
        encs[(sw.capture_input("q_audio", "self", task), task, 0)] = ids(7, 4)
    assert sw.check_alignment(encs, "q_audio", [0], A) == {t: [4] for t in sw.QUESTIONS_A}
    encs[(sw.capture_input("q_audio", "donor", "set"), "set", 0)] = ids(8, 4)  # shifted one position
    with pytest.raises(SystemExit):
        sw.check_alignment(encs, "q_audio", [0], A)


def test_prompts_follow_the_order_and_the_dataset_wording():
    sw = load("audio_sweep")
    r = dict(set=["cat", "bus", "clock"])
    m = sw.prompt_messages("audio_q@donor~set", "set", r)[0]["content"]
    assert [p["type"] for p in m] == ["audio", "text"] and m[1]["text"] == spoken.QUESTIONS["set"].format(a="cat", b="bus", c="clock")
    f = sw.followup_messages("q_audio", "cat or dog")
    assert [p["type"] for p in f[0]["content"]] == ["text", "audio"] and f[1]["content"][0]["text"] == "cat or dog"


def test_order_compare():
    sw = load("audio_sweep")
    a = dict(L_say=19, L_use=7, rule_read=True)
    assert sw.order_compare(dict(cut={"q_audio": a, "audio_q": dict(a)}))["same"]
    assert not sw.order_compare(dict(cut={"q_audio": a, "audio_q": dict(L_say=7, L_use=3, rule_read=True)}))["same"]
    assert sw.order_compare(dict(cut={"q_audio": a, "audio_q": "not perceived, untestable"})) == "untestable"
