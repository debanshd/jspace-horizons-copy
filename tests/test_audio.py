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


def test_clean_messages_and_follow_up():
    ac = load("audio_clean")
    m = ac.messages("q_audio", spoken.QUESTIONS["say"])
    assert [p["type"] for p in m[0]["content"]] == ["text", "audio"] and m[0]["content"][0]["text"] == spoken.QUESTIONS["say"]
    t = ac.messages("typed", "Q?", "horse")
    assert [p["type"] for p in t[0]["content"]] == ["text", "text"] and t[0]["content"][1]["text"] == "Transcript of the audio: horse"
    f = ac.messages("audio_q", "Q?", None, "cat or dog")
    assert [x["role"] for x in f] == ["user", "assistant", "user"] and f[2]["content"][0]["text"] == ac.c3.FOLLOWUP
    assert ac.question("set", ["cat", "bus", "clock"]) == spoken.QUESTIONS["set"].format(a="cat", b="bus", c="clock")


def test_clean_uses_the_sweeps_decoding():
    ac, sw = load("audio_clean"), load("audio_sweep")
    assert ac.N_NEW == sw.N_NEW == 4 and ac.TASKS == list(sw.QUESTIONS_A)


def clean_rows(ac, recs, schedule, wrong=(), hedge=(), wrong_task=()):
    """made-up answers: every scene, role, input and task right except the (recording, input) in wrong (say) and the
    (role, input, task) in wrong_task"""
    rows = []
    for s, rc, dc in schedule:
        for role, a in (("receiver", rc), ("donor", dc)):
            for inp in ac.INPUTS:
                for t in ac.TASKS:
                    ok = not ((t == "say" and (a, inp) in wrong) or (role, inp, t) in wrong_task)
                    h = t == "say" and (a, inp) in hedge
                    rows.append(
                        dict(
                            kind="scene",
                            scene=s,
                            role=role,
                            recording=a,
                            label=recs[a][0],
                            input=inp,
                            task=t,
                            answer="x",
                            hit_limit=False,
                            hedged=h,
                            followup="",
                            correct=ok,
                            synonym=ok,
                            valid=True,
                            followup_correct=h,
                        )
                    )
    rows += [
        dict(
            kind="control",
            scene=-1,
            role="",
            recording=c,
            label="",
            input=i,
            task="say",
            answer="nothing",
            hit_limit=False,
            hedged=False,
            followup="",
            correct=False,
            synonym=False,
            valid=False,
            followup_correct=False,
        )
        for c in ac.CONTROLS
        for i in spoken.INPUTS
    ]
    return pd.DataFrame(rows)


def test_clean_rule_on_made_up_answers():
    ac = load("audio_clean")
    recs = {f"w{i}": (w, "", "") for i, w in enumerate(["cat", "dog", "horse", "bus", "car", "cup", "bed"])}
    schedule = [(i, f"w{i % 7}", f"w{(i + 1) % 7}") for i in range(7)]
    R = clean_rows(ac, recs, schedule, wrong={("w2", "q_audio")}, hedge={("w2", "q_audio")})
    out = ac.summarize(R, schedule, recs)
    rc = out["recognition"]["q_audio"]
    assert rc["named"] == 6 and rc["missed"] == {"w2": dict(label="horse", answer="x", scenes_lost=2)} and not rc["gate_090"]
    assert out["recognition"]["audio_q"]["named"] == 7 and out["secondary"]["followup"]["q_audio"] == dict(asked=1, correct_after=7)
    assert out["gated_by_input"]["q_audio"] == [0, 3, 4, 5, 6] and out["gated_scenes"] == [0, 3, 4, 5, 6]  # scenes 1, 2 use w2
    assert out["ready"] and out["clean"]["q_audio"]["animal"]["donor"]["passes"]
    assert out["controls"]["audio_q"]["silence"]["names_object"] is False and "passes" not in out["clean"]["typed"]["say"]["receiver"]
    # the donors' animal answers under the receiver's question fail: not ready, though every recording is named
    R = clean_rows(ac, recs, schedule, wrong_task={("donor", "audio_q", "animal")})
    out = ac.summarize(R, schedule, recs)
    assert len(out["gated_scenes"]) == 7 and not out["clean"]["audio_q"]["animal"]["donor"]["passes"] and not out["ready"]
    # fewer than 5 gated scenes: not ready
    R = clean_rows(ac, recs, schedule, wrong={("w0", "audio_q"), ("w3", "audio_q")})
    assert not ac.summarize(R, schedule, recs)["ready"]


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
        if x["family"] == "dclean":
            assert x["input"] == f"q_audio@donor~{x['task']}" and not x["site"] and x["donor"] is None
    assert {x["task"] for x in rows} == set(sw.QUESTIONS_A) and sw.capture_input("q_audio", "donor", "say") == "q_audio@donor~say"
    assert len(rows) == len(sw.QUESTIONS_A) * (4 + 3 * len(sw.windows(32)))


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


def test_order_compare():
    sw = load("audio_sweep")
    a = dict(L_say=19, L_use=7, rule_read=True)
    assert sw.order_compare(dict(cut={"q_audio": a, "audio_q": dict(a)}))["same"]
    assert not sw.order_compare(dict(cut={"q_audio": a, "audio_q": dict(L_say=7, L_use=3, rule_read=True)}))["same"]
    assert sw.order_compare(dict(cut={"q_audio": a, "audio_q": "not perceived, untestable"})) == "untestable"


def test_sweep_scores_the_whole_first_answer():
    sw = load("audio_sweep")
    sw.vs2.SETS[3] = ["cat", "bus", "clock"]
    base = dict(scene=3, X="cat", Y="car", Z="dog", ids_x=[1], ids_y=[2], ko=None, layers=(), donor=None, input="q_audio", family="pclean")
    rows = [dict(base, task=t) for t in ("say", "say", "say", "animal", "letter", "set", "say")]
    res = [("Cat.", 0.0, -1.0, False), ("not a cat", 0.0, -1.0, False), ("cats", 0.0, -1.0, False), ("7 or 3", 0, 0, False), ("Cannot determine", 0, 0, False), ("7", 0, 0, False), ("cat", 0, 0, True)]
    R = sw.to_frame(rows, res)
    assert R["correct"].tolist() == [True, False, False, False, False, True, False]
    assert R["synonym"].tolist() == [True, False, True, False, False, True, False] and R["hit_limit"].tolist()[-1]
    assert "first_answer" not in R and not R["follows_Y"].any()


def test_controls_pass_needs_every_gate():
    sw = load("audio_sweep")
    good = dict(gate={"q_audio": {"say": dict(passes=True), "animal": dict(passes=True)}}, controls={"q_audio": dict(machinery_ok=True, positive_ok=True)})
    assert sw.controls_pass(good, ["q_audio"]) == {"q_audio": True, "all": True}
    bad = dict(good, controls={"q_audio": dict(machinery_ok=True, positive_ok=False)})
    assert sw.controls_pass(bad, ["q_audio"])["all"] is False


def test_engine_reports_the_token_limit_only_when_asked():
    from jspace import batched

    class Gen:
        def __init__(self):
            self.eos_token_id = [1, 106]

    seqs = torch.tensor([[5, 5, 7, 8, 9, 10], [5, 5, 7, 106, 0, 0]])  # two prompt tokens; the first row never ends
    lm = types.SimpleNamespace(
        device="cpu",
        tokenizer=types.SimpleNamespace(pad_token_id=0, decode=lambda ids, skip_special_tokens: " ".join(str(int(i)) for i in ids)),
        hf=types.SimpleNamespace(generation_config=Gen(), generate=lambda **kw: types.SimpleNamespace(sequences=seqs, scores=[torch.zeros(2, 200)])),
        register=lambda L, f: None,
    )
    eng = types.SimpleNamespace(lm=lm, token={"a": 5}, T=None, read_layers=[], attn=[], _mm=lambda rows: {})
    rows = [dict(input="a", ids_x=[1], ids_y=[2], ko=None, site=None), dict(input="a", ids_x=[1], ids_y=[2], ko=None, site=None)]
    enc = {"input_ids": torch.tensor([[5, 5], [5, 5]]), "attention_mask": torch.ones(2, 2, dtype=torch.long)}
    orig = batched.residual_hooks
    batched.residual_hooks = lambda *a: []
    try:
        res = batched.Engine._generate_enc(eng, rows, enc, 4)
    finally:
        batched.residual_hooks = orig
    assert [r[4] for r in res] == [True, False] and len(res[0]) == 5


def test_gate_script(tmp_path):
    import json
    import subprocess

    clean = dict(
        gated_scenes=[0, 1],
        n_recordings=15,
        recognition={"q_audio": dict(named=15, missed={})},
        clean={"q_audio": {"say": {"receiver": dict(gated=[1.0, 1, 1]), "donor": dict(gated=None)}}},
        ready=False,
    )
    (tmp_path / "c.json").write_text(json.dumps(clean))
    r = subprocess.run([sys.executable, str(ROOT / "scripts/audio_gate.py"), "clean", str(tmp_path / "c.json")], capture_output=True, text=True)
    assert r.returncode == 1 and "FAILS" in r.stdout
    pilot = dict(
        controls_pass={"all": True},
        patch=dict(controls={"q_audio": dict(self_patch_same=1.0, every_layer_say_follows_donor=[1.0, 1, 1])}, gate={"q_audio": {"say": dict(clean=1.0), "animal": dict(clean=1.0)}}),
        donor_clean={"q_audio": {"say": 1.0, "animal": 0.8}},
        hit_limit=0.0,
    )
    (tmp_path / "p.json").write_text(json.dumps(pilot))
    r = subprocess.run([sys.executable, str(ROOT / "scripts/audio_gate.py"), "pilot", str(tmp_path / "p.json")], capture_output=True, text=True)
    assert r.returncode == 1  # the donors' animal answers are below 0.85
    pilot["donor_clean"]["q_audio"]["animal"] = 0.9
    (tmp_path / "p.json").write_text(json.dumps(pilot))
    r = subprocess.run([sys.executable, str(ROOT / "scripts/audio_gate.py"), "pilot", str(tmp_path / "p.json")], capture_output=True, text=True)
    assert r.returncode == 0 and "passes" in r.stdout


def test_rule_variants_and_their_truth():
    ar = load("audio_rules")
    mem = ["cat", "bus", "clock"]
    assert ar.question("A", "set", mem) == spoken.QUESTIONS["set"].format(a="cat", b="bus", c="clock")
    assert ar.question("B", "animal", mem) == "Rule: if the object named in the audio is an animal, answer 3; otherwise answer 7. Reply with the number only."
    assert ar.question("D", "ride", mem).startswith("Rule: answer 3, unless the object named in the audio is something people ride or drive")
    assert [ar.truth(v, "animal", "dog", mem) for v in "ABD"] == ["7", "3", "7"]
    assert [ar.truth(v, "animal", "car", mem) for v in "ABD"] == ["3", "7", "3"]
    assert [ar.truth(v, "set", "bus", mem) for v in "ABD"] == ["7", "3", "7"]


def rule_rows(ar, wrong_digit):
    """made-up answers: half the items hold the condition; every answer is right except, in B and D, 20 items answered
    wrong_digit[v] where that digit is wrong"""
    rows = []
    for v in "ABD":
        for i in range(40):
            right = ar.YES[v] if i % 2 else ("3" if ar.YES[v] == "7" else "7")
            d = wrong_digit.get(v) if wrong_digit.get(v) and wrong_digit[v] != right and i < 40 else right
            rows.append(dict(variant=v, task="animal", input="audio_q", role="receiver" if i < 20 else "donor", recording=f"w{i % 15}", digit=d, correct=d == right, valid=True, hit_limit=False))
    return pd.DataFrame(rows)


def test_rule_diagnosis():
    ar = load("audio_rules")
    assert ar.diagnose(rule_rows(ar, {"B": "7", "D": "7"}))["verdict"] == "digit 7"
    assert ar.diagnose(rule_rows(ar, {"B": "3", "D": "7"}))["verdict"] == "yes"
    assert ar.diagnose(rule_rows(ar, {"B": "3", "D": "3"}))["verdict"] == "first named"
    assert ar.diagnose(rule_rows(ar, {}))["verdict"] == "too few errors"
    R = rule_rows(ar, {"B": "3", "D": "3"})
    flip = R.index[(R["variant"] == "D") & ~R["correct"]][:10]  # half of D's errors the other way: no explanation fits
    R.loc[flip, "digit"] = "7"
    R.loc[flip, "correct"] = R.loc[flip, "digit"] == "3"  # 7 is right where 3 was wrong, so these become right; add errors
    extra = R[(R["variant"] == "D") & R["correct"]].index[:12]
    R.loc[extra, "digit"] = ["7" if d == "3" else "3" for d in R.loc[extra, "digit"]]
    R.loc[extra, "correct"] = False
    assert ar.diagnose(R)["verdict"] == "mixed"


def test_one_order_readiness():
    ac = load("audio_clean")
    recs = {f"w{i}": (w, "", "") for i, w in enumerate(["cat", "dog", "horse", "bus", "car", "cup", "bed"])}
    schedule = [(i, f"w{i % 7}", f"w{(i + 1) % 7}") for i in range(7)]
    R = clean_rows(ac, recs, schedule, wrong={("w2", "q_audio")}, wrong_task={("donor", "audio_q", "animal")})
    q = ac.order_ready(R, schedule, recs, "q_audio")
    assert q["ready"] and q["named"] == 6 and q["gated_scenes"] == [0, 3, 4, 5, 6] and q["clean"]["animal donor"] == 1.0
    a = ac.order_ready(R, schedule, recs, "audio_q")
    assert not a["ready"] and a["gated_scenes"] == list(range(7)) and a["clean"]["animal donor"] == 0.0


def test_qfirst_schedule_reads_one_order(tmp_path, monkeypatch):
    qf = load("audio_sweep_qfirst")
    ac = qf.ac
    recs = {f"w{i}": (w, "", "") for i, w in enumerate(["cat", "dog", "horse", "bus", "car", "cup", "bed"])}
    meta = [dict(scene_id=i, receiver_audio_id=f"w{i % 7}", donor_audio_id=f"w{(i + 1) % 7}") for i in range(7)]
    schedule = [(m["scene_id"], m["receiver_audio_id"], m["donor_audio_id"]) for m in meta]
    clean_rows(ac, recs, schedule, wrong_task={("donor", "audio_q", "animal")}).to_csv(tmp_path / "audio_clean.csv", index=False)
    monkeypatch.setattr(qf.spoken, "scenes", lambda d: meta)
    monkeypatch.setattr(qf.spoken, "recordings", lambda S: recs)
    args = types.SimpleNamespace(data_dir=tmp_path, n=3, model="m")
    wanted, extra = qf.q_first(args, dict(ready=False), tmp_path)  # audio_clean's two-order ready is false: one order decides
    assert wanted == [0, 1, 2] and extra["order"] == "q_audio" and extra["order_ready"]["ready"]
    clean_rows(ac, recs, schedule, wrong_task={("receiver", "q_audio", "animal")}).to_csv(tmp_path / "audio_clean.csv", index=False)
    with pytest.raises(SystemExit):
        qf.q_first(args, dict(ready=True), tmp_path)
    with pytest.raises(SystemExit):
        qf.sw.both_orders(args, dict(ready=False), tmp_path)
