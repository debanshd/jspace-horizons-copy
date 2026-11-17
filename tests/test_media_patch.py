"""media_patch's pieces on the CPU: the patch hook, the windows and donors, the summary's rules on made-up answers, and the
LaSOT window and crop rules."""

import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from jspace import batched, media

ROOT = Path(__file__).resolve().parents[1]


def load(module, folder):
    """a script from experiments/ or scripts/, with a stand-in for jspace.model where its dependencies (jlens) are missing"""
    sys.path.insert(0, str(ROOT / folder))
    stub = None
    try:
        importlib.import_module("jspace.model")
    except ImportError:
        stub = types.ModuleType("jspace.model")
        stub.Session = None
        sys.modules["jspace.model"] = stub
    try:
        return importlib.import_module(module)
    finally:
        sys.path.remove(str(ROOT / folder))
        if stub is not None:
            del sys.modules["jspace.model"]


def test_patch_rows_write_the_donor_state_at_their_layers_only():
    D = 3
    T = batched.Tables()
    T.donor["clip"] = {L: torch.arange(2 * 2 * D, dtype=torch.float16).reshape(2, 2, D) + 100 * (L + 1) for L in (0, 1)}  # 2 clips, 2 media tokens
    rows = [
        dict(input="clip", scene=0, site="patch", key="", layers=(0,), dose=0.0, ablate="", ko=(), donor=1),
        dict(input="clip", scene=1, site="", key="", layers=(), dose=0.0, ablate="", ko=(), donor=None),
    ]
    P = torch.tensor([1, 3])
    hk = dict(batched.residual_hooks(rows, P, T, []))
    assert set(hk) == {0}
    x = torch.zeros(2, 5, D)
    out = hk[0](None, None, x.clone())
    assert torch.equal(out[0, [1, 3]], T.donor["clip"][0][1].float())  # clip 1's state at layer 0, at the media positions
    assert torch.equal(out[0, [0, 2, 4]], torch.zeros(3, D)) and torch.equal(out[1], torch.zeros(5, D))


def test_windows_and_donors():
    ms = load("media_patch", "experiments")
    assert [w[0] for w in ms.windows(32)] == [-1, 3, 7, 11, 15, 19, 23, 27, 31]  # depth_sweep's window starts on the 4B
    assert ms.windows(32)[-1] == [31] and sum(len(w) for w in ms.windows(32)) == 33
    S = media.assign({w: [f"{w}{k}" for k in range(8)] for w in media.CLASSES}, 8)
    for r in S:
        assert S[ms.donor_scene(r, "Y")]["X"] == r["Y"] and S[ms.donor_scene(r, "Z")]["X"] == r["Z"]
        assert S[ms.donor_scene(r, "Y")]["scene"] // 8 == r["scene"] // 8  # the same position in its class


def made_up(r):
    """a model whose name reads the donor until window 19 and whose rule reads it until window 7, with no damage elsewhere"""
    follow = r["family"] == "all" or (r["family"] == "other" and int(r["set"]) <= (19 if r["task"] == "say" else 7))
    w = r["Y"] if follow else r["X"]
    if r["family"] == "same" and r["task"] == "say" and int(r["set"]) <= 19:
        w = r["Z"]
    return (w if r["task"] == "say" else media.truth("animal", w)), -0.1, -3.0


def test_summary_rules_on_made_up_answers():
    ms = load("media_patch", "experiments")
    S = media.assign({w: [f"{w}{k}" for k in range(4)] for w in media.CLASSES}, 4)
    ID = {w: [1] for w in media.CLASSES} | {"7": [2], "3": [3]}
    inputs = ["clip", "static", "stock", "stock_static"]
    rows = ms.patch_rows(inputs, S, ID, 32)
    R = ms.to_frame(rows, [made_up(r) for r in rows])
    out = ms.summarize(R, inputs, 32, "qwen", None)
    for inp in inputs:
        assert out["controls"][inp]["machinery_ok"] and out["controls"][inp]["positive_ok"]
        assert out["horizon"][inp]["L_say"] == 19 and out["horizon"][inp]["L_use"] == 7 and out["verdict"][inp] == "too late"
        assert out["share"][inp] == dict(say=23 / 32, use=11 / 32)
    assert out["predictions"]["clip"] == dict(P1="kept", P2="kept", P3="kept", P4="kept")
    assert out["predictions"]["stock"]["P5"] == "kept"
    kv = ms.summarize(R, inputs, 32, "gemma", 0.57)
    # Gemma's 0.38-0.48 widened by one window (4 / 32) holds 0.34; naming at 0.72 is past the sharing limit 0.57 + 0.125
    assert kv["predictions"]["clip"]["P2"] == "kept" and kv["predictions"]["clip"]["P3"] == "missed"


def test_failed_controls_leave_the_horizons_unread():
    ms = load("media_patch", "experiments")
    S = media.assign({w: [f"{w}{k}" for k in range(4)] for w in media.CLASSES}, 4)
    ID = {w: [1] for w in media.CLASSES} | {"7": [2], "3": [3]}
    rows = ms.patch_rows(["clip"], S, ID, 32)
    res = [made_up(r) if r["family"] != "self" else ("nonsense", 0.0, 0.0) for r in rows]
    out = ms.summarize(ms.to_frame(rows, res), ["clip"], 32, "qwen", None)
    assert out["verdict"]["clip"] == "patch machinery failed" and set(out["predictions"]["clip"].values()) == {"untestable"}


def test_lasot_window_and_crop():
    prep = load("prepare_lasot", "scripts")
    assert prep.parse_flags("0,0,1,0\n").tolist() == [False, False, True, False]
    assert prep.parse_boxes("1,2,3,4\n5\t6\t7\t8\n").tolist() == [[1, 2, 3, 4], [5, 6, 7, 8]]
    n = 900
    boxes = np.tile([100.0, 100.0, 80.0, 60.0], (n, 1))
    occ, oov = np.zeros(n, bool), np.zeros(n, bool)
    occ[prep.frame_offsets()[3]] = True  # the object is hidden in the first window
    start, idx, crop, ratio = prep.choose_window(boxes, occ, oov, 640, 360)
    assert start == prep.WINDOW_STEP and idx[0] == start + prep.frame_offsets()[0]
    left, top, side = crop
    assert side == pytest.approx(80 * 1.15) and 0 <= left and left + side <= 640 and 0 <= top and top + side <= 360
    assert ratio == pytest.approx(80 / side)
    assert prep.choose_window(boxes, np.ones(n, bool), oov, 640, 360) is None
    big = prep.crop_square(np.array([[0.0, 0.0, 600.0, 350.0]]), 640, 360)
    assert big[2] == 360 and big[1] == 0  # never larger than the shorter side, and inside the frame


def test_kv_limit():
    ms = load("media_patch", "experiments")
    cfg = types.SimpleNamespace(num_kv_shared_layers=18)
    lm = types.SimpleNamespace(final=41, hf=types.SimpleNamespace(config=types.SimpleNamespace(get_text_config=lambda: cfg)))
    assert ms.kv_limit(lm) == pytest.approx(24 / 42)  # Gemma-4 E4B: the REPORT's 0.57
    cfg.num_kv_shared_layers = 0
    assert ms.kv_limit(lm) is None
