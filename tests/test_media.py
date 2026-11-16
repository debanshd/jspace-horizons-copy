"""media_steer's pieces on the CPU: the class rules, the scoring, the scene assignment, the layer blocks, the row-wise
interventions, and the summary's rules on made-up answers."""

import importlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from jspace import batched, media
from jspace.runs import _json_default


def test_label_class():
    assert media.label_class("dog barking") == "dog"
    assert media.label_class("cat purring") == "cat"
    assert media.label_class("bird chirping, tweeting") == "bird"
    assert media.label_class("driving buses") == "bus"
    assert media.label_class("airplane flyby") == "airplane"
    assert media.label_class("railroad car, train wagon") is None  # two classes
    assert media.label_class("cattle, bovinae cowbell") is None  # 'cat' only as part of a word
    assert media.label_class("waterfall burbling") is None


def test_answer_scoring():
    assert media.answer_class("A plane.") == "airplane"
    assert media.answer_class("Dogs") == "dog"
    assert media.answer_class("It is a bus") == "bus"
    assert media.answer_class("nothing") is None
    assert media.answer_matches("say", "Puppy", "dog")
    assert media.answer_matches("animal", "7", "7") and not media.answer_matches("animal", "3", "7")
    assert media.answer_matches("animal", "**7**", "7")
    assert media.truth("animal", "horse") == "7" and media.truth("animal", "train") == "3" and media.truth("say", "cat") == "cat"


def test_assign_is_balanced_and_round_robin():
    by = {w: [f"{w}{k}" for k in range(8)] for w in media.CLASSES}
    S = media.assign(by, 8)
    assert [r["scene"] for r in S] == list(range(64))
    assert {r["X"] for r in S[:16]} == set(media.CLASSES) and all(sum(r["X"] == w for r in S[:16]) == 2 for w in media.CLASSES)
    for r in S:
        assert (r["Y"] in media.ANIMALS) != (r["X"] in media.ANIMALS)
        assert (r["Z"] in media.ANIMALS) == (r["X"] in media.ANIMALS) and r["Z"] != r["X"]
    pairs = {(r["X"], r["Y"]) for r in S}
    assert all((y, x) in pairs for x, y in pairs)  # every pair runs both ways
    for w in media.CLASSES:
        ys = [r["Y"] for r in S if r["X"] == w]
        assert all(ys.count(y) == 2 for y in set(ys))


def test_blocks_match_row_93_on_the_4b():
    b = media.blocks(32)
    assert (b["early"][0], b["early"][-1], b["middle"][0], b["middle"][-1], b["late"][0], b["late"][-1]) == (-1, 10, 11, 22, 23, 31)
    for n in (24, 32, 35, 42, 60, 64):
        b = media.blocks(n)
        assert b["early"] + b["middle"] + b["late"] == list(range(-1, n))
    assert media.sweep_starts(32) == [-1, 3, 7, 11, 15, 19, 23, 27, 31]


def test_frame_metadata_gives_the_true_times():
    m = media.frame_metadata()
    assert [i / m["fps"] for i in m["frames_indices"]] == pytest.approx(media.FRAME_TIMES)


def test_collate():
    a = {"input_ids": torch.tensor([[1, 2, 3]]), "pix": torch.ones(4, 2), "n": [4]}
    b = {"input_ids": torch.tensor([[1, 2, 3]]), "pix": torch.zeros(4, 2), "n": [4]}
    c = batched.collate([a, b])
    assert c["input_ids"].shape == (2, 3) and c["pix"].shape == (8, 2) and c["n"] == [4, 4]


MEDIA_TOKEN, D = 9, 4


def toy_tables():
    T = batched.Tables()
    e = torch.eye(D)
    T.dirs["Y"] = {L: e[[0, 1]] for L in (0, 1)}  # scene 0 steers along e0, scene 1 along e1
    T.dirs["random"] = {L: e[[2, 2]] for L in (0, 1)}
    T.sigma["clip"] = {0: 1.0, 1: 1.0}
    T.mu["clip"] = {L: torch.zeros(D) for L in (0, 1)}
    T.abl["own"] = ("scene", {L: e[[0, 1]] for L in (0, 1)})
    T.abl["part"] = ("shared", {L: e[:, :2] for L in (0, 1)})
    return T


def row(scene, **kw):
    r = dict(input="clip", scene=scene, site="", key="", layers=(), dose=0.0, ablate="", ko=())
    r.update(kw)
    return r


def run_hooks(rows, layer, x, read_layers=(1,)):
    P = torch.tensor([1, 2])  # media positions
    hk = dict(batched.residual_hooks(rows, P, toy_tables(), list(read_layers)))
    return hk[layer](None, None, x.clone()) if layer in hk else x


def test_steering_edits_only_its_rows_layers_and_positions():
    x = torch.zeros(3, 4, D)
    rows = [row(0, site="media", key="Y", layers=(0,), dose=2.0), row(1, site="last", key="Y", layers=(0,), dose=1.0), row(0, site="media", key="random", layers=(1,), dose=1.0)]
    out = run_hooks(rows, 0, x)
    assert torch.equal(out[0, 1], torch.tensor([2.0, 0, 0, 0])) and torch.equal(out[0, 2], out[0, 1])
    assert torch.equal(out[0, 0], torch.zeros(D)) and torch.equal(out[0, 3], torch.zeros(D))  # text positions untouched
    assert torch.equal(out[1, 3], torch.tensor([0, 1.0, 0, 0])) and torch.equal(out[1, :3], torch.zeros(3, D))  # last token only
    assert torch.equal(out[2], torch.zeros(4, D))  # its layer set does not include layer 0
    out1 = run_hooks(rows, 1, x)
    assert torch.equal(out1[2, 1], torch.tensor([0, 0, 1.0, 0])) and torch.equal(out1[0], torch.zeros(4, D))


def test_generated_tokens_pass_through():
    rows = [row(0, site="media", key="Y", layers=(0,), dose=2.0)]
    x = torch.zeros(1, 1, D)
    assert torch.equal(run_hooks(rows, 0, x), x)


def test_ablation_removes_a_direction_or_a_subspace_at_the_read_layers():
    x = torch.ones(2, 4, D)
    rows = [row(0, ablate="own"), row(1, ablate="part")]
    out = run_hooks(rows, 1, x)
    assert torch.equal(out[0, 1], torch.tensor([0, 1.0, 1, 1])) and torch.equal(out[0, 0], torch.ones(D))
    assert torch.equal(out[1, 2], torch.tensor([0, 0, 1.0, 1])) and torch.equal(out[1, 3], torch.ones(D))
    assert torch.equal(run_hooks(rows, 0, x, read_layers=(1,)), x)  # not a read layer: no hook


def test_knockout_mask_blocks_text_to_media_only_for_blocked_rows():
    ids = torch.tensor([5, MEDIA_TOKEN, 6, MEDIA_TOKEN, 7])
    KO = batched.knockout_mask(ids, MEDIA_TOKEN, torch.tensor([True, False]))
    assert KO.shape == (2, 1, 5, 5)
    media_pos, text_pos = [1, 3], [0, 2, 4]
    assert all(KO[0, 0, q, k] for q in text_pos for k in media_pos)
    assert not KO[0, 0][:, text_pos].any() and not KO[0, 0][media_pos].any()  # media read each other; text reads text
    assert not KO[1].any()


def test_assert_unpadded():
    enc = {"input_ids": torch.tensor([[1, MEDIA_TOKEN, 2], [3, MEDIA_TOKEN, 4]]), "attention_mask": torch.ones(2, 3, dtype=torch.long)}
    batched.assert_unpadded(enc, MEDIA_TOKEN)
    with pytest.raises(AssertionError):
        batched.assert_unpadded(dict(enc, attention_mask=torch.tensor([[0, 1, 1], [1, 1, 1]])), MEDIA_TOKEN)


# ---- the summary's rules on made-up answers


def load_experiment():
    """experiments/media_steer.py, with a stand-in for jspace.model where its dependencies (jlens) are not installed"""
    path = Path(__file__).resolve().parents[1] / "experiments"
    sys.path.insert(0, str(path))
    stub = None
    try:
        importlib.import_module("jspace.model")
    except ImportError:
        stub = types.ModuleType("jspace.model")
        stub.Session = None
        sys.modules["jspace.model"] = stub
    try:
        return importlib.import_module("media_steer")
    finally:
        sys.path.remove(str(path))
        if stub is not None:
            del sys.modules["jspace.model"]


def answer(r):
    """a model that follows Y when Y's J-lens direction is written on the media tokens in the early or middle block (and through
    a sweep that reaches layer 22), applies the rule to Y from the early block only, and otherwise answers for X"""
    reaches = r["family"] == "sweep" and r["key"] == "Y" and (min(r["layers"]) <= 22 if r["set"].startswith("suffix") else max(r["layers"]) >= 3)
    in_block = r["family"] == "steer" and r["site"] == "media" and r["key"] == "Y" and r["set"] in ("early", "middle") and r["dose"] >= 2
    w = r["Y"] if (in_block or reaches) and (r["task"] == "say" or r["set"] == "early" or r["family"] == "sweep") else r["X"]
    text = w if r["task"] == "say" else media.truth("animal", w)
    blocked_late = r["family"] == "knockout" and r["ko_kind"] == "from" and r["ko_L"] <= 23
    lx = -0.1 if not blocked_late else -3.0
    return text, lx, -4.0


def test_summary_rules_on_made_up_answers():
    ms = load_experiment()
    by = {w: [f"{w}{k}" for k in range(3)] for w in media.CLASSES}
    S = media.assign(by, 3)
    ID = {w: [1] for w in media.CLASSES} | {"7": [2], "3": [3]}
    n_layers = 32
    BL, AL = media.blocks(n_layers), list(range(3, 32, 4))
    starts = media.sweep_starts(n_layers)
    inputs = ["clip", "static", "sound"]
    rows = ms.phase_a(inputs, S, ID, BL, AL, AL)
    R = ms.to_frame(rows, [answer(r) for r in rows], "A")
    rows_b = ms.phase_b(inputs, S, ID, starts, n_layers, {"clip": 2.0, "static": 2.0, "sound": 2.0})
    R = pd.concat([R, ms.to_frame(rows_b, [answer(r) for r in rows_b], "B")], ignore_index=True)
    out = ms.summarize(R, S, inputs, BL, starts, AL, AL, n_layers, n_cal=8, smoke=False)
    assert out["gate"]["clip"]["say"]["passes"] and out["dose"]["clip"]["calibrated"] == 2.0
    V = out["verdict"]["clip"]
    assert V["early"]["steers_naming"] and V["middle"]["steers_naming"] and not V["late"]["steers_naming"]
    assert V["early"]["J_specific"] and V["early"]["rule"] == "seen" and V["middle"]["rule"] == "only said"
    assert V["output push"].startswith("media-specific")
    assert out["sweeps"]["clip"]["suffix edge"] == 19 and out["sweeps"]["clip"]["prefix edge"] == 3
    assert out["knockout"]["clip"]["say"]["last_reader"] == 23
    assert out["compare"]["sound vs clip"]["verdict"] == "same as the clip"
    assert out["compare"]["clip vs static"]["verdict"] == "nothing temporal tested"
    P = out["predictions"]["clip"]
    assert P["1"] == "kept" and P["2"] == "kept" and P["3"] == "kept" and P["5"] == "kept" and P["6"] == "kept" and P["7"] == "kept"
    assert out["predictions"]["sound"]["8"] == "kept"
    assert P["4"] == "missed"  # the made-up rule follows Y on every early scene, which prediction 4 says it will not
    ab = out["ablation"]["clip"]["say"]
    assert ab["R1"] == "not needed" and ab["R1_nats"] == "not needed"  # removing either name changes nothing in the made-up model
    json.dumps(out, default=_json_default)  # the summary can be saved
    smoke = ms.summarize(R, S, inputs, BL, starts, AL, AL, n_layers, n_cal=8, smoke=True)
    assert "verdict" not in smoke and smoke["counts"]


def test_prediction_5_wants_the_edge_at_or_just_before_the_last_read():
    ms = load_experiment()
    assert ms.suffix_edge_ok(19, 23) and ms.suffix_edge_ok(23, 23)
    assert not ms.suffix_edge_ok(27, 23)  # after the last read: the looser version 3 code passed this
    assert not ms.suffix_edge_ok(15, 23) and not ms.suffix_edge_ok(None, 23)


def test_name_steering_r1():
    ms = load_experiment()
    assert ms.r1((0.30, 0.10, 0.50)) == "necessary" and ms.r1((0.05, -0.05, 0.12)) == "not needed" and ms.r1((0.20, 0.0, 0.4)) == "undetermined"
    assert ms.r1_nats((-0.8, -1.2, -0.3)) == "necessary" and ms.r1_nats((0.0, -0.1, 0.1)) == "not needed" and ms.r1_nats((-0.4, -0.9, 0.1)) == "undetermined"


def test_decoded_clips_are_checked_by_their_arrays(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    try:
        prep = importlib.import_module("prepare_media")
    finally:
        sys.path.pop(0)
    frames, audio = np.zeros((8, 224, 224, 3), np.uint8), np.ones(160000, np.float32)
    np.savez(tmp_path / "c1.npz", frames=frames, audio=audio)
    man = dict(clips=[dict(id="c1", frames_sha256=prep.array_sha256(frames), audio_sha256=prep.array_sha256(audio)), dict(id="c2", frames_sha256="", audio_sha256="")])
    assert prep.check_decoded(man, tmp_path) == [("c2", "missing")]
    np.savez(tmp_path / "c1.npz", frames=frames, audio=audio * 2)
    assert prep.check_decoded(man, tmp_path)[0] == ("c1", "decoded arrays differ")


def test_no_dataframe_method_shadows_a_column():
    """`df.clip` is DataFrame.clip(), not the clip column: `G.clip == c` is False and `G[False]` raised on the first GPU smoke
    run. Columns whose names are DataFrame attributes must be read as df["name"]."""
    import ast

    shadowed = {a for a in ("clip", "cls", "correct", "generation", "input", "task", "scene", "arm") if hasattr(pd.DataFrame, a)}
    root = Path(__file__).resolve().parents[1]
    bad = []
    for f in [root / "experiments/media_steer.py", root / "scripts/compare_media.py", root / "src/jspace/media.py", root / "src/jspace/batched.py"]:
        for node in ast.walk(ast.parse(f.read_text())):
            if (
                isinstance(node, ast.Attribute)
                and node.attr in shadowed
                and (isinstance(node.value, ast.Subscript) or (isinstance(node.value, ast.Name) and node.value.id not in ("media", "np", "torch", "batched", "self", "args")))
            ):
                bad.append(f"{f.name}:{node.lineno} {node.value.id}.{node.attr}")
    assert not bad, bad


def test_encoder_kwargs_follow_the_signature():
    """Gemma-4's get_video_features needs num_frames_per_video; the arguments come from the batch by the encoder's signature, as
    generate() takes them, and keys the encoder does not take stay out"""

    def get_video_features(pixel_values_videos, video_position_ids=None, num_frames_per_video=None, **kwargs):
        return pixel_values_videos, num_frames_per_video

    enc = dict(input_ids=1, attention_mask=2, pixel_values_videos=3, video_position_ids=4, num_frames_per_video=5)
    assert batched.encoder_kwargs(get_video_features, enc) == dict(pixel_values_videos=3, video_position_ids=4, num_frames_per_video=5)

    def get_qwen_video_features(pixel_values_videos, video_grid_thw=None, **kwargs):
        return pixel_values_videos

    assert batched.encoder_kwargs(get_qwen_video_features, dict(pixel_values_videos=3, video_grid_thw=6, input_ids=1)) == dict(pixel_values_videos=3, video_grid_thw=6)
