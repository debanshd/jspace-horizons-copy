"""video_sweep's pieces on the CPU: the matched set's questions and controls, each model's own frame sampler, the canvas
frames, batching by prompt layout, patching from per-clip donors, and the patching summary on made-up answers."""

import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from jspace import batched, matched, stimuli

ROOT = Path(__file__).resolve().parents[1]


def load(module):
    sys.path.insert(0, str(ROOT / "experiments"))
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
        sys.path.remove(str(ROOT / "experiments"))
        if stub is not None:
            del sys.modules["jspace.model"]


def test_questions_as_depth_sweep_asks_them():
    assert matched.truth("say", "dog", []) == "dog" and matched.truth("letter", "dog", []) == "D"
    assert matched.truth("set", "dog", ["cat", "dog", "cup"]) == "7" and matched.truth("set", "bus", ["cat", "dog", "cup"]) == "3"
    assert matched.truth("ride", "horse", []) == "7" and matched.truth("animal", "chair", []) == "3"
    assert matched.matches("say", "A bicycle.", "bicycle") and matched.matches("letter", "d", "D") and matched.matches("animal", "7", "7")


def test_same_kind_word():
    for receiver, donor in (("pizza", "horse"), ("dog", "car"), ("horse", "chair"), ("airplane", "bird")):
        for sc in range(5):
            z = matched.same_kind_word(receiver, donor, sc)
            assert z not in (receiver, donor) and (z in matched.ANIMALS) == (receiver in matched.ANIMALS)
    assert matched.same_kind_word("pizza", "horse", 0) not in matched.RIDDEN


def qwen_and_gemma_samplers():
    pytest.importorskip("torchvision")
    from transformers.models.gemma4.video_processing_gemma4 import Gemma4VideoProcessor
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor

    q = Qwen3VLVideoProcessor(size={"longest_edge": 25165824, "shortest_edge": 4096}, patch_size=16, temporal_patch_size=2, merge_size=2, image_mean=[0.5] * 3, image_std=[0.5] * 3)
    g = Gemma4VideoProcessor(do_sample_frames=True, image_mean=[0.0] * 3, image_std=[1.0] * 3, max_soft_tokens=70, num_frames=32, patch_size=16, pooling_kernel_size=3)
    return q, g


def test_each_models_own_sampler():
    q, g = qwen_and_gemma_samplers()
    assert len(matched.native_indices(q, 240, 24.0)) == 20  # Qwen: 2 frames per second of a 10-second clip
    assert len(matched.native_indices(g, 240, 24.0)) == 32  # Gemma: 32 frames, whatever the length
    assert matched.scene_frame_count(g, 13, 400, 24.0) == 13  # a 13-frame receiver: Gemma's sampler cannot take 32
    assert matched.scene_frame_count(q, 1000, 90, 24.0) == 83  # the receiver's 2 per second, capped at the donor's 90 frames: Qwen gives 83
    for vp in (q, g):
        idx = matched.native_indices(vp, 90, 24.0, 16)
        assert len(idx) == 16 and min(idx) >= 0 and max(idx) < 90 and idx == sorted(idx)


def test_canvas_frames_put_the_clip_in_its_slot():
    from PIL import Image

    frames = matched.canvas_frames([Image.new("RGB", (320, 180), (255, 0, 0))] * 3, slot=5)
    assert frames.shape == (3, 2 * stimuli.SLOT, 3 * stimuli.SLOT, 3)
    red = (frames[0][..., 0] == 255) & (frames[0][..., 1] == 0)
    ys, xs = np.nonzero(red)
    assert xs.min() >= stimuli.SLOT and xs.max() < 2 * stimuli.SLOT and ys.min() >= stimuli.SLOT  # slot 5: row 2, column 2
    assert matched.input_of(frames, "frozen").shape == frames.shape and matched.input_of(frames, "still").size == (672, 448)


def test_qwen_processes_the_canvas_video():
    q, _ = qwen_and_gemma_samplers()
    from transformers.video_utils import VideoMetadata

    frames = np.zeros((20, 448, 672, 3), np.uint8)
    md = VideoMetadata(total_num_frames=240, fps=24.0, frames_indices=matched.native_indices(q, 240, 24.0), duration=10.0, width=672, height=448)
    o = q(videos=[frames], video_metadata=[md], do_sample_frames=False, cap_pixels_per_frame=False, return_tensors="pt")
    assert o["video_grid_thw"].tolist() == [[10, 28, 42]]  # 20 frames in 10 pairs, 448 x 672 in 16-pixel patches: 294 tokens a pair


def test_batches_group_by_prompt_layout():
    eng = types.SimpleNamespace(token={"moving": 9}, _layouts={})
    eng.encs = {
        ("moving", "say", 0): {"input_ids": torch.tensor([[1, 9, 9, 2]])},
        ("moving", "say", 1): {"input_ids": torch.tensor([[1, 9, 9, 9, 2]])},
        ("moving", "say", 2): {"input_ids": torch.tensor([[3, 9, 9, 4]])},
    }
    lay = {s: batched.Engine.layout(eng, dict(input="moving", task="say", scene=s)) for s in range(3)}
    assert lay[0] == lay[2] and lay[0] != lay[1]  # same length and media positions batch together; another frame count does not


def test_patch_from_per_clip_donors():
    T = batched.Tables()
    T.donor["moving"] = {0: {("donor", 3): torch.ones(2, 4, dtype=torch.float16), ("self", 3): torch.zeros(2, 4, dtype=torch.float16)}}
    rows = [dict(input="moving", scene=3, site="patch", key="", layers=(0,), dose=0.0, ablate="", ko=(), donor=("donor", 3))]
    (L, f), *_ = batched.residual_hooks(rows, torch.tensor([1, 2]), T, [])
    out = f(None, None, torch.full((1, 4, 4), 5.0))
    assert L == 0 and torch.equal(out[0, 1:3], torch.ones(2, 4)) and torch.equal(out[0, [0, 3]], torch.full((2, 4), 5.0))


def test_patch_summary_on_made_up_answers():
    vs = load("video_sweep")
    S = [
        dict(scene=i, X=x, Y=y, Z=z, clip=f"c{i}", slot=1 + i % 6, set=["cat", y, "cup"])
        for i, (x, y, z) in enumerate([("airplane", "bird", "train"), ("dog", "car", "cat"), ("pizza", "horse", "cup"), ("cat", "bus", "dog")] * 2)
    ]
    vs.SETS.update({r["scene"]: r["set"] for r in S})
    words = sorted({w for r in S for w in (r["X"], r["Y"], r["Z"])} | {"cat", "cup"})
    ID = {w: [1] for w in words} | {c: [2] for c in ["7", "3", *sorted({w[0].upper() for w in words})]}
    rows = vs.patch_rows("moving", S, ID, 32)

    def answer(r):
        follow = r["family"] == "full" or (r["family"] == "transplant" and int(r["set"]) <= (19 if r["task"] in ("say", "letter") else 7))
        w = r["Y"] if follow else r["X"]
        mem = vs.SETS[r["scene"]]
        return matched.truth(r["task"], w, mem), -0.1, -3.0

    R = vs.to_frame(rows, [answer(r) for r in rows])
    out = vs.patch_summary(R, ["moving"], 32, "qwen", None)
    assert out["controls"]["moving"]["machinery_ok"] and out["controls"]["moving"]["positive_ok"]
    assert out["cut"]["moving"]["L_say"] == 19 and out["cut"]["moving"]["L_use"] == 7 and out["verdict"]["moving"] == "too late"
    assert out["horizon"]["moving"]["letter"] == 19 and out["horizon"]["moving"]["set"] == 7
    assert out["predictions"]["moving"]["P1"] == "kept" and out["predictions"]["moving"]["P3"] == "kept"


def test_decoding_with_pyav(tmp_path):
    av = pytest.importorskip("av")
    path = tmp_path / "clip.mp4"
    c = av.open(str(path), "w")
    v = c.add_stream("libx264", rate=24)
    v.width, v.height, v.pix_fmt = 64, 48, "yuv420p"
    for i in range(30):
        for p in v.encode(av.VideoFrame.from_ndarray(np.full((48, 64, 3), i * 8, np.uint8), format="rgb24")):
            c.mux(p)
    for p in v.encode():
        c.mux(p)
    c.close()
    n, fps = matched.video_info(path)
    assert n == 30 and fps == 24.0
    ims = matched.decode(path, [0, 15, 15, 29])
    means = [int(np.asarray(im).mean()) for im in ims]
    assert len(ims) == 4 and means[1] == means[2] and abs(means[1] - 120) <= 4 and means[0] < means[1] < means[3]


def test_the_cache_check_never_batches_two_layouts():
    """the 4 clips of the check can differ in length: each layout group is generated on its own, results in row order"""
    seen = []

    def fake_batch(rows, n_new):
        lays = {eng.layout(r) for r in rows}
        assert len(lays) == 1, "a batch mixed prompt layouts"
        seen.append([r["scene"] for r in rows])
        return [(f"answer {r['scene']}", 0.0, 0.0, np.zeros(10)) for r in rows]

    eng = types.SimpleNamespace(cache_on={}, layout=lambda r: r["scene"] % 2, _generate_batch=fake_batch)
    eng._layout_groups = types.MethodType(batched.Engine._layout_groups, eng)
    eng._generate_grouped = types.MethodType(batched.Engine._generate_grouped, eng)
    rows = [dict(input="moving", scene=s) for s in range(4)]
    rep = batched.Engine.check_encoder_cache(eng, "moving", rows, 4)
    assert rep["on"] and rep["identical"] == 4 and seen[0] == [0, 2] and seen[1] == [1, 3]


def made_up_steering(r):
    """a model that follows Y when it is steered toward Y, and is otherwise correct"""
    w = r["Y"] if (r["family"] in ("steer", "sweep") and r["key"] == "Y") else r["X"]
    return matched.truth(r["task"], w, []), -0.1, -3.0


def run_summaries(n):
    vs, ms = load("video_sweep"), load("media_steer")
    from jspace import media as media_rules

    pairs = [("airplane", "bird", "train"), ("dog", "car", "cat"), ("pizza", "horse", "cup"), ("cat", "bus", "dog")]
    S = [dict(scene=i, X=x, Y=y, Z=z, clip=f"c{i}", slot=1 + i % 6, set=["cat", y, "cup"]) for i, (x, y, z) in ((i, pairs[i % 4]) for i in range(n))]
    vs.SETS.update({r["scene"]: r["set"] for r in S})
    words = sorted({w for r in S for w in (r["X"], r["Y"], r["Z"])} | {"cat", "cup"})
    ID = {w: [1] for w in words} | {c: [2] for c in ["7", "3", *sorted({w[0].upper() for w in words})]}
    n_layers, AL, inputs = 32, list(range(3, 32, 4)), ["moving", "frozen"]
    BL, starts = media_rules.blocks(n_layers), list(range(-1, n_layers, 4))
    rows = [dict(x, donor=None) for x in ms.phase_a(inputs, S, ID, BL, AL, AL) + ms.phase_b(inputs, S, ID, starts, n_layers, {i: 2.0 for i in inputs})]
    for inp in inputs:
        rows += vs.patch_rows(inp, S, ID, n_layers)
    res = [made_up_steering(r) if r["family"] not in ("full", "transplant") else (matched.truth(r["task"], r["Y"], vs.SETS[r["scene"]]), -0.1, -3.0) for r in rows]
    R = vs.to_frame(rows, res)
    return vs.summaries(R, S, inputs, {"patch", "steer"}, n_layers, AL, "qwen", None, BL, starts, {i: 2.0 for i in inputs})


def test_a_five_scene_smoke_check_summarizes_without_a_held_out_split():
    out = run_summaries(5)
    assert out["interventions"]["note"] == "smoke check: counts only, no verdict" and out["interventions"]["counts"]
    assert out["interventions_compare"] == "smoke check: counts only, no comparison"
    assert out["patch"]["controls"]["moving"]["machinery_ok"]


def test_a_full_run_compares_on_the_held_out_scenes():
    out = run_summaries(64)
    assert out["interventions_compare"]["moving vs frozen"]["verdict"] == "same"
    assert out["interventions"]["verdict"]["moving"]["middle"]["steers_naming"]
