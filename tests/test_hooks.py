"""The interventions edit exactly the positions and the subspace they name, on the prompt pass only."""

import numpy as np
import pytest
import torch

from jspace import hooks, lens

IMG = 7  # the image token id in these toy batches
D = 6


def batch():
    """two left-padded rows; row 0 has image tokens at 2-3 of its own tokens, row 1 at 0-1"""
    ids = torch.tensor([[0, 0, 5, 5, IMG, IMG, 5], [5, 5, IMG, IMG, 5, 5, 5]])
    am = torch.tensor([[0, 0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1]])
    return {"input_ids": ids, "attention_mask": am}


def apply(builder, h, idx=(0, 1)):
    """run every hook a builder makes (one layer here) on the residual h, as a forward hook would"""
    (_, fn), *_ = builder(batch(), np.array(idx))
    return fn(None, None, h.clone())


@pytest.fixture
def h():
    return torch.randn(2, 7, D, generator=torch.Generator().manual_seed(0))


def basis(r=2, seed=1):
    return lens.random_subspace(D, r, seed)


def test_image_positions():
    pos = hooks.image_positions(IMG)
    assert pos([5, IMG, IMG, 5], 0) == [1, 2]


def test_ablate_subspace_removes_the_part_at_image_tokens_only(h):
    Q = basis()
    out = apply(hooks.ablate_subspace({0: Q}, hooks.image_positions(IMG), [0]), h)
    Qt = torch.as_tensor(Q)
    img = batch()["input_ids"] == IMG
    assert torch.allclose(out[img] @ Qt, torch.zeros(int(img.sum()), 2), atol=1e-5)
    assert torch.equal(out[~img], h[~img])
    # the rest of the state is kept
    assert torch.allclose(out[img] - (out[img] @ Qt) @ Qt.T, h[img] - (h[img] @ Qt) @ Qt.T, atol=1e-5)


def test_complement_keeps_only_the_part(h):
    Q = basis()
    out = apply(hooks.ablate_subspace({0: Q}, hooks.image_positions(IMG), [0], part="complement"), h)
    Qt = torch.as_tensor(Q)
    img = batch()["input_ids"] == IMG
    assert torch.allclose(out[img], (h[img] @ Qt) @ Qt.T, atol=1e-5)


def test_mean_ablation_puts_the_mean_part_back(h):
    Q, mu = basis(), np.arange(D, dtype=np.float32)
    out = apply(hooks.ablate_subspace({0: Q}, hooks.image_positions(IMG), [0], mu={0: mu}), h)
    Qt = torch.as_tensor(Q)
    img = batch()["input_ids"] == IMG
    assert torch.allclose(out[img] @ Qt, (torch.as_tensor(mu) @ Qt).expand(int(img.sum()), -1), atol=1e-5)


def test_ablate_directions_uses_each_prompts_own_direction(h):
    q = lens.unit_rows(2, D, 3)  # [2, d, 1]: one direction per prompt
    out = apply(hooks.ablate_directions({0: q}, hooks.image_positions(IMG), [0]), h)
    ids = batch()["input_ids"]
    for i in range(2):
        v = out[i][ids[i] == IMG] @ torch.as_tensor(q[i, :, 0])
        assert torch.allclose(v, torch.zeros_like(v), atol=1e-5)


def test_add_directions_adds_scale_times_the_direction(h):
    q = lens.unit_rows(2, D, 4)
    out = apply(hooks.add_directions({0: q}, hooks.image_positions(IMG), [0], {0: 3.0}), h)
    ids = batch()["input_ids"]
    for i in range(2):
        assert torch.allclose(out[i][ids[i] == IMG], h[i][ids[i] == IMG] + 3.0 * torch.as_tensor(q[i, :, 0]), atol=1e-5)
        assert torch.equal(out[i][ids[i] != IMG], h[i][ids[i] != IMG])


def test_patch_writes_the_donor_state_and_its_parts(h):
    rec = {0: np.random.default_rng(5).normal(size=(2, 2, D)).astype(np.float16)}
    pos = [lambda s: np.array([s - 3, s - 2]), lambda s: np.array([s - 5, s - 4])]  # the image tokens of each row
    full = apply(hooks.patch([0], rec, pos), h)
    assert torch.allclose(full[0, 4:6], torch.as_tensor(rec[0][0], dtype=torch.float32))
    assert torch.allclose(full[1, 2:4], torch.as_tensor(rec[0][1], dtype=torch.float32))
    Q = basis()
    Qt = torch.as_tensor(Q)
    part = apply(hooks.patch([0], rec, pos, {0: Q}, "lens"), h)
    donor = torch.as_tensor(rec[0][0], dtype=torch.float32)
    assert torch.allclose(part[0, 4:6] @ Qt, donor @ Qt, atol=1e-3)  # the Q part is the donor's
    assert torch.allclose(part[0, 4:6] - (part[0, 4:6] @ Qt) @ Qt.T, h[0, 4:6] - (h[0, 4:6] @ Qt) @ Qt.T, atol=1e-3)  # the rest is the receiver's


def test_generated_tokens_pass_through(h):
    """a one-position pass (a generated token) is not edited: the prompt's edit lives on in the KV cache"""
    builder = hooks.add_directions({0: lens.unit_rows(2, D, 6)}, hooks.image_positions(IMG), [0], {0: 5.0})
    (_, fn), *_ = builder(batch(), np.array([0, 1]))
    step = h[:, :1].clone()
    assert torch.equal(fn(None, None, step), step)


def test_tuple_outputs_keep_their_other_members(h):
    builder = hooks.add_directions({0: lens.unit_rows(2, D, 6)}, hooks.image_positions(IMG), [0], {0: 1.0})
    (_, fn), *_ = builder(batch(), np.array([0, 1]))
    out = fn(None, None, (h.clone(), "cache"))
    assert isinstance(out, tuple) and out[1] == "cache"


def test_combine_registers_both():
    a = hooks.add_directions({0: lens.unit_rows(2, D, 1)}, hooks.image_positions(IMG), [0], {0: 1.0})
    b = hooks.add_directions({1: lens.unit_rows(2, D, 2)}, hooks.image_positions(IMG), [1], {1: 1.0})
    assert [L for L, _ in hooks.combine(a, b)(batch(), np.array([0, 1]))] == [0, 1]
