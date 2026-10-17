"""The lens subspaces and directions: orthonormal, strongest first, deterministic."""

from types import SimpleNamespace

import numpy as np
import torch

from jspace import lens


def toy_session(d=8, vocab=50, seed=0):
    g = torch.Generator().manual_seed(seed)
    return SimpleNamespace(unembed_weight=torch.randn(vocab, d, generator=g), jac={3: torch.randn(d, d, generator=g)}, jpp=None, device="cpu", cache={})


def test_orth_is_orthonormal():
    Q = lens.orth(np.random.default_rng(0).normal(size=(4, 10)))
    assert Q.shape == (10, 4)
    assert np.allclose(Q.T @ Q, np.eye(4), atol=1e-5)


def test_random_subspace_is_fixed_by_its_seed():
    assert np.array_equal(lens.random_subspace(16, 3, 7), lens.random_subspace(16, 3, 7))
    assert not np.allclose(lens.random_subspace(16, 3, 7), lens.random_subspace(16, 3, 8))


def test_lens_subspace_is_the_top_singular_directions_of_WU_J():
    lm = toy_session()
    Q = lens.lens_subspace(lm, 3, 3)
    M = (lm.unembed_weight @ lm.jac[3]).numpy()  # W_U J
    _, _, Vt = np.linalg.svd(M)
    top = Vt[:3].T  # the right singular directions, strongest first
    assert np.allclose(np.abs(np.sum(Q * top, 0)), 1, atol=1e-3)  # the same directions, up to sign
    lens.assert_descending(lm, 3, Q)


def test_a_slice_of_a_larger_basis_is_the_smaller_basis():
    lm = toy_session()
    big, small = lens.lens_subspace(lm, 3, 5), lens.lens_subspace(lm, 3, 2)
    assert np.allclose(np.abs(np.sum(big[:, :2] * small, 0)), 1, atol=1e-4)


def test_word_directions_are_unit_and_differ_by_kind():
    lm = toy_session()
    J = lens.word_directions(lm, 3, [1, 2], "J")
    W = lens.word_directions(lm, 3, [1, 2], "W_U")
    assert J.shape == (2, 8, 1)
    assert np.allclose(np.linalg.norm(J[:, :, 0], axis=1), 1, atol=1e-5)
    assert np.allclose(W[:, :, 0], (lm.unembed_weight[[1, 2]] / lm.unembed_weight[[1, 2]].norm(dim=1, keepdim=True)).numpy(), atol=1e-6)
    assert not np.allclose(J, W)


def test_unit_rows():
    U = lens.unit_rows(5, 7, 0)
    assert U.shape == (5, 7, 1) and np.allclose(np.linalg.norm(U[:, :, 0], axis=1), 1, atol=1e-5)


def test_readable_is_the_top_10():
    assert lens.readable(np.array([0, 1, 10, 11, -1])).tolist() == [False, True, True, False, False]
