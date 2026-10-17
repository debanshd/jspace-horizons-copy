"""What the J-lens reads, and the directions and subspaces it defines.

The J-lens at layer L scores a residual h by W_U J_L h (W_U the unembedding, J_L the fitted Jacobian). From it:
- ranks: the rank of a word among all tokens, through the J-lens, the plain logit lens (W_U h) or the J++ lens
- the lens subspace: the top-r right singular directions of W_U J_L (the r directions the lens reads most strongly)
- a word's lens direction: J_L^T W_U[t], unit length (adding it pushes the lens, and often the model, toward t)
"""

import numpy as np
import torch

TOP = 10  # a word is "readable" when it is in the lens top 10


@torch.no_grad()
def ranks(lm, H, id_sets, jac=True):
    """Rank (1 = top) of the best variant of each id set at every read layer, [n, len(layers), len(id_sets)], from {layer: [n, d]}.
    jac=True reads through the J-lens, "jpp" through the J++ lens, False through the plain logit lens."""
    M = lm.jpp if jac == "jpp" else lm.jac
    n = np.asarray(H[lm.layers[0]]).shape[0]
    m = max(len(t) for t in id_sets)
    Sx = torch.as_tensor([list(t) + [t[0]] * (m - len(t)) for t in id_sets], device=lm.device)
    out = np.zeros((n, len(lm.layers), len(id_sets)), int)
    for li, L in enumerate(lm.layers):
        for s in range(0, n, lm.batch):
            h = torch.as_tensor(np.asarray(H[L][s : s + lm.batch]), device=lm.device, dtype=torch.float32)
            x = lm.model.unembed(h @ M[L].T if jac else h).float()
            for q in range(len(id_sets)):
                v = x[:, Sx[q]]
                out[s : s + lm.batch, li, q] = ((x[:, None, :] > v[:, :, None]).sum(-1) + 1).min(1).values.cpu().numpy()
    return out


@torch.no_grad()
def ranks_at(lm, texts, images, where, n_pos, id_sets, chunk):
    """J-lens, logit-lens and J++ ranks of each id set at the n_pos positions `where` names plus the last token, every read layer:
    three int arrays [n, len(layers), n_pos + 1, len(id_sets)] (J++ is -1 without a J++ lens), and the last-token logits [n, vocab]."""
    out_j, out_l, out_p, logits = [], [], [], []
    for s in range(0, len(texts), chunk):
        tx = texts[s : s + chunk]
        im = images[s : s + chunk] if images is not None else None
        lg, H = lm.forward(tx, im, lm.layers, where=where)
        n = lg.shape[0]
        assert H[lm.layers[0]].shape[1] == n_pos + 1, (H[lm.layers[0]].shape, n_pos + 1)
        flat = {L: H[L].reshape(n * (n_pos + 1), -1).numpy() for L in lm.layers}
        shape = (n, len(lm.layers), n_pos + 1, len(id_sets))
        rj = ranks(lm, flat, id_sets, jac=True).reshape(shape)
        rl = ranks(lm, flat, id_sets, jac=False).reshape(shape)
        rp = ranks(lm, flat, id_sets, jac="jpp").reshape(shape) if lm.jpp is not None else rl * 0 - 1
        out_j.append(rj)
        out_l.append(rl)
        out_p.append(rp)
        logits.append(lg)
        del H
    return np.concatenate(out_j), np.concatenate(out_l), np.concatenate(out_p), torch.cat(logits)


def where_image(lm, n_image):
    """`where` for Session.forward: a row's image tokens (the n_image tokens before <|vision_end|>), then its last token"""

    def f(ids):
        ve = ids.index(lm.vision_end_id)
        return list(range(ve - n_image, ve)) + [len(ids) - 1]

    return f


def readable(rank):
    """rank in the lens top 10 (rank 0 or -1 means not read)"""
    return (rank >= 1) & (rank <= TOP)


# ---- subspaces


def orth(V):
    """orthonormal basis [d, r] of the rows of V [r, d] (QR, in order)"""
    Qm, _ = torch.linalg.qr(torch.as_tensor(np.asarray(V), dtype=torch.float32).T)
    err = float((Qm.T @ Qm - torch.eye(Qm.shape[1])).abs().max())
    assert err < 1e-3, f"projector is not orthogonal: {err}"
    return Qm.numpy()


@torch.no_grad()
def _gram(lm):
    """W_U^T W_U [d, d], accumulated in chunks of the vocabulary and kept on the session"""
    if "G" not in lm.cache:
        U = lm.unembed_weight
        G = torch.zeros(U.shape[1], U.shape[1], device=lm.device)
        for s in range(0, U.shape[0], 8192):
            w = U[s : s + 8192].float().to(lm.device)
            G += w.T @ w
        lm.cache["G"] = G
    return lm.cache["G"]


@torch.no_grad()
def lens_subspace(lm, L, r):
    """Q [d, r]: the top-r right singular directions of W_U J_L, strongest first, so Q[:, :k] is the top-k for every k <= r"""
    G = _gram(lm)
    if L not in lm.cache:
        M = lm.jac[L].float().T @ G @ lm.jac[L].float()
        lm.cache[L] = torch.linalg.eigh(M)[1].cpu().numpy()
    return orth(np.ascontiguousarray(lm.cache[L][:, -r:][:, ::-1].T))


@torch.no_grad()
def jpp_subspace(lm, L, r):
    """the J++ analogue of lens_subspace"""
    G = _gram(lm)
    M = lm.jpp[L].float().T @ G @ lm.jpp[L].float()
    return orth(np.ascontiguousarray(torch.linalg.eigh(M)[1].cpu().numpy()[:, -r:][:, ::-1].T))


@torch.no_grad()
def unembed_subspace(lm, r):
    """Q [d, r]: the top-r right singular directions of W_U alone, the same at every layer (the output-sensitivity control)"""
    if "U" not in lm.cache:
        lm.cache["U"] = torch.linalg.eigh(_gram(lm))[1].cpu().numpy()
    return orth(np.ascontiguousarray(lm.cache["U"][:, -r:][:, ::-1].T))


def random_subspace(d, r, seed):
    """a random r-dimensional subspace of R^d, [d, r]"""
    return orth(np.random.default_rng(seed).normal(0, 1, (r, d)))


@torch.no_grad()
def assert_descending(lm, L, Q, maps=None):
    """guard for slicing Q[:, :k] out of a larger basis: the readout gain q^T (J^T W_U^T W_U J) q must not rise along Q's columns"""
    Jm = (lm.jac if maps is None else maps)[L].float()
    Qd = torch.as_tensor(Q, dtype=torch.float32, device=lm.device)
    g = ((_gram(lm) @ (Jm @ Qd)) * (Jm @ Qd)).sum(0)
    assert g[0] >= g[-1] and g[: len(g) // 2].mean() >= g[len(g) // 2 :].mean(), f"subspace columns are not in descending readout gain at layer {L}"


# ---- directions


@torch.no_grad()
def word_directions(lm, L, token_ids, kind="J"):
    """unit token directions at layer L, [n, d, 1]: "J" is J_L^T W_U[t] (the word's lens direction), "J++" the J++ analogue,
    "W_U" the unembedding row alone (no Jacobian, the same at every layer)"""
    W = lm.unembed_weight[token_ids].float()
    V = W if kind == "W_U" else (lm.jac[L].float().T @ W.T).T if kind == "J" else (lm.jpp[L].float().T @ W.T).T
    V = V / V.norm(dim=1, keepdim=True).clamp_min(1e-6)
    return V.cpu().numpy()[:, :, None]


def unit_rows(n, d, seed):
    """n random unit directions in R^d, [n, d, 1]"""
    V = np.random.default_rng(seed).normal(0, 1, (n, d)).astype(np.float32)
    return (V / np.linalg.norm(V, axis=1, keepdims=True))[:, :, None]
