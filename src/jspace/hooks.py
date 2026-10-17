"""Interventions on the residual stream: patching, ablation and steering at chosen positions and layers.

Every builder returns `hooks(enc, idx) -> [(layer, fn)]`: for a batch encoded as `enc` whose rows are the prompts `idx`, the
forward hooks to register (Session.forward and Session.generate register and remove them). Layer -1 is the input to layer 0.
An edit applies on the prompt pass only; a generated token (a one-position pass) goes through unchanged, so what was written
into the prompt stays in the KV cache for the whole answer.

Positions: `pos_of(row_ids, prompt_index) -> [positions]` names positions in a row without its left padding;
`image_positions(token_id)` gives every image token.
"""

import numpy as np
import torch


def residual_of(out):
    return out[0] if isinstance(out, tuple) else out


def _residual_hook(edit):
    """a forward hook that hands the residual, in float32, to edit(x) (which changes x in place), then casts back"""

    def fn(mod, inp, out):
        h = residual_of(out)
        if h.shape[1] == 1:
            return out
        x = h.float()
        edit(x)
        x = x.to(h.dtype)
        return (x,) + tuple(out[1:]) if isinstance(out, tuple) else x

    return fn


def image_positions(token_id):
    """pos_of for every image token of a row"""
    return lambda row, i: [j for j, t in enumerate(row) if t == token_id]


def _rows_and_positions(enc, idx, pos_of):
    """(rows, Pm) index tensors [b, P] for x[rows, Pm]: pos_of's positions shifted past each row's left padding; a row with fewer
    positions repeats its last one"""
    ids, am = enc["input_ids"], enc["attention_mask"]
    b, s = ids.shape
    rows_pos = []
    for i in range(b):
        n = int(am[i].sum())
        rows_pos.append([s - n + p for p in pos_of(ids[i, s - n :].tolist(), int(idx[i]))])
    P = max(len(r) for r in rows_pos)
    Pm = torch.as_tensor([r + [r[-1]] * (P - len(r)) for r in rows_pos], device=ids.device)
    rows = torch.arange(b, device=ids.device)[:, None].expand(-1, P)
    return rows, Pm


def patch(layers, rec, pos, Q=None, part="all"):
    """Residual-stream patching. Prompt i receives rec[L][i] (fp16 [n, P, d]) at the absolute positions pos[i](row_length) of the
    padded row, at every layer in `layers`. With Q (per layer, [d, r] orthonormal): part="lens" patches only the Q part of the
    state and keeps the receiver's rest; part="complement" patches the rest and keeps the receiver's Q part."""

    def hooks(enc, idx):
        dev = enc["input_ids"].device
        b, s = enc["input_ids"].shape
        idx = np.asarray(idx)
        Pm = torch.as_tensor(np.stack([pos[i](s) for i in idx]), device=dev)
        rows = torch.arange(b, device=dev)[:, None].expand(-1, Pm.shape[1])
        out = []
        for L in layers:
            D = torch.as_tensor(np.asarray(rec[L])[idx], device=dev, dtype=torch.float32)
            Qd = None if Q is None else torch.as_tensor(Q[L], device=dev, dtype=torch.float32)

            def edit(x, D=D, Qd=Qd):
                v = x[rows, Pm]
                assert v.shape == D.shape, (v.shape, D.shape)
                if part == "all":
                    v = D
                elif part == "lens":
                    v = v - (v @ Qd) @ Qd.T + (D @ Qd) @ Qd.T
                else:
                    v = (v @ Qd) @ Qd.T + (D - (D @ Qd) @ Qd.T)
                x[rows, Pm] = v

            out.append((L, _residual_hook(edit)))
        return out

    return hooks


def ablate_subspace(Q, pos_of, layers, part="lens", mu=None):
    """Subspace ablation with one basis for every prompt (Q[L], [d, r] orthonormal): part="lens" removes the Q part,
    part="complement" keeps only the Q part. With mu (per layer, [d]) the removed part is replaced by mu's part (mean-ablation)."""

    def hooks(enc, idx):
        rows, Pm = _rows_and_positions(enc, idx, pos_of)
        dev = Pm.device
        out = []
        for L in layers:
            Qd = torch.as_tensor(Q[L], device=dev, dtype=torch.float32)
            Md = None if mu is None else (torch.as_tensor(mu[L], device=dev, dtype=torch.float32) @ Qd) @ Qd.T

            def edit(x, Qd=Qd, Md=Md):
                v = x[rows, Pm]
                v = (v - (v @ Qd) @ Qd.T) if part == "lens" else (v @ Qd) @ Qd.T
                if Md is not None:
                    v = v + Md
                x[rows, Pm] = v

            out.append((L, _residual_hook(edit)))
        return out

    return hooks


def _row_basis(Qrows, L, idx, b, dev):
    """Qrows[L] as [b, d, r] for the batch: one basis per prompt ([n, d, r]) or one shared ([d, r])"""
    Qn = np.asarray(Qrows[L])
    Qd = torch.as_tensor(Qn[np.asarray(idx)] if Qn.ndim == 3 else Qn, device=dev, dtype=torch.float32)
    return Qd[None].expand(b, -1, -1) if Qd.ndim == 2 else Qd


def ablate_directions(Qrows, pos_of, layers, mu=None):
    """Directional ablation with a basis per prompt (Qrows[L], [n, d, r] with orthonormal columns per prompt, or one shared [d, r]):
    removes each prompt's own part. With mu (per layer, [d]) the removed part is replaced by mu's part (mean-ablation)."""

    def hooks(enc, idx):
        rows, Pm = _rows_and_positions(enc, idx, pos_of)
        b, dev = rows.shape[0], Pm.device
        out = []
        for L in layers:
            Qd = _row_basis(Qrows, L, idx, b, dev)
            Md = None if mu is None else torch.einsum("br,bdr->bd", torch.einsum("d,bdr->br", torch.as_tensor(mu[L], device=dev, dtype=torch.float32), Qd), Qd)

            def edit(x, Qd=Qd, Md=Md):
                v = x[rows, Pm]
                v = v - torch.einsum("bpr,bdr->bpd", torch.einsum("bpd,bdr->bpr", v, Qd), Qd)
                if Md is not None:
                    v = v + Md[:, None, :]
                x[rows, Pm] = v

            out.append((L, _residual_hook(edit)))
        return out

    return hooks


def add_directions(Qrows, pos_of, layers, scale):
    """Activation steering: adds scale[L] * q_i to prompt i's positions, q_i = Qrows[L][i, :, 0] (or the shared [d, r] basis's first column)."""

    def hooks(enc, idx):
        rows, Pm = _rows_and_positions(enc, idx, pos_of)
        b, dev = rows.shape[0], Pm.device
        out = []
        for L in layers:
            qd = _row_basis(Qrows, L, idx, b, dev)[:, :, 0] * float(scale[L])

            def edit(x, qd=qd):
                x[rows, Pm] = x[rows, Pm] + qd[:, None, :]

            out.append((L, _residual_hook(edit)))
        return out

    return hooks


def combine(*builders):
    """several interventions at once (their hooks are registered together)"""
    return lambda enc, idx: [h for b in builders for h in b(enc, idx)]
