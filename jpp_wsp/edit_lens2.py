"""Experiment 2 (jpp_wsp/RULE2.md): the 50% OLMo 3 7B J++ lens from the 100% one by rotation-only basis transport, from weights alone.

J50_l = A_L J100_l B_l with A_L and B_l orthogonal Procrustes solutions over the same matrices as experiment 1 (edit_lens.py): a
rotation can turn the residual basis but cannot shrink it or fold noise into it, which experiment 1's unconstrained least squares
did. Writes P1 (both), P2 (output only), P3 (input only), diagnostics.json and sha256.json; per-layer shards on disk, so it resumes.

    python jpp_wsp/edit_lens2.py --lens <olmo7b_jpp_n64.pt> --m100 <dir> --m50 <dir> --out <dir>
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
from edit_lens import CHUNK, READERS, SOURCE_SHA256, Weights, save_lens, sha256  # noqa: E402


def procrustes(cross: torch.Tensor) -> torch.Tensor:
    """the orthogonal R maximising tr(R^T C), C = X^T Y, i.e. minimising ||X R - Y||_F: U V^T of the SVD of C"""
    U, _, Vh = torch.linalg.svd(cross)
    return U @ Vh


def input_rotation(w100: Weights, w50: Weights, block: int) -> tuple[torch.Tensor, dict]:
    """B with W50 ~= W100 B, B orthogonal, over the reading matrices of `block`, each scaled by 1 / ||W100||_F as in experiment 1"""
    cross = None
    pieces = []
    for r in READERS:
        a = w100.get(f"model.layers.{block}.{r}.weight")
        b = w50.get(f"model.layers.{block}.{r}.weight")
        s = a.norm()
        a, b = a / s, b / s
        c = a.T @ b
        cross = c if cross is None else cross + c
        pieces.append((a, b))
    B = procrustes(cross)
    num = sum(float((a @ B - b).pow(2).sum()) for a, b in pieces) ** 0.5
    ident = sum(float((a - b).pow(2).sum()) for a, b in pieces) ** 0.5
    ref = sum(float(b.pow(2).sum()) for _, b in pieces) ** 0.5
    eye = torch.eye(B.shape[0], dtype=B.dtype)
    return B, {
        "rel_residual_rotation": num / ref,
        "rel_residual_identity": ident / ref,
        "rotation_minus_identity_rel": float((B - eye).norm() / eye.norm()),
        "rotation_diag_mean": float(torch.diagonal(B).mean()),
    }


def output_rotation(w100: Weights, w50: Weights) -> tuple[torch.Tensor, dict]:
    """A_L with M50 A_L ~= M100, A_L orthogonal, M = lm_head diag(g), over row chunks of the head"""
    g100, g50 = w100.get("model.norm.weight"), w50.get("model.norm.weight")
    with safe_open(w100.root / w100.where["lm_head.weight"], framework="pt") as f100, safe_open(
        w50.root / w50.where["lm_head.weight"], framework="pt"
    ) as f50:
        u100, u50 = f100.get_slice("lm_head.weight"), f50.get_slice("lm_head.weight")
        vocab, d = u100.get_shape()[0], g100.shape[0]
        cross = torch.zeros(d, d, dtype=torch.float64)
        for i in range(0, vocab, CHUNK):
            cross += (u50[i : i + CHUNK].double() * g50).T @ (u100[i : i + CHUNK].double() * g100)
        A = procrustes(cross)
        num = ident = ref = 0.0
        for i in range(0, vocab, CHUNK):
            m100 = u100[i : i + CHUNK].double() * g100
            m50 = u50[i : i + CHUNK].double() * g50
            num += float((m50 @ A - m100).pow(2).sum())
            ident += float((m50 - m100).pow(2).sum())
            ref += float(m100.pow(2).sum())
    eye = torch.eye(d, dtype=A.dtype)
    return A, {
        "rel_residual_rotation": (num / ref) ** 0.5,
        "rel_residual_identity": (ident / ref) ** 0.5,
        "rotation_minus_identity_rel": float((A - eye).norm() / eye.norm()),
        "rotation_diag_mean": float(torch.diagonal(A).mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lens", type=Path, required=True)
    ap.add_argument("--m100", type=Path, required=True)
    ap.add_argument("--m50", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    got = sha256(args.lens)
    if got != SOURCE_SHA256:
        raise SystemExit(f"{args.lens} has sha256 {got}, the rule pins {SOURCE_SHA256}")
    source = torch.load(args.lens, map_location="cpu", weights_only=True)
    J100 = source["parameters"]["jacobians"]
    w100, w50 = Weights(args.m100), Weights(args.m50)

    shards = args.out / "shards"
    shards.mkdir(exist_ok=True)
    names = [f"wspP{v}_jpp_n64" for v in (1, 2, 3)]
    A = None
    for L in source["source_layers"]:
        shard = shards / f"layer{L}.pt"
        if shard.exists():
            print(f"layer {L}: done before", flush=True)
            continue
        if A is None:
            A, diag_out = output_rotation(w100, w50)
            print("output rotation", json.dumps(diag_out), flush=True)
            (shards / "output.json").write_text(json.dumps(diag_out))
        B, d_in = input_rotation(w100, w50, L + 1)  # the residual after block L is read by block L + 1
        print(f"layer {L}", json.dumps(d_in), flush=True)
        J = J100[L].double()
        edited = {"wspP1_jpp_n64": A @ J @ B, "wspP2_jpp_n64": A @ J, "wspP3_jpp_n64": J @ B}
        change = {n: float((Jv - J).norm() / J.norm()) for n, Jv in edited.items()}
        torch.save({"J": {n: Jv.float() for n, Jv in edited.items()}, "input": d_in, "change": change}, shard)
        del edited, B, J

    diagnostics = {"output": json.loads((shards / "output.json").read_text()), "input": {}, "lens_change_rel": {n: {} for n in names}}
    hashes = {}
    for name in names:
        jac = {}
        for L in source["source_layers"]:
            s = torch.load(shards / f"layer{L}.pt", map_location="cpu", weights_only=True)
            jac[L] = s["J"][name]
            diagnostics["input"][L] = s["input"]
            diagnostics["lens_change_rel"][name][L] = s["change"][name]
            del s
        path = save_lens(source, jac, name, args.out)
        del jac
        hashes[name] = sha256(path)
        print(name, hashes[name], flush=True)
    (args.out / "diagnostics.json").write_text(json.dumps(diagnostics, indent=1))
    (args.out / "sha256.json").write_text(json.dumps(hashes, indent=1))


if __name__ == "__main__":
    main()
