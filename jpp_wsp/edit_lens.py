"""Write the 50% OLMo 3 7B J++ lens from the 100% one by basis transport, from weights alone (jpp_wsp/RULE.md).

J50_l = A_L J100_l B_l, where
- B_l = A_l^{-1} maps the 50% model's residual after block l into the 100% model's coordinates. It solves W50 = W100 B_l over the
  matrices of block l+1 that read that residual (q, k, v, gate, up; OLMo 3 has no pre-norm, so they read the raw residual).
- A_L maps the 100% final residual into the 50% one. It solves M50 A_L = M100 with M = lm_head . diag(final norm gain).

Both are ridge least squares (lambda = 1e-4 tr(G) / d, fixed in the rule), in float64 on CPU. No prompt, forward pass or gradient
is involved. Writes the four variants of the rule, a diagnostics JSON (how well a linear map explains the change of weights, against
the identity) and the sha256 of every lens.

    python jpp_wsp/edit_lens.py --lens <olmo7b_jpp_n64.pt> --m100 <dir> --m50 <dir> --out <dir>
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open

SOURCE_SHA256 = "6b853a64ab98384058949f3349765bddc39cc05d4bb0b4283174444e77378add"
READERS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "mlp.gate_proj", "mlp.up_proj"]
RIDGE = 1e-4
CHUNK = 8192  # rows of lm_head at a time: the full head in float64 is 3.3 GB per checkpoint


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


class Weights:
    """one checkpoint's tensors, read one at a time from its safetensors shards"""

    def __init__(self, root: Path):
        self.root = root
        self.where = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]

    def get(self, name: str) -> torch.Tensor:
        with safe_open(self.root / self.where[name], framework="pt") as f:
            return f.get_tensor(name).double()


def ridge_solve(gram: torch.Tensor, cross: torch.Tensor) -> torch.Tensor:
    """(G + lambda I)^{-1} C, lambda = RIDGE tr(G) / d"""
    d = gram.shape[0]
    lam = RIDGE * torch.trace(gram) / d
    return torch.linalg.solve(gram + lam * torch.eye(d, dtype=gram.dtype), cross)


def input_map(w100: Weights, w50: Weights, block: int) -> tuple[torch.Tensor, dict]:
    """B with W50 ~= W100 B over the reading matrices of `block`, each scaled by 1 / ||W100||_F (both checkpoints alike)"""
    gram = cross = None
    pieces = []
    for r in READERS:
        a = w100.get(f"model.layers.{block}.{r}.weight")
        b = w50.get(f"model.layers.{block}.{r}.weight")
        s = a.norm()
        a, b = a / s, b / s
        g, c = a.T @ a, a.T @ b
        gram = g if gram is None else gram + g
        cross = c if cross is None else cross + c
        pieces.append((a, b))
    B = ridge_solve(gram, cross)
    # how much of the change the map explains: ||W100 B - W50|| against ||W100 - W50|| (the identity map)
    num = sum(float((a @ B - b).pow(2).sum()) for a, b in pieces) ** 0.5
    ident = sum(float((a - b).pow(2).sum()) for a, b in pieces) ** 0.5
    ref = sum(float(b.pow(2).sum()) for _, b in pieces) ** 0.5
    eye = torch.eye(B.shape[0], dtype=B.dtype)
    diag = {
        "rel_residual_map": num / ref,
        "rel_residual_identity": ident / ref,
        "map_minus_identity_rel": float((B - eye).norm() / eye.norm()),
        "map_diag_mean": float(torch.diagonal(B).mean()),
    }
    return B, diag


def output_map(w100: Weights, w50: Weights) -> tuple[torch.Tensor, dict]:
    """A_L with M50 A_L ~= M100, M = lm_head diag(g), accumulated over row chunks of the head"""
    g100, g50 = w100.get("model.norm.weight"), w50.get("model.norm.weight")
    with safe_open(w100.root / w100.where["lm_head.weight"], framework="pt") as f100, safe_open(w50.root / w50.where["lm_head.weight"], framework="pt") as f50:
        u100, u50 = f100.get_slice("lm_head.weight"), f50.get_slice("lm_head.weight")
        vocab = u100.get_shape()[0]
        assert u50.get_shape()[0] == vocab
        d = g100.shape[0]
        gram = torch.zeros(d, d, dtype=torch.float64)
        cross = torch.zeros(d, d, dtype=torch.float64)
        for i in range(0, vocab, CHUNK):
            m100 = u100[i : i + CHUNK].double() * g100
            m50 = u50[i : i + CHUNK].double() * g50
            gram += m50.T @ m50
            cross += m50.T @ m100
        A = ridge_solve(gram, cross)
        num = ident = ref = 0.0
        for i in range(0, vocab, CHUNK):
            m100 = u100[i : i + CHUNK].double() * g100
            m50 = u50[i : i + CHUNK].double() * g50
            num += float((m50 @ A - m100).pow(2).sum())
            ident += float((m50 - m100).pow(2).sum())
            ref += float(m100.pow(2).sum())
    eye = torch.eye(d, dtype=A.dtype)
    diag = {
        "rel_residual_map": (num / ref) ** 0.5,
        "rel_residual_identity": (ident / ref) ** 0.5,
        "map_minus_identity_rel": float((A - eye).norm() / eye.norm()),
        "map_diag_mean": float(torch.diagonal(A).mean()),
        "final_gain_mean_100": float(g100.mean()),
        "final_gain_mean_50": float(g50.mean()),
    }
    return A, diag


def save_lens(source: dict, jacobians: dict[int, torch.Tensor], name: str, out: Path) -> Path:
    """the source file's structure (BaseLens.save), float32 like the source, renamed"""
    config = dict(source["config"], checkpoint_name=name)
    path = out / f"{name}.pt"
    torch.save(
        {
            "parameters": {"jacobians": {L: J.float().contiguous() for L, J in jacobians.items()}},
            "source_layers": source["source_layers"],
            "config": config,
        },
        path,
    )
    return path


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
    # float32 as stored; each layer is lifted to float64 only while it is edited
    J100 = source["parameters"]["jacobians"]
    w100, w50 = Weights(args.m100), Weights(args.m50)

    # Each layer's edited matrices go to disk as they are made, and a lens file is assembled from them one variant at a time:
    # four lenses held in memory at once (8 GB) next to the working float64 matrices overran a 15 GB machine. A layer whose
    # shards exist is not recomputed, so a stopped run resumes.
    shards = args.out / "shards"
    shards.mkdir(exist_ok=True)
    names = [f"wspV{v}_jpp_n64" for v in range(4)]
    A = None
    diagnostics = {"input": {}, "lens_change_rel": {n: {} for n in names}}
    for L in source["source_layers"]:
        shard = shards / f"layer{L}.pt"
        if shard.exists():
            print(f"layer {L}: done before", flush=True)
            continue
        if A is None:
            A, diag_out = output_map(w100, w50)
            print("output map", json.dumps(diag_out), flush=True)
            (shards / "output.json").write_text(json.dumps(diag_out))
        B, d_in = input_map(w100, w50, L + 1)  # the residual after block L is read by block L + 1
        print(f"layer {L}", json.dumps(d_in), flush=True)
        J = J100[L].double()
        edited = {"wspV0_jpp_n64": J, "wspV1_jpp_n64": A @ J @ B, "wspV2_jpp_n64": A @ J, "wspV3_jpp_n64": J @ B}
        change = {n: float((Jv - J).norm() / J.norm()) for n, Jv in edited.items()}
        torch.save({"J": {n: Jv.float() for n, Jv in edited.items()}, "input": d_in, "change": change}, shard)
        del edited, B, J

    diagnostics["output"] = json.loads((shards / "output.json").read_text())
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
