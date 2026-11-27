"""Rewrite jeeva2812's Jall_main.pt into the released J-lens format the J++ harness reads (RULE3.md), with the matrices unchanged
(float16 widened exactly to float32).

The source holds {"revision", "layers", "J", "dtype_note"}; load_legacy_format_lens wants "J", "source_layers", "d_model" and
"n_prompts", and reads the block a lens maps into from provenance.target_layer. That block is 30, since the source's layer-30
matrix is the identity. Its prompt count is not recorded, so n_prompts is 0.

    python jpp_wsp/convert_jall.py <Jall_main.pt> <out.pt>
"""

import hashlib
import sys

import torch

SOURCE_SHA256 = "582d71ee0d5b1f0fc99ccc3a7aee3da1fcfa78750445b6957c77a2837cd2f85b"
TARGET_LAYER = 30


def main() -> None:
    src, out = sys.argv[1], sys.argv[2]
    h = hashlib.sha256()
    with open(src, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    if h.hexdigest() != SOURCE_SHA256:
        raise SystemExit(f"{src} has sha256 {h.hexdigest()}, RULE3.md pins {SOURCE_SHA256}")
    c = torch.load(src, map_location="cpu", weights_only=True)
    J = c["J"]
    layers = list(c["layers"])
    assert sorted(J) == layers, (sorted(J), layers)
    eye = torch.eye(J[TARGET_LAYER].shape[0], dtype=J[TARGET_LAYER].dtype)
    assert torch.equal(J[TARGET_LAYER], eye), "the layer-30 matrix is not the identity: the target block is not 30"
    torch.save(
        {
            # float32, not the source's float16: the conversion is exact, and the harness multiplies the lens on the CPU, where
            # float16 has no fast kernels (the first scoring run crawled at about 1/40 of the speed of the float32 lenses)
            "J": {L: J[L].float() for L in layers},
            "source_layers": layers,
            "d_model": J[layers[0]].shape[0],
            "n_prompts": 0,  # not recorded in the source
            "provenance": {"target_layer": TARGET_LAYER, "source": "jeeva2812/olmo3-jlens-checkpoints@a661f25:Jall_main.pt"},
        },
        out,
    )
    print("wrote", out, "float32, from", J[layers[0]].dtype)


if __name__ == "__main__":
    main()
