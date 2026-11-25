# Lens-quality prompt sets

Five of the six lens-quality prompt sets of Gurnee et al. (2026), copied unchanged from the paper's companion code,
[anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens) at commit `581d398613e5602a5af361e1c34d3a92ea82ba8e`
(the commit `uv.lock` pins), `data/evaluations/`. Apache License 2.0, Copyright 2026 Anthropic PBC.

`scripts/lens_readout.py` reads them; `config.LENS_EVAL_FILES` holds their sha256s. The sixth set, order-ops, is left out: its
intermediates must be expanded into synonym sets the companion code does not ship.
