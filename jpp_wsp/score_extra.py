"""PR 18's run_all.py, scoring also every lens file in <fits>/<model key>/extra/ (RULE3.md), without changing run_all.py itself.

run_all.py --fits scores only files named like fit_lenses.py's outputs (*_rlens_n*.pt, *_jpp_n*.pt); a lens from elsewhere would
otherwise have to be renamed as one. Run from a jpp_lens checkout: uv run python score_extra.py <jpp_evals dir> <run_all args ...>
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.argv.pop(1)).resolve()))
import run_all  # noqa: E402

_fitted = run_all.fitted_lenses


def fitted_lenses(fits_dir, key):
    extra = sorted((fits_dir / key / "extra").glob("*.pt")) if fits_dir is not None else []
    return _fitted(fits_dir, key) + extra


run_all.fitted_lenses = fitted_lenses
run_all.main()
