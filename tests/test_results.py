"""The reference results agree with the code that reads them."""

import subprocess
import sys

import pandas as pd
import pytest

from jspace.config import DATA_DIR, RESULTS_DIR, ROOT, require_data


def test_check_passes_on_the_reference_itself():
    r = subprocess.run([sys.executable, str(ROOT / "scripts/check.py"), "--runs", str(RESULTS_DIR)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


@pytest.mark.skipif(not (DATA_DIR / "gqa").exists() or not (DATA_DIR / "scam").exists(), reason="needs data/gqa and data/scam (scripts/download.py)")
def test_the_figures_rebuild_from_the_reference(tmp_path):
    r = subprocess.run([sys.executable, str(ROOT / "scripts/make_figures.py"), "--runs", str(RESULTS_DIR), "--out", str(tmp_path)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert pd.read_csv(tmp_path / "outputs.csv").equals(pd.read_csv(RESULTS_DIR / "examples/outputs.csv"))


def test_missing_data_stops_a_run_before_the_model_loads(tmp_path):
    with pytest.raises(SystemExit, match="scripts/download.py"):
        require_data(tmp_path, "coco")
