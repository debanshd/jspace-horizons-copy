"""The 21 October examples rebuild from the repository alone (results/oct21 and results/samples, no datasets), and their logged
answers match the committed outputs_21oct.csv."""

import subprocess
import sys

import pandas as pd
import pytest

from jspace.config import RESULTS_DIR, ROOT

SAMPLES = RESULTS_DIR / "samples"


@pytest.mark.skipif(not (SAMPLES / "Qwen3.5-4B" / "bank").exists(), reason="needs results/samples (scripts/fig_examples21.py writes it)")
def test_two_examples_rebuild_from_the_samples(tmp_path):
    r = subprocess.run([sys.executable, str(ROOT / "scripts/fig_examples21.py"), "--out", str(tmp_path), "--only", "ex13,ex14", "--no-samples"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (tmp_path / "ex13_nothing_blocked.png").exists() and (tmp_path / "ex14_which_token_reads.png").exists()
    got = pd.read_csv(tmp_path / "outputs_21oct.csv")
    ref = pd.read_csv(RESULTS_DIR / "examples" / "outputs_21oct.csv")
    ref = ref[ref.example.isin(["ex13", "ex14"])].reset_index(drop=True)
    assert got.equals(ref)
