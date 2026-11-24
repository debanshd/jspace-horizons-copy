"""config.ALT_LENSES and lens_for: a model's lens by name, its pinned one unchanged (PR #15)."""

import pytest

from jspace import config


def test_the_pinned_lens_is_unchanged_by_default():
    for name, s in config.MODELS.items():
        repo, rev, file, sha = config.lens_for(name)
        assert (repo, rev) == s.lens_source() and file == s.lens_file and sha == s.lens_sha256


def test_qwen35_9b_reads_a_j_lens_on_request():
    assert config.MODELS["Qwen/Qwen3.5-9B"].lens_file.endswith("_jpp_lens.pt")  # its pinned lens, which the results tables read
    repo, rev, file, sha = config.lens_for("Qwen/Qwen3.5-9B", "jlens")
    assert repo == "bcywinski/jacobian-lens-qwen3.5-9b" and len(rev) == 40 and file == "lens_n1000.pt" and len(sha) == 64
    with pytest.raises(SystemExit):
        config.lens_for("Qwen/Qwen3.5-4B", "jlens")
    with pytest.raises(SystemExit):
        config.lens_for("Qwen/Qwen3.5-9B", "jpp2")
