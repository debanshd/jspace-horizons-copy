"""The stimuli: canvas geometry, and drawing that is pixel-identical to the inputs of the original runs.

The original runs saved the pictures they showed (results/*/*/stimuli.png). Redrawing the parts that need no dataset (empty slots,
slot numbers, a printed word) must give the same pixels: this fails when the DejaVu fonts are missing or Pillow draws differently,
either of which would change every stimulus.
"""

import os

import numpy as np
import pytest
from PIL import Image

from jspace import stimuli
from jspace.config import RESULTS_DIR

needs_fonts = pytest.mark.skipif(not all(os.path.exists(f) for f in stimuli.FONT_FILES), reason="DejaVu fonts not installed")


def test_canvas_geometry():
    assert stimuli.canvas(3, 2, {}).size == (672, 448)
    tokens = [t for s in range(1, 7) for t in stimuli.slot_tokens(3, 2, s)]
    assert len(tokens) == 6 * 49 and sorted(tokens) == list(range(294))  # the six slots tile the 21 x 14 token grid
    assert stimuli.slot_tokens(3, 2, 1)[:8] == [0, 1, 2, 3, 4, 5, 6, 21]


def test_fit_photo(tmp_path):
    path = tmp_path / "photo.png"
    Image.new("RGB", (1000, 600), "red").save(path)
    im = stimuli.fit_photo(path)
    assert im.size == (448, 256) and im.width % 32 == 0 and im.height % 32 == 0


@needs_fonts
def test_printed_word_canvas_matches_the_original_run():
    sheet = RESULTS_DIR / "name_steering/Qwen3.5-4B/stimuli.png"  # frames: photo scene 0, photo scene 1, printed scene 0 ('dog' in slot 3), printed scene 1
    drawn = stimuli.scene(3, stimuli.word_tile("dog"))
    assert np.array_equal(np.asarray(drawn), np.asarray(stimuli.sheet_frame(sheet, 2, 4, 672, 448)))


@needs_fonts
def test_empty_slots_match_the_original_run():
    sheet = RESULTS_DIR / "subspace_patching/Qwen3.5-4B/stimuli.png"  # frame 0: a bird in slot 1, the other five empty
    drawn = np.asarray(stimuli.scene(None, None))
    assert np.array_equal(drawn[:, 224:], np.asarray(stimuli.sheet_frame(sheet, 0, 4, 672, 448))[:, 224:])


def test_gemma_saw_the_same_first_scene():
    """example 1 shows one input for both models: the canvases of scene 0 are the same pixels in both runs"""
    q, g = RESULTS_DIR / "subspace_patching/Qwen3.5-4B/stimuli.png", RESULTS_DIR / "subspace_patching/gemma-4-31B-it/stimuli.png"
    for j in (0, 2):
        assert np.array_equal(np.asarray(stimuli.sheet_frame(q, j, 4, 672, 448)), np.asarray(stimuli.sheet_frame(g, j, 4, 672, 448)))
