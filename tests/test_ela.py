# test_ela.py
import numpy as np
from src.ela import compute_ela, make_ela_heatmap


def test_ela_basic():
    """Make sure ELA runs without crashing on a simple image."""
    img = np.full((120, 160, 3), 128, dtype=np.uint8)
    img[30:70, 30:70] = [255, 0, 0]  # red square

    amp, diff, score = compute_ela(img, quality=90, scale=10.0)

    assert amp.shape == (120, 160, 3)
    assert diff.shape == (120, 160)
    assert 0.0 <= score <= 1.0


def test_ela_heatmap():
    """Check that the heatmap comes out as a proper RGB image."""
    diff = np.random.uniform(0, 10, (100, 100)).astype(np.float32)
    heatmap = make_ela_heatmap(diff)
    assert heatmap.shape == (100, 100, 3)
    assert heatmap.dtype == np.uint8


def test_ela_all_black():
    """ELA on a solid black image should produce near-zero diff."""
    img = np.zeros((80, 80, 3), dtype=np.uint8)
    _, diff, score = compute_ela(img)
    assert diff.max() < 5  # should be basically nothing
