# test_noise.py
import numpy as np
from src.noise_analysis import compute_noise_map


def test_noise_runs():
    """Basic smoke test - make sure it doesn't crash."""
    img = np.random.randint(50, 200, (150, 150, 3), dtype=np.uint8)
    norm, heatmap, score = compute_noise_map(img, block_size=16)
    assert norm.shape == (150, 150)
    assert heatmap.shape == (150, 150, 3)
    assert 0.0 <= score <= 1.0


def test_noise_flat_image():
    """A flat solid image should have near-zero noise variance."""
    img = np.full((100, 100), 100, dtype=np.uint8)
    norm, _, score = compute_noise_map(img, block_size=8)
    assert norm.max() < 10  # basically no deviation
