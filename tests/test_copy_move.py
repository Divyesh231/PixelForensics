# test_copy_move.py
import cv2
import numpy as np
from src.copy_move import detect_copy_move


def test_blank_image():
    """A blank image should return zero matches."""
    blank = np.zeros((200, 200, 3), dtype=np.uint8)
    result = detect_copy_move(blank)
    assert result["match_count"] == 0
    assert result["cluster_count"] == 0
    assert not np.any(result["mask"])


def test_synthetic_clone():
    """Create a known copy-move and check detection runs."""
    canvas = np.zeros((300, 400, 3), dtype=np.uint8)
    # add some texture lines so there are features to detect
    for i in range(0, 300, 20):
        cv2.line(canvas, (0, i), (400, i), (60, 60, 60), 1)

    # draw a distinctive object
    cv2.circle(canvas, (80, 80), 30, (200, 100, 50), -1)
    cv2.rectangle(canvas, (60, 60), (100, 100), (50, 220, 100), 2)
    cv2.putText(canvas, "X", (70, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1)

    # clone it
    patch = canvas[40:120, 40:120].copy()
    canvas[160:240, 260:340] = patch

    result = detect_copy_move(canvas, min_dist=20.0, cluster_eps=30.0,
                              min_cluster=3)

    assert "overlay" in result
    assert "mask" in result
    assert result["mask"].shape == (300, 400)
