# noise_analysis.py - Local Noise Inconsistency Detection
#
# Every camera sensor has its own noise pattern. If you splice in
# content from a different camera or AI, the local noise statistics
# will differ from the rest of the image.
#
# We extract noise by subtracting a median-filtered image,
# then calculate local variance in sliding windows.

import cv2
import numpy as np


def compute_noise_map(image, filter_size=3, block_size=16):
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    gray_f = gray.astype(np.float32)

    # step 1: extract noise residual
    # median filter preserves edges while smoothing noise
    denoised = cv2.medianBlur(gray, filter_size).astype(np.float32)
    residual = np.abs(gray_f - denoised)

    # step 2: compute local variance using box filter
    # Var(X) = E[X^2] - (E[X])^2
    k = max(3, int(block_size))
    mean_r = cv2.boxFilter(residual, cv2.CV_32F, (k, k))
    mean_sq = cv2.boxFilter(residual ** 2, cv2.CV_32F, (k, k))
    local_var = np.maximum(0.0, mean_sq - mean_r ** 2)
    local_std = np.sqrt(local_var)

    # step 3: find regions deviating from global median noise
    global_median = float(np.median(local_std))
    deviation = np.abs(local_std - global_median)

    # normalize to 0-255
    p98 = float(np.percentile(deviation, 98))
    denom = max(p98, 0.001)
    norm_map = np.clip((deviation / denom) * 255, 0, 255).astype(np.uint8)

    colored_bgr = cv2.applyColorMap(norm_map, cv2.COLORMAP_VIRIDIS)
    heatmap = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)

    high_pct = float(np.mean(norm_map > 180))
    score = float(np.clip(high_pct * 12.0, 0.0, 1.0))

    return norm_map, heatmap, score
