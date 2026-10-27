# ela.py - Error Level Analysis
#
# Basic idea: re-save a JPEG at known quality, then subtract.
# Regions that were pasted in from another image have different
# compression artifacts so they show up brighter in the difference.

import io
import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance


def load_image(path):
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8)


def compute_ela(image, quality=90, scale=15.0):
    if isinstance(image, np.ndarray):
        pil_img = Image.fromarray(image).convert("RGB")
    else:
        pil_img = image.convert("RGB")

    # re-save to in-memory buffer
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    resaved = Image.open(buf).convert("RGB")

    # pixel difference
    diff = ImageChops.difference(pil_img, resaved)
    diff_arr = np.array(diff, dtype=np.float32)

    # single channel version (average across RGB)
    raw_diff = np.mean(diff_arr, axis=2)

    # amplify for display
    enhanced = ImageEnhance.Brightness(diff).enhance(scale)
    amplified = np.array(enhanced, dtype=np.uint8)

    # simple anomaly score based on spread of high errors vs median
    p50 = float(np.percentile(raw_diff, 50))
    p95 = float(np.percentile(raw_diff, 95))
    score = float(np.clip((p95 - p50) / 18.0, 0.0, 1.0))

    return amplified, raw_diff, score


def make_ela_heatmap(raw_diff, colormap=cv2.COLORMAP_INFERNO):
    mx = raw_diff.max()
    if mx > 0:
        normalized = np.clip((raw_diff / mx) * 255, 0, 255).astype(np.uint8)
    else:
        normalized = np.zeros(raw_diff.shape, dtype=np.uint8)

    colored_bgr = cv2.applyColorMap(normalized, colormap)
    return cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
