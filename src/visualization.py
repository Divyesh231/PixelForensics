# visualization.py - Heatmap overlays and evidence fusion
# All the display/rendering stuff goes here so app.py stays clean.

from io import BytesIO
import cv2
import numpy as np
from PIL import Image


def make_thermal_overlay(base_image, score_map, alpha=0.55, blur_size=21):
    """
    Create a JET thermal heatmap blended onto the original image.
    Blue = clean, Red/Yellow = suspicious.
    This gives us that forensic paper look.
    """
    h, w = base_image.shape[:2]

    # normalize the score map to [0, 1]
    s = score_map.astype(np.float32).copy()
    if s.shape[:2] != (h, w):
        s = cv2.resize(s, (w, h))
    if s.max() > 0:
        s = s / s.max()

    # gaussian blur makes it look smooth and professional
    # instead of blocky pixel-level noise
    if blur_size > 1:
        k = blur_size if blur_size % 2 == 1 else blur_size + 1
        s = cv2.GaussianBlur(s, (k, k), 0)
        if s.max() > 0:
            s = s / s.max()

    # apply JET colormap and blend
    score_uint8 = (s * 255).astype(np.uint8)
    jet_bgr = cv2.applyColorMap(score_uint8, cv2.COLORMAP_JET)
    jet_rgb = cv2.cvtColor(jet_bgr, cv2.COLOR_BGR2RGB)

    blended = cv2.addWeighted(base_image, 1.0 - alpha, jet_rgb, alpha, 0)
    return blended.astype(np.uint8)


def make_standalone_heatmap(score_map, blur_size=21):
    """Pure heatmap without the photo underneath."""
    s = score_map.astype(np.float32).copy()
    if s.max() > 0:
        s = s / s.max()

    if blur_size > 1:
        k = blur_size if blur_size % 2 == 1 else blur_size + 1
        s = cv2.GaussianBlur(s, (k, k), 0)
        if s.max() > 0:
            s = s / s.max()

    u8 = (s * 255).astype(np.uint8)
    bgr = cv2.applyColorMap(u8, cv2.COLORMAP_JET)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def draw_mask_overlay(image, mask, color=(255, 50, 50), alpha=0.45):
    """Draw a colored transparent overlay + white contours on masked regions."""
    if image.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]),
                          interpolation=cv2.INTER_NEAREST)

    result = image.copy()
    active = mask > 20
    if not np.any(active):
        return result

    # blend the colored highlight onto the original image
    overlay = image.copy()
    overlay[active] = color
    blended = cv2.addWeighted(overlay, alpha, result, 1.0 - alpha, 0)
    result[active] = blended[active]

    # contour outline around detected regions
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (255, 255, 255), 2)
    return result


def fuse_evidence(ela_diff, cm_mask, noise_map, weights=(0.50, 0.30, 0.20)):
    """
    Combine all three engine outputs into one fused evidence map.
    Copy-move receives the largest heuristic weight. This map is for
    visualization and is not a calibrated probability map.
    """
    h, w = ela_diff.shape[:2]

    # resize everything to match
    cm = cv2.resize(cm_mask, (w, h)).astype(np.float32) / 255.0
    noise = cv2.resize(noise_map, (w, h)).astype(np.float32) / 255.0

    ela = ela_diff.astype(np.float32)
    if ela.max() > 0:
        ela = ela / ela.max()

    w1, w2, w3 = weights
    total = w1 + w2 + w3
    fused = (w1 * ela + w2 * cm + w3 * noise) / total
    fused = np.clip(fused, 0.0, 1.0)

    # make heatmap version too
    heatmap = make_standalone_heatmap(fused, blur_size=31)

    return fused, heatmap


def get_verdict(ela_score, cm_score, noise_score, cm_matches=0, cm_clusters=0):
    """
    Combine all engine scores into a final verdict.
    Returns dict with verdict text, risk percentage, color, and explanation bullets.
    """
    risk = (cm_score * 0.50) + (ela_score * 0.30) + (noise_score * 0.20)
    risk_pct = int(min(100, max(0, risk * 100)))

    findings = []

    # copy-move findings
    if cm_clusters > 0 and cm_matches >= 5:
        findings.append(
            f"Copy-move-like feature cluster found: {cm_matches} keypoint matches in "
            f"{cm_clusters} cluster(s)."
        )
    elif cm_matches > 0:
        findings.append(f"Minor feature correlations ({cm_matches} pairs), "
                        f"below threshold.")
    else:
        findings.append("No copy-move-like feature cluster found.")

    # ELA findings
    if ela_score > 0.65:
        findings.append("Strong ELA variation; inspect the map for context.")
    elif ela_score > 0.40:
        findings.append("Moderate ELA variation; inspect the map for context.")
    else:
        findings.append("No strong ELA variation in this heuristic check.")

    # noise findings
    if noise_score > 0.60:
        findings.append("Local residual variation is elevated in some regions.")
    else:
        findings.append("No strong local residual variation in this heuristic check.")

    # final verdict
    if cm_clusters > 0 and cm_matches >= 10:
        verdict = "COPY-MOVE-LIKE MATCHES FOUND"
        color = "#FF3333"
    elif risk_pct >= 65:
        verdict = "HIGH HEURISTIC RISK — REVIEW"
        color = "#FF4B4B"
    elif risk_pct >= 40:
        verdict = "REVIEW RECOMMENDED"
        color = "#FFA500"
    else:
        verdict = "NO STRONG INDICATORS FOUND"
        color = "#00CC66"

    return {
        "verdict": verdict,
        "risk_pct": risk_pct,
        "color": color,
        "findings": findings,
    }


def image_to_bytes(arr, fmt="PNG"):
    """Convert numpy image array to downloadable bytes."""
    pil = Image.fromarray(arr.astype(np.uint8))
    buf = BytesIO()
    pil.save(buf, format=fmt)
    return buf.getvalue()
