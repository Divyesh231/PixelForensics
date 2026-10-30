# copy_move.py - Copy-Move Forgery Detection
#
# If someone copy-pastes a region inside the same image,
# both the original and pasted version have the same local features.
# We extract keypoints, match them against the image itself,
# filter out noise, and cluster points that share similar displacement vectors.

import cv2
import numpy as np
from sklearn.cluster import DBSCAN


def _get_detector(method="auto"):
    if method in ("sift", "auto"):
        try:
            return cv2.SIFT_create(nfeatures=4000), "SIFT"
        except (AttributeError, cv2.error):
            pass
    return cv2.ORB_create(nfeatures=3000), "ORB"


def detect_copy_move(image, method="auto", min_dist=35.0, ratio=0.72,
                     cluster_eps=25.0, min_cluster=5):
    if image.ndim == 2:
        gray = image
        canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        canvas = image.copy()

    h, w = gray.shape
    empty_mask = np.zeros((h, w), dtype=np.uint8)

    empty_result = {
        "overlay": canvas,
        "mask": empty_mask,
        "match_count": 0,
        "cluster_count": 0,
        "score": 0.0,
        "pairs": [],
    }

    detector, algo = _get_detector(method)
    kps, descs = detector.detectAndCompute(gray, None)

    if descs is None or len(kps) < 4:
        return empty_result

    # match descriptors against themselves to find duplicate patches
    norm = cv2.NORM_L2 if algo == "SIFT" else cv2.NORM_HAMMING
    bf = cv2.BFMatcher(norm)
    matches = bf.knnMatch(descs, descs, k=3)

    candidates = []
    displacements = []

    for match_group in matches:
        if len(match_group) < 2:
            continue

        valid = [m for m in match_group if m.queryIdx != m.trainIdx]
        if len(valid) < 2:
            continue

        best, second = valid[0], valid[1]

        # Lowe's ratio test
        if best.distance >= ratio * second.distance:
            continue

        pt1 = np.array(kps[best.queryIdx].pt)
        pt2 = np.array(kps[best.trainIdx].pt)
        dist = float(np.linalg.norm(pt1 - pt2))

        # skip nearby points (textures)
        if dist < min_dist:
            continue

        dx = pt2[0] - pt1[0]
        dy = pt2[1] - pt1[1]

        # normalize direction
        if dx < 0 or (dx == 0 and dy < 0):
            dx, dy = -dx, -dy

        candidates.append((pt1, pt2))
        displacements.append([dx, dy])

    if len(displacements) < min_cluster:
        result = empty_result.copy()
        result["match_count"] = len(candidates)
        return result

    # group similar displacement vectors with DBSCAN
    db = DBSCAN(eps=cluster_eps, min_samples=min_cluster)
    labels = db.fit_predict(displacements)

    good_labels = set(labels) - {-1}
    confirmed = []
    mask = np.zeros((h, w), dtype=np.uint8)
    overlay = canvas.copy()

    colors = [(0, 255, 128), (255, 90, 90), (255, 210, 0),
              (0, 200, 255), (230, 90, 255)]

    for label_id in good_labels:
        color = colors[label_id % len(colors)]
        indices = np.where(labels == label_id)[0]

        src_pts = []
        dst_pts = []

        for idx in indices:
            p1, p2 = candidates[idx]
            confirmed.append(((int(p1[0]), int(p1[1])),
                              (int(p2[0]), int(p2[1]))))
            src_pts.append(p1)
            dst_pts.append(p2)

            a = (int(p1[0]), int(p1[1]))
            b = (int(p2[0]), int(p2[1]))
            cv2.line(overlay, a, b, color, 1, cv2.LINE_AA)
            cv2.circle(overlay, a, 4, (255, 0, 0), -1)
            cv2.circle(overlay, b, 4, (0, 0, 255), -1)
            cv2.circle(mask, a, 16, 255, -1)
            cv2.circle(mask, b, 16, 255, -1)

        if len(src_pts) >= 3:
            hull1 = cv2.convexHull(np.array(src_pts, dtype=np.int32))
            hull2 = cv2.convexHull(np.array(dst_pts, dtype=np.int32))
            cv2.drawContours(overlay, [hull1], -1, (255, 255, 255), 2)
            cv2.drawContours(overlay, [hull2], -1, (255, 255, 255), 2)
            cv2.fillPoly(mask, [hull1, hull2], 255)

    if np.any(mask):
        kernel = np.ones((11, 11), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    score = float(np.clip(len(confirmed) / 25.0, 0.0, 1.0))

    return {
        "overlay": overlay,
        "mask": mask,
        "match_count": len(confirmed),
        "cluster_count": len(good_labels),
        "score": score,
        "pairs": confirmed,
    }
