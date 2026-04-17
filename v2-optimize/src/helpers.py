"""
Bradford Bulls v2 — Helper functions.

Sharpness metrics, pitch detection, person detection, timestamps.
Key change from v1: added compute_torso_sharpness() for logo-region-focused quality scoring.
"""

import cv2
import numpy as np
import imagehash
from PIL import Image

from . import config


# ═══════════════════════════════════════════════════════════════════════
# TIMESTAMPS
# ═══════════════════════════════════════════════════════════════════════

def fmt_timestamp(sec):
    """Convert seconds to HH:MM:SS string for filenames."""
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return f"{h:02d}h{m:02d}m{s:02d}s" if h > 0 else f"{m:02d}m{s:02d}s"


# ═══════════════════════════════════════════════════════════════════════
# SHARPNESS METRICS
# ═══════════════════════════════════════════════════════════════════════

def compute_sharpness(region, mask=None):
    """
    Multi-metric sharpness score (0-1). Higher = sharper.
    Combines Laplacian variance (55%) + Tenengrad (45%).

    Args:
        region: BGR image (numpy array)
        mask: Optional binary mask, 1=clean pixel, 0=overlay pixel
    """
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region

    if mask is not None:
        valid = mask.astype(bool)
        if valid.sum() < 100:
            return 0.0
    else:
        valid = np.ones(gray.shape, dtype=bool)

    # Laplacian variance
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = float(lap[valid].var()) if valid.any() else 0.0

    # Tenengrad (Sobel gradient magnitude)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    ten_vals = (gx ** 2 + gy ** 2)[valid]
    ten = float(ten_vals.mean()) if len(ten_vals) > 0 else 0.0

    score = min(lap_var / 300, 1.0) * 0.55 + min(ten / 5000, 1.0) * 0.45
    return round(score, 4)


def compute_player_sharpness(frame, detections, overlay_mask=None):
    """
    Compute sharpness on player bounding boxes, masking overlays.
    Returns the MAX sharpness across all detected persons.
    """
    if not detections:
        return compute_sharpness(frame, overlay_mask)

    h, w = frame.shape[:2]
    scores = []

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        crop_mask = None
        if overlay_mask is not None:
            crop_mask = overlay_mask[y1:y2, x1:x2]
            if crop_mask.mean() < 0.4:
                continue

        scores.append(compute_sharpness(crop, crop_mask))

    return round(max(scores), 4) if scores else 0.0


def compute_torso_sharpness(frame, detections, overlay_mask=None):
    """
    Compute sharpness ONLY on the TORSO region (where logos are).

    Torso = 15%-65% of player height from top of bbox.
    This is the key improvement over v1: a frame where legs are sharp
    but torso is blurred (due to body rotation) will correctly score LOW.

    Returns the MAX torso sharpness across all detected persons.
    """
    if not detections:
        return compute_sharpness(frame, overlay_mask)

    h, w = frame.shape[:2]
    scores = []

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        bh = y2 - y1
        if bh < 60 or x2 <= x1:
            continue

        # Torso region: 15%-65% of player height
        torso_y1 = max(0, y1 + int(bh * 0.15))
        torso_y2 = min(h, y1 + int(bh * 0.65))
        if torso_y2 <= torso_y1:
            continue

        torso = frame[torso_y1:torso_y2, x1:x2]
        if torso.size == 0:
            continue

        torso_mask = None
        if overlay_mask is not None:
            torso_mask = overlay_mask[torso_y1:torso_y2, x1:x2]
            if torso_mask.mean() < 0.4:
                continue

        scores.append(compute_sharpness(torso, torso_mask))

    return round(max(scores), 4) if scores else 0.0


def assign_sharpness_tier(torso_sharpness):
    """Assign quality tier based on torso sharpness score."""
    if torso_sharpness >= config.SHARPNESS_TIER_GOLD:
        return "gold"
    elif torso_sharpness >= config.SHARPNESS_TIER_SILVER:
        return "silver"
    elif torso_sharpness >= config.SHARPNESS_TIER_BRONZE:
        return "bronze"
    else:
        return "reject"


# ═══════════════════════════════════════════════════════════════════════
# PITCH DETECTION
# ═══════════════════════════════════════════════════════════════════════

def compute_pitch_green_ratio(frame_bgr, roi_y_start=None):
    """Fraction of grass-green pixels in bottom ROI (0-1)."""
    if roi_y_start is None:
        roi_y_start = config.PITCH_ROI_Y_START
    h, w = frame_bgr.shape[:2]
    y0 = int(max(0, min(h - 1, roi_y_start * h)))
    roi = frame_bgr[y0:h, 0:w]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower = np.array([35, 40, 40])
    upper = np.array([95, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    return float(mask.mean() / 255.0)


# ═══════════════════════════════════════════════════════════════════════
# PERSON DETECTION
# ═══════════════════════════════════════════════════════════════════════

def detect_persons(yolo_model, frame, device, confidence=None):
    """Run YOLO person detection on a single frame."""
    if confidence is None:
        confidence = config.PERSON_CONFIDENCE
    results = yolo_model.predict(
        frame, classes=[0], conf=confidence,
        device=device, verbose=False
    )
    detections = []
    if results and results[0].boxes is not None:
        xyxy = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i]
            detections.append({
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "conf": float(confs[i]),
                "area": float((x2 - x1) * (y2 - y1)),
            })
    return detections


def filter_foreground_players(detections, frame_h, frame_w):
    """
    Filter out background crowd detections.
    Keeps only significant-sized persons (actual players, not spectators).
    """
    if not detections or len(detections) < 2:
        return detections

    areas = [d["area"] for d in detections]
    median_area = np.median(areas)

    foreground = []
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        bh = y2 - y1
        is_significant = d["area"] > median_area * 0.25
        is_tall_enough = bh > frame_h * 0.05

        if is_significant and is_tall_enough:
            d["is_foreground"] = True
            foreground.append(d)
        else:
            d["is_foreground"] = False

    return foreground if foreground else [detections[0]]


# ═══════════════════════════════════════════════════════════════════════
# SHOT TYPE
# ═══════════════════════════════════════════════════════════════════════

def get_shot_type(max_person_area_ratio):
    """Classify shot type based on largest person size."""
    if max_person_area_ratio > 0.12:
        return "closeup"
    elif max_person_area_ratio > 0.05:
        return "medium"
    else:
        return "wide"


# ═══════════════════════════════════════════════════════════════════════
# HASHING
# ═══════════════════════════════════════════════════════════════════════

def compute_phash(frame_bgr):
    """Compute perceptual hash of a frame."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return imagehash.phash(Image.fromarray(frame_rgb))
