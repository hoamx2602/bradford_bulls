"""
Geometry and image-quality helpers.

Bbox is represented as a 4-tuple in xyxy format: (x1, y1, x2, y2) in pixel coords.
"""

from __future__ import annotations

import cv2
import numpy as np

Bbox = tuple[float, float, float, float]


def iou(box_a: Bbox, box_b: Bbox) -> float:
    """Intersection over Union of two xyxy bboxes."""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def bbox_area(box: Bbox) -> float:
    """Area of an xyxy bbox in pixel^2."""
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_area_ratio(box: Bbox, frame_shape: tuple[int, int]) -> float:
    """Area of bbox / area of frame. frame_shape = (H, W)."""
    h, w = frame_shape
    frame_area = h * w
    if frame_area == 0:
        return 0.0
    return bbox_area(box) / frame_area


def compute_sharpness(image: np.ndarray, bbox: Bbox | None = None) -> float:
    """
    Compute sharpness via variance of Laplacian.

    If bbox is provided, only the bbox region is measured (better for evaluating
    sharpness of a specific track). Otherwise the whole frame is measured.
    """
    if bbox is not None:
        x1, y1, x2, y2 = (int(round(v)) for v in bbox)
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        crop = image[y1:y2, x1:x2]
    else:
        crop = image

    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def crop_with_padding(
    image: np.ndarray,
    bbox: Bbox,
    pad_ratio: float = 0.15,
    min_size: int = 96,
) -> tuple[np.ndarray, Bbox]:
    """
    Crop image around bbox with padding. Clamps to image bounds.

    Returns
    -------
    crop : np.ndarray
        The cropped image (resized up to min_size if smaller).
    actual_bbox : Bbox
        The bbox actually used for cropping (after padding/clamping), in original image coords.
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1

    pad_w = bw * pad_ratio
    pad_h = bh * pad_ratio

    cx1 = max(0, int(round(x1 - pad_w)))
    cy1 = max(0, int(round(y1 - pad_h)))
    cx2 = min(w, int(round(x2 + pad_w)))
    cy2 = min(h, int(round(y2 + pad_h)))

    crop = image[cy1:cy2, cx1:cx2].copy()

    # Upscale if too small (for human readability in reviewer UI)
    if crop.size > 0:
        ch, cw = crop.shape[:2]
        if max(ch, cw) < min_size:
            scale = min_size / max(ch, cw)
            new_w = int(cw * scale)
            new_h = int(ch * scale)
            crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    return crop, (cx1, cy1, cx2, cy2)


def bbox_center(box: Bbox) -> tuple[float, float]:
    """Return (cx, cy) center of bbox."""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
