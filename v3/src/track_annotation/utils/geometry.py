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


def compute_sharpness(
    image: np.ndarray,
    bbox: Bbox | None = None,
    torso_only: bool = False,
) -> float:
    """
    Compute sharpness via variance of Laplacian.

    Parameters
    ----------
    image : np.ndarray
        BGR image.
    bbox : Bbox | None
        If provided, restrict measurement to this bbox.
    torso_only : bool
        If True (and bbox provided), restrict further to the TORSO region —
        top 60% vertically, central 60% horizontally — where logos live.
        This biases keyframe selection toward frames where the LOGO area is
        sharp, not just any high-frequency noise (text on UI overlays, etc.).

    Returns
    -------
    float
        Variance of Laplacian; clamped to 0 for empty/invalid crops.
    """
    if bbox is not None:
        x1, y1, x2, y2 = (int(round(v)) for v in bbox)
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return 0.0

        if torso_only:
            bw = x2 - x1
            bh = y2 - y1
            head_skip = int(0.10 * bh)
            ty1 = y1 + head_skip
            ty2 = y1 + int(0.60 * bh)
            mx = int(0.20 * bw)
            tx1 = x1 + mx
            tx2 = x2 - mx
            if ty2 <= ty1 or tx2 <= tx1:
                crop = image[y1:y2, x1:x2]
            else:
                crop = image[ty1:ty2, tx1:tx2]
        else:
            crop = image[y1:y2, x1:x2]
    else:
        crop = image

    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def draw_highlighted_bbox(
    image: np.ndarray,
    bbox: Bbox,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 4,
    dim_outside: float = 0.55,
    label: str | None = None,
) -> np.ndarray:
    """
    Draw a bbox and dim everything outside it. Makes the subject obvious in
    busy frames (vs. a thin rectangle that disappears).

    Parameters
    ----------
    dim_outside : float in [0, 1]
        How much to darken the area outside the bbox. 0.55 = keep 55% brightness
        outside; 1.0 = no dimming; 0.0 = full black outside.
    label : str | None
        Optional text label drawn above the bbox.
    """
    out = image.copy()
    h, w = out.shape[:2]
    x1, y1, x2, y2 = (max(0, min(int(round(v)), d)) for v, d in zip(bbox, [w, h, w, h]))

    if dim_outside < 1.0 and 0.0 <= dim_outside <= 1.0:
        # Build mask: 1.0 inside bbox, dim_outside outside
        mask = np.full((h, w), dim_outside, dtype=np.float32)
        mask[y1:y2, x1:x2] = 1.0
        out = (out.astype(np.float32) * mask[..., None]).clip(0, 255).astype(np.uint8)

    cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

    # Corner brackets for extra visibility
    bracket_len = max(20, min(60, (x2 - x1) // 4))
    bracket_color = color
    bracket_thick = thickness + 2
    for corner in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
        cx, cy = corner
        sx = -1 if cx == x2 else 1
        sy = -1 if cy == y2 else 1
        cv2.line(out, (cx, cy), (cx + sx * bracket_len, cy), bracket_color, bracket_thick)
        cv2.line(out, (cx, cy), (cx, cy + sy * bracket_len), bracket_color, bracket_thick)

    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.8
        font_thick = 2
        (tw, th), _ = cv2.getTextSize(label, font, scale, font_thick)
        ty = max(th + 6, y1 - 6)
        cv2.rectangle(out, (x1, ty - th - 6), (x1 + tw + 8, ty + 4), color, -1)
        cv2.putText(out, label, (x1 + 4, ty), font, scale, (0, 0, 0), font_thick, cv2.LINE_AA)

    return out


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
