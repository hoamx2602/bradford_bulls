"""
Color-based team classifier.

For each detected person, score how well their torso pixels match the target
team's color palette (declared in MatchContext.target_team_colors). Tracks with
mean team-score below threshold are dropped.

This is a fast, no-neural-net solution suitable for v0 person tracking. After
Stage A is trained and v1 logo tracking is in use, team filtering happens at
the logo level (logo placement implies player team) and this module becomes
optional.

Algorithm
---------
1. Crop torso = top 60% of person bbox, center 60% horizontally
2. Convert to HSV
3. For each color range in target_team_colors, compute mask of matching pixels
4. Score = (max over color ranges) of (matching_pixel_count / torso_pixel_count)
5. Track passes if mean score across detections >= min_team_score

Defining team colors
--------------------
HSV ranges work better than RGB for color matching under varying lighting.
Use H ∈ [0, 180] (OpenCV convention), S ∈ [0, 255], V ∈ [0, 255].

Bradford Bulls home kit (red + amber/yellow):
  - red:    H ∈ [0, 10] ∪ [170, 180], S ≥ 100, V ≥ 80
  - amber:  H ∈ [15, 30],              S ≥ 100, V ≥ 100

Bradford Bulls away kit (white):
  - white:  S ≤ 40, V ≥ 180  (low saturation, high brightness)
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from track_annotation.utils.geometry import Bbox
from track_annotation.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class HSVRange:
    """One HSV color range. H wraps via two ranges (e.g., red has H ∈ [0,10] ∪ [170,180])."""

    name: str
    h_min: int
    h_max: int
    s_min: int = 0
    s_max: int = 255
    v_min: int = 0
    v_max: int = 255

    def mask(self, hsv_img: np.ndarray) -> np.ndarray:
        """Return uint8 mask (0/255) of pixels matching this range."""
        if self.h_min <= self.h_max:
            return cv2.inRange(
                hsv_img,
                (self.h_min, self.s_min, self.v_min),
                (self.h_max, self.s_max, self.v_max),
            )
        # Hue wraps around (e.g., red): split into two ranges
        m1 = cv2.inRange(hsv_img, (self.h_min, self.s_min, self.v_min), (180, self.s_max, self.v_max))
        m2 = cv2.inRange(hsv_img, (0, self.s_min, self.v_min), (self.h_max, self.s_max, self.v_max))
        return cv2.bitwise_or(m1, m2)


def torso_crop(image: np.ndarray, bbox: Bbox, top_ratio: float = 0.6, center_ratio: float = 0.6) -> np.ndarray:
    """
    Extract the torso region (where logos live) from a person bbox.

    Default: top 60% vertically, central 60% horizontally — covers chest area
    while excluding head, legs, and arm extremities.
    """
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((0, 0, 3), dtype=image.dtype)

    bw = x2 - x1
    bh = y2 - y1

    # Vertical: top portion (skip head a tiny bit)
    head_skip = int(0.10 * bh)
    torso_y1 = y1 + head_skip
    torso_y2 = y1 + int(top_ratio * bh)

    # Horizontal: center portion
    margin_x = int((1.0 - center_ratio) / 2.0 * bw)
    torso_x1 = x1 + margin_x
    torso_x2 = x2 - margin_x

    if torso_y2 <= torso_y1 or torso_x2 <= torso_x1:
        return np.zeros((0, 0, 3), dtype=image.dtype)

    return image[torso_y1:torso_y2, torso_x1:torso_x2]


def team_score(
    image_bgr: np.ndarray,
    bbox: Bbox,
    color_ranges: list[HSVRange],
    min_torso_pixels: int = 400,
) -> float:
    """
    Score how well the torso of a person matches the target color palette.

    Returns a value in [0, 1]: ratio of torso pixels matching ANY of the color
    ranges (max across ranges). 0 if torso is too small to evaluate.
    """
    if not color_ranges:
        return 0.0
    torso = torso_crop(image_bgr, bbox)
    if torso.size < min_torso_pixels * 3:  # 3 channels
        return 0.0
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    n = torso.shape[0] * torso.shape[1]
    best = 0.0
    for cr in color_ranges:
        mask = cr.mask(hsv)
        ratio = float(np.count_nonzero(mask)) / n
        if ratio > best:
            best = ratio
    return best


def parse_color_ranges(spec: list[dict]) -> list[HSVRange]:
    """Parse a list of dict color specs into HSVRange objects.

    Each dict is like:
        {name: red, h: [0, 10], s: [100, 255], v: [80, 255]}
    """
    out: list[HSVRange] = []
    for c in spec:
        h = c.get("h", [0, 180])
        s = c.get("s", [0, 255])
        v = c.get("v", [0, 255])
        out.append(HSVRange(
            name=c.get("name", f"color_{len(out)}"),
            h_min=int(h[0]), h_max=int(h[1]),
            s_min=int(s[0]), s_max=int(s[1]),
            v_min=int(v[0]), v_max=int(v[1]),
        ))
    return out
