"""
Ignore-region mask: drop detections whose center falls inside fixed
broadcast UI overlays (scoreboard, channel logo, etc.).

Regions are specified in normalized [x1, y1, x2, y2] coordinates
(0..1 of frame width/height) so they're resolution-independent.

Common regions for Bradford Bulls broadcasts (BullsTV / Sky Sports):
  - scoreboard: bottom-left, ~y > 0.85, x < 0.50
  - BullsTV logo: top-right, ~x > 0.85, y < 0.15
  - Sky Sports bug: top-left, varies

You can capture per-broadcast templates and reuse via match.meta.yaml.
"""

from __future__ import annotations

from track_annotation.utils.geometry import Bbox, bbox_center


def bbox_in_any_region(
    bbox: Bbox,
    frame_shape: tuple[int, int],   # (H, W)
    ignore_regions: list[list[float]],
) -> bool:
    """
    True if the bbox CENTER falls inside any of the normalized ignore regions.

    Parameters
    ----------
    bbox : (x1, y1, x2, y2) in pixel coords
    frame_shape : (H, W)
    ignore_regions : list of [x1, y1, x2, y2] each in normalized [0, 1]
    """
    if not ignore_regions:
        return False
    h, w = frame_shape
    if h <= 0 or w <= 0:
        return False
    cx, cy = bbox_center(bbox)
    nx, ny = cx / w, cy / h
    for region in ignore_regions:
        rx1, ry1, rx2, ry2 = region
        if rx1 <= nx <= rx2 and ry1 <= ny <= ry2:
            return True
    return False


def bbox_overlaps_region(
    bbox: Bbox,
    frame_shape: tuple[int, int],
    ignore_regions: list[list[float]],
    min_overlap_ratio: float = 0.5,
) -> bool:
    """
    Stricter check: True if `min_overlap_ratio` or more of bbox area falls
    inside any ignore region. Use when you want to keep partial-overlap detections.
    """
    if not ignore_regions:
        return False
    h, w = frame_shape
    if h <= 0 or w <= 0:
        return False
    bx1, by1, bx2, by2 = bbox
    bw = max(0.0, bx2 - bx1)
    bh = max(0.0, by2 - by1)
    bbox_area = bw * bh
    if bbox_area <= 0:
        return False

    for region in ignore_regions:
        rx1 = region[0] * w
        ry1 = region[1] * h
        rx2 = region[2] * w
        ry2 = region[3] * h

        ix1 = max(bx1, rx1)
        iy1 = max(by1, ry1)
        ix2 = min(bx2, rx2)
        iy2 = min(by2, ry2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        if iw * ih / bbox_area >= min_overlap_ratio:
            return True
    return False
