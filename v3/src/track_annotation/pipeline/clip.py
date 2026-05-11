"""
Per-track video clip extraction.

For each track, extract a short video clip (default 2s) centered on the track
midpoint. Clip is used by the reviewer UI so the annotator can perceive the
logo via temporal integration (ý tưởng 3).

The clip follows the tracked player: each frame is cropped to a fixed window
centered on the player's detected position, so the annotator only sees the
relevant player — not the full 1920×1080 scene with all opponents visible.

Backend: cv2 VideoWriter (pure Python, no extra deps).
Note: ffmpeg backend is retained for full-frame fallback but player-following
always uses cv2 since ffmpeg cannot do per-frame adaptive crop easily.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from track_annotation.config import ClipConfig
from track_annotation.pipeline.detect_track import Track
from track_annotation.utils.logging import get_logger
from track_annotation.utils.video_io import VideoReader, get_video_metadata

log = get_logger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _build_frame_bbox_lookup(track: Track) -> dict[int, tuple[float, float, float, float]]:
    """Map frame_idx → bbox for every detection in the track."""
    return {d.frame_idx: d.bbox for d in track.detections}


def _nearest_bbox(
    lookup: dict[int, tuple],
    frame_idx: int,
) -> tuple[float, float, float, float] | None:
    """Return the bbox of the detection closest in time to frame_idx."""
    if not lookup:
        return None
    if frame_idx in lookup:
        return lookup[frame_idx]
    # Find nearest key
    nearest = min(lookup.keys(), key=lambda k: abs(k - frame_idx))
    return lookup[nearest]


def _compute_crop_window(
    track: Track,
    frame_h: int,
    frame_w: int,
    context_pad: float,
) -> tuple[int, int]:
    """
    Compute a fixed (crop_h, crop_w) for the entire clip based on the
    median bbox size across the track + context padding.

    Using a fixed size keeps the clip stable (no zoom jumps frame-to-frame).
    """
    widths  = [d.bbox[2] - d.bbox[0] for d in track.detections]
    heights = [d.bbox[3] - d.bbox[1] for d in track.detections]
    med_w = float(np.median(widths))
    med_h = float(np.median(heights))
    crop_w = int(min(frame_w, med_w * (1 + 2 * context_pad)))
    crop_h = int(min(frame_h, med_h * (1 + 2 * context_pad)))
    # At least 128×128 so the clip is always readable
    crop_w = max(crop_w, 128)
    crop_h = max(crop_h, 128)
    return crop_h, crop_w


def _crop_frame_to_player(
    frame: np.ndarray,
    bbox: tuple[float, float, float, float],
    crop_h: int,
    crop_w: int,
) -> np.ndarray:
    """
    Crop *frame* to a (crop_h × crop_w) window centered on *bbox*.
    Clamps to frame bounds, pads with black if near edges.
    """
    fh, fw = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)

    # Desired crop region
    r_x1 = cx - crop_w // 2
    r_y1 = cy - crop_h // 2
    r_x2 = r_x1 + crop_w
    r_y2 = r_y1 + crop_h

    # Clamp and track offsets for padding
    src_x1 = max(0, r_x1)
    src_y1 = max(0, r_y1)
    src_x2 = min(fw, r_x2)
    src_y2 = min(fh, r_y2)

    dst_x1 = src_x1 - r_x1
    dst_y1 = src_y1 - r_y1

    canvas = np.zeros((crop_h, crop_w, 3), dtype=np.uint8)
    canvas[dst_y1:dst_y1 + (src_y2 - src_y1),
           dst_x1:dst_x1 + (src_x2 - src_x1)] = frame[src_y1:src_y2, src_x1:src_x2]
    return canvas


# ── Public API ────────────────────────────────────────────────────────────────

def extract_clip(
    video_path: str | Path,
    track: Track,
    output_path: Path,
    config: ClipConfig,
) -> Path | None:
    """
    Extract a player-following clip around the midpoint of a track.

    Each frame in the clip is cropped to a fixed window centered on the
    tracked player's position (from track.detections), so opponents in the
    same scene are cropped out.

    Returns
    -------
    Path | None
        Path to the written clip file, or None if extraction failed.
    """
    video_path = Path(video_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not track.detections:
        return None

    meta = get_video_metadata(video_path)
    midpoint_ts = (track.start_ts + track.end_ts) / 2.0
    half = config.duration_s / 2.0
    start_ts = max(0.0, midpoint_ts - half)
    end_ts = min(meta.duration_s, midpoint_ts + half)

    return _extract_clip_cv2_following(
        video_path, track, output_path, start_ts, end_ts, meta.fps, config
    )


def _extract_clip_cv2_following(
    video_path: Path,
    track: Track,
    output_path: Path,
    start_ts: float,
    end_ts: float,
    src_fps: float,
    config: ClipConfig,
) -> Path | None:
    """
    Extract clip, cropping each frame to a window that follows the tracked player.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.error(f"Cannot open video for clip extraction: {video_path}")
        return None

    context_pad = getattr(config, "context_pad_ratio", 1.5)
    frame_lookup = _build_frame_bbox_lookup(track)

    try:
        start_frame = int(round(start_ts * src_fps))
        end_frame   = int(round(end_ts   * src_fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Read first frame to determine crop window size
        ok, first = cap.read()
        if not ok:
            return None
        fh, fw = first.shape[:2]
        crop_h, crop_w = _compute_crop_window(track, fh, fw, context_pad)

        fourcc = cv2.VideoWriter_fourcc(*config.codec)
        writer = cv2.VideoWriter(str(output_path), fourcc, config.fps, (crop_w, crop_h))

        # Write first frame
        bbox = _nearest_bbox(frame_lookup, start_frame)
        if bbox is not None:
            writer.write(_crop_frame_to_player(first, bbox, crop_h, crop_w))
        else:
            # No detection near start — write black frame
            writer.write(np.zeros((crop_h, crop_w, 3), dtype=np.uint8))

        # Skip-stride to match output fps
        stride = max(1, int(round(src_fps / config.fps)))
        cur = start_frame + 1
        while cur < end_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, cur)
            ok, frame = cap.read()
            if not ok:
                break
            bbox = _nearest_bbox(frame_lookup, cur)
            if bbox is not None:
                writer.write(_crop_frame_to_player(frame, bbox, crop_h, crop_w))
            else:
                writer.write(np.zeros((crop_h, crop_w, 3), dtype=np.uint8))
            cur += stride

        writer.release()
    finally:
        cap.release()

    return output_path if output_path.exists() and output_path.stat().st_size > 0 else None


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None
