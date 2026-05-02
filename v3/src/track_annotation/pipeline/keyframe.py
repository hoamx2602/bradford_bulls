"""
Keyframe selection per track.

For each Track, pick the N most informative frames using configurable strategies:
  - sharpest: highest Laplacian variance on bbox crop
  - largest: largest bbox area
  - midpoint: frame at middle of track timeline
  - first / last: temporal endpoints

The selected keyframes drive what the human annotator sees in the reviewer UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2

from track_annotation.config import KeyframeConfig
from track_annotation.pipeline.detect_track import Detection, Track
from track_annotation.utils.geometry import crop_with_padding, draw_highlighted_bbox
from track_annotation.utils.logging import get_logger
from track_annotation.utils.video_io import VideoReader

log = get_logger(__name__)


@dataclass
class Keyframe:
    """One selected keyframe of a track."""

    strategy: str                   # "sharpest" | "largest" | "midpoint" | ...
    detection: Detection
    full_frame_path: Path | None = None
    crop_path: Path | None = None


def select_keyframes(track: Track, config: KeyframeConfig) -> list[Keyframe]:
    """
    Choose N keyframes from a track per the configured strategies.

    Returns up to config.num_per_track keyframes, deduplicated by frame_idx
    (so two strategies pointing at the same frame yield one Keyframe with the
    earlier strategy label).
    """
    if not track.detections:
        return []

    # Build candidate index -> strategy map preserving order
    candidates: list[tuple[str, Detection]] = []
    seen_frames: set[int] = set()

    strategies = config.strategies[: config.num_per_track]
    for strat in strategies:
        det = _pick_by_strategy(track, strat)
        if det is None or det.frame_idx in seen_frames:
            continue
        candidates.append((strat, det))
        seen_frames.add(det.frame_idx)

    return [Keyframe(strategy=s, detection=d) for s, d in candidates]


def _pick_by_strategy(track: Track, strategy: str) -> Detection | None:
    if not track.detections:
        return None
    if strategy == "sharpest":
        return max(track.detections, key=lambda d: d.sharpness)
    if strategy == "largest":
        return max(track.detections, key=lambda d: d.area_ratio)
    if strategy == "midpoint":
        return track.detections[len(track.detections) // 2]
    if strategy == "first":
        return track.detections[0]
    if strategy == "last":
        return track.detections[-1]
    raise ValueError(f"Unknown keyframe strategy: {strategy}")


def write_keyframes(
    video_path: str | Path,
    track: Track,
    keyframes: list[Keyframe],
    out_dir: Path,
    config: KeyframeConfig,
    write_full_frame: bool = True,
    write_crop: bool = True,
) -> None:
    """
    Read the actual keyframe images from the video and save to disk.

    For each Keyframe, optionally writes:
      - {out_dir}/keyframe_{strategy}_full.jpg : full original frame with bbox drawn
      - {out_dir}/keyframe_{strategy}_crop.jpg : padded bbox crop, upscaled if small

    Mutates Keyframe.full_frame_path and .crop_path.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read all required frames in one VideoReader open (cheap reseek per frame)
    with VideoReader(video_path) as reader:
        for kf in keyframes:
            frame = reader.read_at(kf.detection.frame_idx)
            if frame is None:
                log.warning(
                    f"track {track.track_id}: failed to read frame {kf.detection.frame_idx}"
                )
                continue

            if write_full_frame:
                annotated = draw_highlighted_bbox(
                    frame,
                    kf.detection.bbox,
                    color=(0, 255, 0),
                    thickness=4,
                    dim_outside=0.55,
                    label=f"track {track.track_id} · {kf.strategy}",
                )
                p = out_dir / f"keyframe_{kf.strategy}_full.jpg"
                cv2.imwrite(str(p), annotated)
                kf.full_frame_path = p

            if write_crop:
                crop, _ = crop_with_padding(
                    frame,
                    kf.detection.bbox,
                    pad_ratio=config.pad_ratio,
                    min_size=config.min_crop_size,
                )
                if crop.size > 0:
                    p = out_dir / f"keyframe_{kf.strategy}_crop.jpg"
                    cv2.imwrite(str(p), crop)
                    kf.crop_path = p


# Note: bbox drawing now lives in utils.geometry.draw_highlighted_bbox so it can
# also be reused by the reviewer UI and exporters if needed.
