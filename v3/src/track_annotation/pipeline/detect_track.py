"""
Detection + Multi-Object Tracking using Ultralytics YOLO + BoT-SORT.

Outputs a list of Track objects after applying:
  1. Ignore-region mask (UI overlays)
  2. Per-detection team color filter (mean per track)
  3. Min track duration / area thresholds

Subsequent pipeline modules (keyframe, clip, package_builder) consume these tracks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

from track_annotation.config import MatchContext, TrackAnnotationConfig
from track_annotation.pipeline.region_mask import bbox_in_any_region
from track_annotation.pipeline.team_filter import (
    parse_color_ranges,
    team_score,
)
from track_annotation.utils.geometry import Bbox, bbox_area_ratio, compute_sharpness
from track_annotation.utils.logging import get_logger
from track_annotation.utils.video_io import VideoReader, get_video_metadata

log = get_logger(__name__)


# ============================================================
# Data classes
# ============================================================


@dataclass
class Detection:
    frame_idx: int
    timestamp_s: float
    bbox: Bbox
    confidence: float
    class_id: int
    sharpness: float = 0.0
    area_ratio: float = 0.0
    team_score: float = 0.0  # 0..1 — mean color match for this detection's torso


@dataclass
class Track:
    track_id: int
    detections: list[Detection] = field(default_factory=list)

    @property
    def num_frames(self) -> int:
        return len(self.detections)

    @property
    def start_frame(self) -> int:
        return self.detections[0].frame_idx if self.detections else -1

    @property
    def end_frame(self) -> int:
        return self.detections[-1].frame_idx if self.detections else -1

    @property
    def start_ts(self) -> float:
        return self.detections[0].timestamp_s if self.detections else 0.0

    @property
    def end_ts(self) -> float:
        return self.detections[-1].timestamp_s if self.detections else 0.0

    @property
    def duration_s(self) -> float:
        return self.end_ts - self.start_ts

    @property
    def mean_area_ratio(self) -> float:
        if not self.detections:
            return 0.0
        return float(np.mean([d.area_ratio for d in self.detections]))

    @property
    def mean_confidence(self) -> float:
        if not self.detections:
            return 0.0
        return float(np.mean([d.confidence for d in self.detections]))

    @property
    def mean_team_score(self) -> float:
        if not self.detections:
            return 0.0
        return float(np.mean([d.team_score for d in self.detections]))


# ============================================================
# Main entry
# ============================================================


def run_detect_track(
    video_path: str | Path,
    config: TrackAnnotationConfig,
    match_context: MatchContext | None = None,
) -> list[Track]:
    """
    Run detection + tracking + filtering on a video.

    Filters applied (in order):
      1. Ignore-region mask (per detection)
      2. Track-level: min duration frames
      3. Track-level: min mean area ratio
      4. Track-level: min mean team_score (if MatchContext.target_team.primary_colors non-empty)
    """
    video_path = Path(video_path)
    log.info(f"Running detect+track on {video_path.name}")
    log.info(f"  weights: {config.detection.weights}")
    log.info(f"  device : {config.resolve_device()}")

    color_ranges = []
    ignore_regions: list[list[float]] = []
    min_team_score = 0.0
    if match_context is not None:
        ignore_regions = match_context.ignore_regions or []
        if match_context.target_team and match_context.target_team.primary_colors:
            color_ranges = parse_color_ranges(
                [c.model_dump() for c in match_context.target_team.primary_colors]
            )
            min_team_score = match_context.target_team.min_team_score
    log.info(f"  ignore_regions  : {len(ignore_regions)}")
    log.info(f"  team color ranges: {len(color_ranges)} (min_score={min_team_score})")

    meta = get_video_metadata(video_path)
    log.info(f"  video meta: {meta.width}x{meta.height} @ {meta.fps:.2f}fps, {meta.duration_s:.1f}s")

    model = YOLO(str(config.detection.weights))

    tracks: dict[int, Track] = {}
    device = config.resolve_device()
    frame_step = max(1, int(round(meta.fps / config.video.processing_fps)))
    max_frames = meta.frame_count
    if config.video.max_duration_s is not None:
        max_frames = min(max_frames, int(config.video.max_duration_s * meta.fps))
    expected_iters = max_frames // frame_step
    log.info(f"  processing every {frame_step}-th frame ({expected_iters} iterations)")

    n_dropped_overlay = 0
    with VideoReader(
        video_path,
        target_fps=config.video.processing_fps,
        max_duration_s=config.video.max_duration_s,
    ) as reader:
        pbar = tqdm(reader, total=expected_iters, desc="detect+track", unit="fr")
        for frame_idx, ts, frame in pbar:
            results = model.track(
                source=frame,
                persist=config.tracking.persist,
                tracker=config.tracking.tracker,
                conf=config.detection.conf,
                iou=config.detection.iou,
                imgsz=config.detection.imgsz,
                classes=config.detection.target_classes,
                half=config.detection.half,
                max_det=config.detection.max_det,
                device=device,
                verbose=False,
            )
            if not results:
                continue
            res = results[0]
            if res.boxes is None or res.boxes.id is None:
                continue

            xyxy = res.boxes.xyxy.cpu().numpy()
            ids = res.boxes.id.int().cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            cls = res.boxes.cls.int().cpu().numpy()
            frame_shape = (frame.shape[0], frame.shape[1])

            for i in range(len(ids)):
                box = tuple(float(v) for v in xyxy[i])

                # Filter 1 (per-detection): ignore-region mask
                if ignore_regions and bbox_in_any_region(box, frame_shape, ignore_regions):  # type: ignore[arg-type]
                    n_dropped_overlay += 1
                    continue

                tid = int(ids[i])
                ts_score = (
                    team_score(frame, box, color_ranges) if color_ranges else 1.0
                )
                # Sharpness measured on the TORSO region (where logos live)
                # so we don't reward frames where unrelated UI text/edges are crisp.
                det = Detection(
                    frame_idx=frame_idx,
                    timestamp_s=ts,
                    bbox=box,  # type: ignore[arg-type]
                    confidence=float(confs[i]),
                    class_id=int(cls[i]),
                    sharpness=compute_sharpness(frame, box, torso_only=True),  # type: ignore[arg-type]
                    area_ratio=bbox_area_ratio(box, frame_shape),  # type: ignore[arg-type]
                    team_score=ts_score,
                )
                if tid not in tracks:
                    tracks[tid] = Track(track_id=tid)
                tracks[tid].detections.append(det)

    log.info(f"  raw tracks            : {len(tracks)}")
    log.info(f"  detections in overlay : {n_dropped_overlay} (dropped)")

    filtered = _filter_tracks(tracks, config, min_team_score=min_team_score)
    log.info(f"  tracks after filtering: {len(filtered)}")

    return filtered


def _filter_tracks(
    tracks: dict[int, Track],
    config: TrackAnnotationConfig,
    min_team_score: float,
) -> list[Track]:
    """Drop tracks not meeting duration / area / team-score thresholds."""
    out: list[Track] = []
    n_dur = n_area = n_team = 0
    for tid in sorted(tracks.keys()):
        track = tracks[tid]
        if track.num_frames < config.tracking.min_track_duration_frames:
            n_dur += 1
            continue
        if track.mean_area_ratio < config.tracking.min_track_area_ratio:
            n_area += 1
            continue
        if min_team_score > 0.0 and track.mean_team_score < min_team_score:
            n_team += 1
            continue
        out.append(track)
    log.info(
        f"  filter dropped: duration={n_dur}, area={n_area}, team_score={n_team}"
    )
    return out
