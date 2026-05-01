"""
Detection + Multi-Object Tracking using Ultralytics YOLO + BoT-SORT.

Outputs a list of Track objects, each containing the per-frame Detection sequence
for one tracked instance over the video. Subsequent pipeline modules (keyframe,
clip, package_builder) consume these tracks.

Design notes
------------
- Uses ultralytics built-in tracking via `model.track(...)` with botsort.yaml.
  This handles BoT-SORT automatically including ReID embedding when available.
- Streams frames through the model rather than loading the whole video into
  memory; suitable for 2-hour matches at 1080p.
- Track filtering (min duration, min area) happens AFTER tracking so we can keep
  full provenance for debugging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

from track_annotation.config import TrackAnnotationConfig
from track_annotation.utils.geometry import Bbox, bbox_area_ratio, compute_sharpness
from track_annotation.utils.logging import get_logger
from track_annotation.utils.video_io import VideoReader, get_video_metadata

log = get_logger(__name__)


# ============================================================
# Data classes
# ============================================================


@dataclass
class Detection:
    """A single detection in one frame within a track."""

    frame_idx: int                  # Original source-video frame index
    timestamp_s: float              # Time in seconds from video start
    bbox: Bbox                      # xyxy in pixel coords of source frame
    confidence: float               # Detector confidence
    class_id: int                   # Class id from detector
    sharpness: float = 0.0          # Laplacian variance computed on bbox crop
    area_ratio: float = 0.0         # bbox area / frame area


@dataclass
class Track:
    """A continuous appearance of one object across multiple frames."""

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


# ============================================================
# Main entry
# ============================================================


def run_detect_track(
    video_path: str | Path,
    config: TrackAnnotationConfig,
) -> list[Track]:
    """
    Run detection + tracking on a video; return filtered list of Track objects.

    Parameters
    ----------
    video_path : str | Path
        Path to input video.
    config : TrackAnnotationConfig
        Pipeline config.

    Returns
    -------
    list[Track]
        Tracks meeting min duration and area thresholds, sorted by track_id.
    """
    video_path = Path(video_path)
    log.info(f"Running detect+track on {video_path.name}")
    log.info(f"  weights: {config.detection.weights}")
    log.info(f"  device: {config.resolve_device()}")

    meta = get_video_metadata(video_path)
    log.info(f"  video meta: {meta.width}x{meta.height} @ {meta.fps:.2f}fps, {meta.duration_s:.1f}s")

    model = YOLO(str(config.detection.weights))

    tracks: dict[int, Track] = {}

    # Use ultralytics' built-in tracking which handles BoT-SORT internally.
    # We pass the video path and let ultralytics stream frames; we also need
    # frame indices, so we use a manual loop and call model.track() per frame
    # with persist=True. This is slightly slower than streaming but keeps the
    # frame indexing consistent with our VideoReader subsampling.
    device = config.resolve_device()
    frame_step = max(1, int(round(meta.fps / config.video.processing_fps)))
    max_frames = meta.frame_count
    if config.video.max_duration_s is not None:
        max_frames = min(max_frames, int(config.video.max_duration_s * meta.fps))

    expected_iters = max_frames // frame_step
    log.info(f"  processing every {frame_step}-th frame ({expected_iters} iterations)")

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
                tid = int(ids[i])
                box = tuple(float(v) for v in xyxy[i])
                det = Detection(
                    frame_idx=frame_idx,
                    timestamp_s=ts,
                    bbox=box,  # type: ignore[arg-type]
                    confidence=float(confs[i]),
                    class_id=int(cls[i]),
                    sharpness=compute_sharpness(frame, box),  # type: ignore[arg-type]
                    area_ratio=bbox_area_ratio(box, frame_shape),  # type: ignore[arg-type]
                )
                if tid not in tracks:
                    tracks[tid] = Track(track_id=tid)
                tracks[tid].detections.append(det)

    log.info(f"  raw tracks: {len(tracks)}")

    filtered = _filter_tracks(tracks, config)
    log.info(f"  tracks after filtering: {len(filtered)}")

    return filtered


def _filter_tracks(
    tracks: dict[int, Track],
    config: TrackAnnotationConfig,
) -> list[Track]:
    """Drop tracks too short or too small per config thresholds."""
    out: list[Track] = []
    for tid in sorted(tracks.keys()):
        track = tracks[tid]
        if track.num_frames < config.tracking.min_track_duration_frames:
            continue
        if track.mean_area_ratio < config.tracking.min_track_area_ratio:
            continue
        out.append(track)
    return out
