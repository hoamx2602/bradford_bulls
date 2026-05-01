"""
Annotation package builder.

End-to-end orchestrator: takes a video + config, runs detect_track + keyframe +
clip extraction, and writes a self-contained annotation package directory.

Output structure
----------------
    output/
    ├── manifest.json
    ├── tracks/
    │   ├── track_00001/
    │   │   ├── keyframe_sharpest_full.jpg
    │   │   ├── keyframe_sharpest_crop.jpg
    │   │   ├── keyframe_largest_full.jpg
    │   │   ├── ...
    │   │   ├── clip.mp4
    │   │   └── meta.json
    │   ├── track_00002/
    │   └── ...
    └── annotations.jsonl       # initially empty; reviewer writes labels here
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

from track_annotation.config import TrackAnnotationConfig
from track_annotation.pipeline.detect_track import Track, run_detect_track
from track_annotation.pipeline.keyframe import select_keyframes, write_keyframes
from track_annotation.pipeline.clip import extract_clip
from track_annotation.utils.logging import get_logger
from track_annotation.utils.video_io import get_video_metadata

log = get_logger(__name__)


def build_package(
    video_path: str | Path,
    output_dir: str | Path,
    config: TrackAnnotationConfig,
) -> Path:
    """
    Build a complete annotation package for a video.

    Parameters
    ----------
    video_path : str | Path
        Input video.
    output_dir : str | Path
        Where to write the package directory.
    config : TrackAnnotationConfig
        Pipeline config.

    Returns
    -------
    Path
        Path to the created package directory.
    """
    video_path = Path(video_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tracks_root = output_dir / "tracks"
    tracks_root.mkdir(exist_ok=True)

    log.info(f"Building annotation package")
    log.info(f"  video : {video_path}")
    log.info(f"  output: {output_dir}")

    # 1. Detection + tracking
    tracks = run_detect_track(video_path, config)

    # 2. Per-track: keyframes + clip + meta
    log.info(f"Writing {len(tracks)} track packages...")
    for track in tqdm(tracks, desc="track packages", unit="track"):
        _write_one_track_package(video_path, track, tracks_root, config)

    # 3. Manifest
    manifest = _build_manifest(video_path, output_dir, tracks, config)
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    log.info(f"  wrote manifest: {manifest_path}")

    # 4. Empty annotations.jsonl (reviewer appends to this)
    ann_path = output_dir / "annotations.jsonl"
    if not ann_path.exists():
        ann_path.touch()

    log.info(f"DONE. Package: {output_dir}")
    return output_dir


def _write_one_track_package(
    video_path: Path,
    track: Track,
    tracks_root: Path,
    config: TrackAnnotationConfig,
) -> None:
    """Write keyframes + clip + meta for one track."""
    track_dir = tracks_root / f"track_{track.track_id:05d}"
    track_dir.mkdir(parents=True, exist_ok=True)

    # Keyframes
    keyframes = select_keyframes(track, config.keyframe)
    if config.package.include_full_frames or config.package.include_crops:
        write_keyframes(
            video_path=video_path,
            track=track,
            keyframes=keyframes,
            out_dir=track_dir,
            config=config.keyframe,
            write_full_frame=config.package.include_full_frames,
            write_crop=config.package.include_crops,
        )

    # Clip
    if config.package.include_clips:
        clip_path = track_dir / "clip.mp4"
        extract_clip(video_path, track, clip_path, config.clip)

    # Meta
    meta = {
        "track_id": track.track_id,
        "num_frames": track.num_frames,
        "start_frame": track.start_frame,
        "end_frame": track.end_frame,
        "start_ts": track.start_ts,
        "end_ts": track.end_ts,
        "duration_s": track.duration_s,
        "mean_area_ratio": track.mean_area_ratio,
        "mean_confidence": track.mean_confidence,
        "keyframes": [
            {
                "strategy": kf.strategy,
                "frame_idx": kf.detection.frame_idx,
                "timestamp_s": kf.detection.timestamp_s,
                "bbox": list(kf.detection.bbox),
                "sharpness": kf.detection.sharpness,
                "area_ratio": kf.detection.area_ratio,
                "full_frame": kf.full_frame_path.name if kf.full_frame_path else None,
                "crop": kf.crop_path.name if kf.crop_path else None,
            }
            for kf in keyframes
        ],
        "detections": [asdict(d) for d in track.detections],
    }
    with open(track_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def _build_manifest(
    video_path: Path,
    output_dir: Path,
    tracks: list[Track],
    config: TrackAnnotationConfig,
) -> dict[str, Any]:
    meta = get_video_metadata(video_path)
    return {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "video": {
            "path": str(video_path),
            "filename": video_path.name,
            "fps": meta.fps,
            "frame_count": meta.frame_count,
            "duration_s": meta.duration_s,
            "width": meta.width,
            "height": meta.height,
        },
        "config": {
            "device_used": config.resolve_device(),
            "detection_weights": str(config.detection.weights),
            "detection_target_classes": config.detection.target_classes,
            "detection_conf": config.detection.conf,
            "tracker": config.tracking.tracker,
            "processing_fps": config.video.processing_fps,
            "min_track_duration_frames": config.tracking.min_track_duration_frames,
            "min_track_area_ratio": config.tracking.min_track_area_ratio,
            "keyframe_strategies": config.keyframe.strategies,
            "clip_duration_s": config.clip.duration_s,
        },
        "tracks_root": "tracks",
        "annotations_file": "annotations.jsonl",
        "logo_templates": {
            "dir": str(config.logo_templates.dir),
            "brand_ids": config.logo_templates.brand_ids,
        },
        "stats": {
            "num_tracks": len(tracks),
            "total_detections": sum(t.num_frames for t in tracks),
            "mean_track_duration_s": (
                sum(t.duration_s for t in tracks) / len(tracks) if tracks else 0.0
            ),
        },
    }
