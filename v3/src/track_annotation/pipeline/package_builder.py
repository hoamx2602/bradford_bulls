"""
Annotation package builder.

End-to-end orchestrator: takes a video + config + brand registry + match context,
runs detect_track + keyframe + clip extraction, and writes a self-contained
annotation package directory.

Output structure
----------------
    output/
    ├── manifest.json            # match metadata, kit_context, active brands/variants
    ├── tracks/
    │   ├── track_00001/
    │   │   ├── keyframe_*_full.jpg
    │   │   ├── keyframe_*_crop.jpg
    │   │   ├── clip.mp4
    │   │   └── meta.json
    │   └── ...
    └── annotations.jsonl        # initially empty; reviewer writes labels here

The manifest pins the active brand pool for this match (per kit_context) so the
reviewer UI and exporters consume a consistent view.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

from track_annotation.config import (
    BrandRegistry,
    MatchContext,
    TrackAnnotationConfig,
)
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
    registry: BrandRegistry,
    match_context: MatchContext,
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
    registry : BrandRegistry
        Brand/variant registry (loaded from data/logo_templates/brands.yaml).
    match_context : MatchContext
        Per-match metadata; .kit_context determines which variants are active.

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
    log.info(f"  video       : {video_path}")
    log.info(f"  output      : {output_dir}")
    log.info(f"  kit_context : {match_context.kit_context}")
    active_brands = registry.list_active_brands(match_context.kit_context)
    active_variants = registry.list_active_variants(match_context.kit_context)
    log.info(
        f"  active pool : {len(active_brands)} brands, {len(active_variants)} variants"
    )

    # 1. Detection + tracking + filtering (uses ignore_regions and team_filter from match_context)
    tracks = run_detect_track(video_path, config, match_context=match_context)

    # 2. Per-track artifacts
    log.info(f"Writing {len(tracks)} track packages...")
    for track in tqdm(tracks, desc="track packages", unit="track"):
        _write_one_track_package(video_path, track, tracks_root, config)

    # 3. Manifest
    manifest = _build_manifest(
        video_path=video_path,
        output_dir=output_dir,
        tracks=tracks,
        config=config,
        registry=registry,
        match_context=match_context,
    )
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    log.info(f"  wrote manifest: {manifest_path}")

    # 4. Empty annotations.jsonl
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

    if config.package.include_clips:
        clip_path = track_dir / "clip.mp4"
        extract_clip(video_path, track, clip_path, config.clip)

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
        "mean_team_score": track.mean_team_score,
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
    (track_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def _build_manifest(
    video_path: Path,
    output_dir: Path,
    tracks: list[Track],
    config: TrackAnnotationConfig,
    registry: BrandRegistry,
    match_context: MatchContext,
) -> dict[str, Any]:
    meta = get_video_metadata(video_path)
    active_brands = registry.list_active_brands(match_context.kit_context)
    return {
        "schema_version": "2.0",
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
        "match_context": match_context.model_dump(),
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
            "registry_file": config.logo_templates.registry_file,
            # Snapshot of active brands & variants for this match — pinned at
            # build time so the reviewer / exporters never disagree with the
            # runtime registry even if it changes later.
            "active_brands": [
                {
                    "id": b.id,
                    "display_name": b.display_name,
                    "variants": [
                        {
                            "id": v.id,
                            "kit_contexts": v.kit_contexts,
                            "template_path": str(v.template_path),
                        }
                        for v in b.active_variants(match_context.kit_context)
                    ],
                }
                for b in active_brands
            ],
        },
        "stats": {
            "num_tracks": len(tracks),
            "total_detections": sum(t.num_frames for t in tracks),
            "mean_track_duration_s": (
                sum(t.duration_s for t in tracks) / len(tracks) if tracks else 0.0
            ),
        },
    }
