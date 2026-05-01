"""
Per-track video clip extraction.

For each track, extract a short video clip (default 2s) centered on the track
midpoint. Clip is used by the reviewer UI so the annotator can perceive the
logo via temporal integration (ý tưởng 3).

Two backends:
  - cv2 VideoWriter (default; pure Python, no extra deps)
  - ffmpeg subprocess (better quality, requires ffmpeg in PATH)
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2

from track_annotation.config import ClipConfig
from track_annotation.pipeline.detect_track import Track
from track_annotation.utils.logging import get_logger
from track_annotation.utils.video_io import VideoReader, get_video_metadata

log = get_logger(__name__)


def extract_clip(
    video_path: str | Path,
    track: Track,
    output_path: Path,
    config: ClipConfig,
) -> Path | None:
    """
    Extract a clip around the midpoint of a track.

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

    if config.use_ffmpeg and _ffmpeg_available():
        return _extract_clip_ffmpeg(video_path, output_path, start_ts, end_ts - start_ts)

    return _extract_clip_cv2(video_path, output_path, start_ts, end_ts, meta.fps, config)


def _extract_clip_cv2(
    video_path: Path,
    output_path: Path,
    start_ts: float,
    end_ts: float,
    src_fps: float,
    config: ClipConfig,
) -> Path | None:
    """Extract clip using cv2 VideoWriter."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.error(f"Cannot open video for clip extraction: {video_path}")
        return None

    try:
        start_frame = int(round(start_ts * src_fps))
        end_frame = int(round(end_ts * src_fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Read first frame to get dimensions
        ok, first = cap.read()
        if not ok:
            return None
        h, w = first.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*config.codec)
        writer = cv2.VideoWriter(str(output_path), fourcc, config.fps, (w, h))
        writer.write(first)

        # Skip-stride to match output fps
        stride = max(1, int(round(src_fps / config.fps)))
        cur = start_frame + 1
        while cur < end_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, cur)
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
            cur += stride

        writer.release()
    finally:
        cap.release()

    return output_path if output_path.exists() and output_path.stat().st_size > 0 else None


def _extract_clip_ffmpeg(
    video_path: Path,
    output_path: Path,
    start_ts: float,
    duration_s: float,
) -> Path | None:
    """Extract clip via ffmpeg (better quality, requires ffmpeg)."""
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel", "error",
        "-ss", f"{start_ts:.3f}",
        "-i", str(video_path),
        "-t", f"{duration_s:.3f}",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-an",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        log.error(f"ffmpeg failed: {e}")
        return None
    return output_path if output_path.exists() else None


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None
