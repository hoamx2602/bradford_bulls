"""
Video I/O utilities.

Provides a thin wrapper over OpenCV VideoCapture with FPS subsampling, seek-by-time,
and frame indexing helpers tailored for the track annotation pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoMetadata:
    """Container for video file metadata."""

    path: Path
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0


def get_video_metadata(video_path: str | Path) -> VideoMetadata:
    """
    Read metadata from video file without decoding frames.

    Parameters
    ----------
    video_path : str | Path
        Path to video file.

    Returns
    -------
    VideoMetadata

    Raises
    ------
    FileNotFoundError
        If video file does not exist.
    RuntimeError
        If video cannot be opened.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()

    return VideoMetadata(path=video_path, fps=fps, frame_count=frame_count, width=width, height=height)


class VideoReader:
    """
    Iterable video reader with optional FPS subsampling.

    Usage
    -----
        with VideoReader("match.mp4", target_fps=5.0) as reader:
            for frame_idx, ts, frame in reader:
                # frame_idx: original frame index in source video
                # ts: timestamp in seconds
                # frame: BGR ndarray (H, W, 3)
                ...
    """

    def __init__(
        self,
        video_path: str | Path,
        target_fps: float | None = None,
        max_duration_s: float | None = None,
    ):
        self.video_path = Path(video_path)
        self.target_fps = target_fps
        self.max_duration_s = max_duration_s
        self._cap: cv2.VideoCapture | None = None
        self._meta: VideoMetadata | None = None

    def __enter__(self) -> "VideoReader":
        self._cap = cv2.VideoCapture(str(self.video_path))
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")
        self._meta = get_video_metadata(self.video_path)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def metadata(self) -> VideoMetadata:
        if self._meta is None:
            raise RuntimeError("VideoReader must be used as context manager")
        return self._meta

    def _frame_step(self) -> int:
        """How many source frames to skip between yielded frames."""
        if self.target_fps is None or self.target_fps >= self.metadata.fps:
            return 1
        return max(1, int(round(self.metadata.fps / self.target_fps)))

    def __iter__(self) -> Iterator[tuple[int, float, np.ndarray]]:
        if self._cap is None or self._meta is None:
            raise RuntimeError("VideoReader must be used as context manager")

        step = self._frame_step()
        max_frames = self._meta.frame_count
        if self.max_duration_s is not None:
            max_frames = min(max_frames, int(self.max_duration_s * self._meta.fps))

        frame_idx = 0
        while frame_idx < max_frames:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = self._cap.read()
            if not ok or frame is None:
                break
            ts = frame_idx / self._meta.fps
            yield frame_idx, ts, frame
            frame_idx += step

    def read_at(self, frame_idx: int) -> np.ndarray | None:
        """
        Read a specific frame by index. Returns None if read fails.

        Note: random access via cv2.CAP_PROP_POS_FRAMES is not always frame-accurate
        for all containers/codecs (especially without keyframe at that position). For
        critical use, prefer ffmpeg-based extraction.
        """
        if self._cap is None:
            raise RuntimeError("VideoReader must be used as context manager")
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = self._cap.read()
        return frame if ok else None
