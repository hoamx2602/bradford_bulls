"""Shared utilities for track annotation pipeline."""

from track_annotation.utils.logging import get_logger, setup_logging
from track_annotation.utils.video_io import VideoReader, get_video_metadata
from track_annotation.utils.geometry import (
    bbox_area_ratio,
    compute_sharpness,
    crop_with_padding,
    iou,
)

__all__ = [
    "get_logger",
    "setup_logging",
    "VideoReader",
    "get_video_metadata",
    "bbox_area_ratio",
    "compute_sharpness",
    "crop_with_padding",
    "iou",
]
