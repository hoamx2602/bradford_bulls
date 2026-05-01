"""Core pipeline modules for track annotation."""

from track_annotation.pipeline.detect_track import Detection, Track, run_detect_track
from track_annotation.pipeline.keyframe import Keyframe, select_keyframes
from track_annotation.pipeline.clip import extract_clip
from track_annotation.pipeline.package_builder import build_package

__all__ = [
    "Detection",
    "Track",
    "Keyframe",
    "run_detect_track",
    "select_keyframes",
    "extract_clip",
    "build_package",
]
