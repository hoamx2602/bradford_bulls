"""
Track-level annotation pipeline for Bradford Bulls logo exposure system.

Implements ý tưởng 2 (track-level annotation) + ý tưởng 3 (video clip review)
from the master plan. Generates an annotation package per video so that human
annotators only label one BRAND PER TRACK instead of per-frame, addressing
the motion-blur problem on low-resolution sports footage.

High-level workflow
-------------------
    video.mp4
        |
        v
    [detect_track]   class-agnostic detection + BoT-SORT tracking
        |
        v
    [keyframe]       3 best-evidence keyframes per track
        |              (sharpest / largest / midpoint)
        v
    [clip]           2-second video clip around each track
        |
        v
    [package]        assemble annotation package (folder + manifest.json)
        |
        v
    [reviewer_app]   Streamlit UI: annotator confirms brand per track
                     -> exports annotations.jsonl
                     -> can be converted to YOLO / CVAT / Roboflow format
"""

from .config import TrackAnnotationConfig
from .detect_track import run_detect_track
from .keyframe import select_keyframes
from .clip import extract_clip
from .package_builder import build_annotation_package

__all__ = [
    "TrackAnnotationConfig",
    "run_detect_track",
    "select_keyframes",
    "extract_clip",
    "build_annotation_package",
]
