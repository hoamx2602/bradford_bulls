"""
Bradford Bulls — Track Annotation Pipeline (v3)

Track-level annotation pipeline implementing ý tưởng 2 + 3 from the master plan.
Replaces frame-level annotation with track-level paradigm to handle motion-blur
on low-resolution sports video.

Public API
----------
    from track_annotation import build_package, load_config

    cfg = load_config("configs/person_tracking.yaml")
    build_package(video="match.mp4", output="package_dir", config=cfg)
"""

from track_annotation.config import TrackAnnotationConfig, load_config
from track_annotation.pipeline.package_builder import build_package

__version__ = "0.1.0"
__all__ = ["TrackAnnotationConfig", "load_config", "build_package"]
