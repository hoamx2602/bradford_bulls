"""
Bradford Bulls — Track Annotation Pipeline (v3)

Track-level annotation pipeline implementing ý tưởng 2 + 3 from the master plan.
Replaces frame-level annotation with track-level paradigm to handle motion-blur
on low-resolution sports video.

Public API
----------
    from track_annotation import build_package, load_config, load_brand_registry

    cfg = load_config("configs/person_tracking.yaml")
    registry = load_brand_registry(cfg.logo_templates.registry_path())
    build_package(
        video="match.mp4",
        output="package_dir",
        config=cfg,
        registry=registry,
        match_context=MatchContext(kit_context="home"),
    )
"""

from track_annotation.config import (
    Brand,
    BrandRegistry,
    HSVColorSpec,
    MatchContext,
    TargetTeamSpec,
    TrackAnnotationConfig,
    Variant,
    load_brand_registry,
    load_config,
    load_match_context,
)
from track_annotation.pipeline.package_builder import build_package

__version__ = "0.3.0"
__all__ = [
    "Brand",
    "BrandRegistry",
    "HSVColorSpec",
    "MatchContext",
    "TargetTeamSpec",
    "TrackAnnotationConfig",
    "Variant",
    "build_package",
    "load_brand_registry",
    "load_config",
    "load_match_context",
]
