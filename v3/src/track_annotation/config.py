"""
Pydantic config schema for track annotation pipeline.

Loads from YAML files (configs/*.yaml) with optional env var overrides.
Validates types and value ranges at load time so misconfigurations fail fast.

Brand model
-----------
Brands and variants are managed via a separate registry file
(data/logo_templates/brands.yaml). Each brand has 1+ variants; each variant is
gated by kit_context (home / away / special / any). At pipeline runtime, a
match's kit_context determines which variants are "active" — the annotator
sees only those variants in the reviewer UI, and the brand recognizer narrows
its template pool accordingly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ============================================================
# Brand registry models
# ============================================================


KitContext = Literal["home", "away", "special", "any"]


class Variant(BaseModel):
    """One visual appearance of a brand (e.g., aon_red for home kit)."""

    id: str = Field(..., min_length=1, description="Unique variant identifier")
    kit_contexts: list[KitContext] = Field(
        ..., min_length=1, description="Kit contexts this variant appears in"
    )
    template_path: Path = Field(..., description="Path relative to logo_templates dir")

    def is_active_for(self, kit_context: str) -> bool:
        """True if this variant should be shown for the given match kit_context."""
        return "any" in self.kit_contexts or kit_context in self.kit_contexts


class Brand(BaseModel):
    """A master sponsor brand with one or more visual variants."""

    id: str = Field(..., min_length=1, description="Master brand identifier")
    display_name: str = Field(..., min_length=1)
    variants: list[Variant] = Field(..., min_length=1)
    notes: str | None = None

    def active_variants(self, kit_context: str) -> list[Variant]:
        """Variants applicable for a given kit context."""
        return [v for v in self.variants if v.is_active_for(kit_context)]


class BrandRegistry(BaseModel):
    """Top-level container; one of these per project."""

    brands: list[Brand] = Field(..., min_length=1)

    def get_brand(self, brand_id: str) -> Brand:
        for b in self.brands:
            if b.id == brand_id:
                return b
        raise KeyError(f"Brand not found: {brand_id}")

    def get_variant(self, variant_id: str) -> tuple[Brand, Variant]:
        for b in self.brands:
            for v in b.variants:
                if v.id == variant_id:
                    return b, v
        raise KeyError(f"Variant not found: {variant_id}")

    def list_active_brands(self, kit_context: str) -> list[Brand]:
        """Brands that have at least one variant active in this kit context."""
        return [b for b in self.brands if b.active_variants(kit_context)]

    def list_active_variants(self, kit_context: str) -> list[Variant]:
        out: list[Variant] = []
        for b in self.brands:
            out.extend(b.active_variants(kit_context))
        return out

    def all_brand_ids(self) -> list[str]:
        return [b.id for b in self.brands]

    def all_variant_ids(self) -> list[str]:
        return [v.id for b in self.brands for v in b.variants]


def load_brand_registry(registry_path: str | Path) -> BrandRegistry:
    """Load a brand registry from YAML."""
    registry_path = Path(registry_path)
    if not registry_path.exists():
        raise FileNotFoundError(f"Brand registry not found: {registry_path}")
    raw = yaml.safe_load(registry_path.read_text())
    return BrandRegistry(**raw)


# ============================================================
# Match context
# ============================================================


class HSVColorSpec(BaseModel):
    """One HSV color range for team color matching."""

    name: str = Field(..., min_length=1)
    h: list[int] = Field(..., min_length=2, max_length=2, description="[h_min, h_max] in [0, 180]")
    s: list[int] = Field(default_factory=lambda: [0, 255], min_length=2, max_length=2)
    v: list[int] = Field(default_factory=lambda: [0, 255], min_length=2, max_length=2)


class TargetTeamSpec(BaseModel):
    """How to identify the target team in this match (color-based)."""

    primary_colors: list[HSVColorSpec] = Field(default_factory=list)
    min_team_score: float = Field(
        0.10,
        ge=0.0,
        le=1.0,
        description="Min mean color-match ratio over the track to keep it",
    )


class MatchContext(BaseModel):
    """
    Per-match metadata that influences variant selection AND filtering.

    Loaded from CLI flag (--kit-context) or a per-match YAML file
    (e.g., data/videos/match.meta.yaml) sitting next to the video.
    """

    kit_context: KitContext = Field(..., description="home | away | special | any")
    match_date: str | None = None
    opponent: str | None = None
    venue: str | None = None
    notes: str | None = None

    # ---- Filtering hints (optional but strongly recommended) ----
    target_team: TargetTeamSpec = Field(
        default_factory=TargetTeamSpec,
        description="If primary_colors is non-empty, tracks failing color match are dropped",
    )
    ignore_regions: list[list[float]] = Field(
        default_factory=list,
        description=(
            "Normalized [x1, y1, x2, y2] regions to mask out (UI overlays, "
            "channel logos, scoreboards). Detection centers inside these are dropped."
        ),
    )


def load_match_context(meta_path: str | Path | None) -> MatchContext | None:
    """Load match metadata from sidecar YAML, returns None if file does not exist."""
    if meta_path is None:
        return None
    meta_path = Path(meta_path)
    if not meta_path.exists():
        return None
    raw = yaml.safe_load(meta_path.read_text())
    return MatchContext(**raw)


# ============================================================
# Sub-config models
# ============================================================


class VideoConfig(BaseModel):
    processing_fps: float = Field(5.0, ge=0.5, le=60.0)
    max_duration_s: int | None = Field(None, ge=1)


class DetectionConfig(BaseModel):
    weights: Path
    target_classes: list[int] = Field([0])
    conf: float = Field(0.25, ge=0.0, le=1.0)
    iou: float = Field(0.45, ge=0.0, le=1.0)
    imgsz: int = Field(1280, ge=320, le=1920)
    half: bool = True
    max_det: int = Field(100, ge=1, le=1000)


class TrackingConfig(BaseModel):
    tracker: str = Field("botsort.yaml")
    min_track_duration_frames: int = Field(5, ge=1)
    min_track_area_ratio: float = Field(0.0005, ge=0.0, le=1.0)
    persist: bool = True

    @field_validator("tracker")
    @classmethod
    def validate_tracker(cls, v: str) -> str:
        if v not in {"botsort.yaml", "bytetrack.yaml"}:
            raise ValueError(f"Tracker must be 'botsort.yaml' or 'bytetrack.yaml', got {v}")
        return v


class KeyframeConfig(BaseModel):
    num_per_track: int = Field(3, ge=1, le=10)
    strategies: list[Literal["sharpest", "largest", "midpoint", "first", "last"]] = Field(
        default_factory=lambda: ["sharpest", "largest", "midpoint"]
    )
    pad_ratio: float = Field(0.15, ge=0.0, le=1.0)
    min_crop_size: int = Field(96, ge=16)


class ClipConfig(BaseModel):
    duration_s: float = Field(2.0, gt=0.0, le=30.0)
    fps: int = Field(30, ge=1, le=120)
    codec: str = Field("mp4v")
    use_ffmpeg: bool = False


class PackageConfig(BaseModel):
    include_full_frames: bool = True
    include_crops: bool = True
    include_clips: bool = True
    manifest_format: Literal["json", "yaml"] = "json"


class LogoTemplatesConfig(BaseModel):
    """Path to brand registry; the registry itself is loaded separately via
    load_brand_registry() because it is large."""

    dir: Path = Field(..., description="Logo templates directory")
    registry_file: str = Field(
        "brands.yaml", description="Brand registry filename inside `dir`"
    )

    def registry_path(self) -> Path:
        return self.dir / self.registry_file


class LoggingConfig(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_to_file: bool = True
    log_dir: Path = Path("logs")


# ============================================================
# Top-level config
# ============================================================


class TrackAnnotationConfig(BaseSettings):
    """
    Top-level pipeline config.

    Env var overrides use prefix TRACK_ANN__ with __ as nested separator:
        TRACK_ANN__DETECTION__CONF=0.3 ...
    """

    model_config = SettingsConfigDict(
        env_prefix="TRACK_ANN__",
        env_nested_delimiter="__",
        case_sensitive=False,
    )

    device: str = Field("auto")
    video: VideoConfig = Field(default_factory=VideoConfig)
    detection: DetectionConfig
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    keyframe: KeyframeConfig = Field(default_factory=KeyframeConfig)
    clip: ClipConfig = Field(default_factory=ClipConfig)
    package: PackageConfig = Field(default_factory=PackageConfig)
    logo_templates: LogoTemplatesConfig
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        if torch.cuda.is_available():
            return "cuda:0"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"


# ============================================================
# Config loader
# ============================================================


def load_config(config_path: str | Path) -> TrackAnnotationConfig:
    """Load pipeline config from YAML."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return TrackAnnotationConfig(**raw)
