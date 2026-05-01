"""
Pydantic config schema for track annotation pipeline.

Loads from YAML files (configs/*.yaml) with optional env var overrides.
Validates types and value ranges at load time so misconfigurations fail fast.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ============================================================
# Sub-config models
# ============================================================


class VideoConfig(BaseModel):
    processing_fps: float = Field(5.0, ge=0.5, le=60.0, description="FPS to sample video at for tracking")
    max_duration_s: int | None = Field(None, ge=1, description="Max seconds to process; null = full video")


class DetectionConfig(BaseModel):
    weights: Path = Field(..., description="Path to YOLO .pt weights")
    target_classes: list[int] = Field([0], description="COCO class IDs to detect (0 = person)")
    conf: float = Field(0.25, ge=0.0, le=1.0)
    iou: float = Field(0.45, ge=0.0, le=1.0)
    imgsz: int = Field(1280, ge=320, le=1920)
    half: bool = Field(True, description="Use FP16 inference")
    max_det: int = Field(100, ge=1, le=1000)


class TrackingConfig(BaseModel):
    tracker: str = Field("botsort.yaml", description="Ultralytics tracker config")
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
    dir: Path
    brand_ids: list[str]


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

    Loads from YAML by default; env vars with prefix TRACK_ANN__ override (use __ as
    nested separator). Example:

        TRACK_ANN__DETECTION__CONF=0.3 python -m track_annotation.cli ...
    """

    model_config = SettingsConfigDict(
        env_prefix="TRACK_ANN__",
        env_nested_delimiter="__",
        case_sensitive=False,
    )

    device: str = Field("auto", description="auto | cuda | cuda:0 | mps | cpu")
    video: VideoConfig = Field(default_factory=VideoConfig)
    detection: DetectionConfig
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    keyframe: KeyframeConfig = Field(default_factory=KeyframeConfig)
    clip: ClipConfig = Field(default_factory=ClipConfig)
    package: PackageConfig = Field(default_factory=PackageConfig)
    logo_templates: LogoTemplatesConfig
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    def resolve_device(self) -> str:
        """Resolve 'auto' to actual device string."""
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
    """
    Load config from YAML file. Env vars with TRACK_ANN__ prefix override values.

    Parameters
    ----------
    config_path : str | Path
        Path to YAML config file.

    Returns
    -------
    TrackAnnotationConfig
        Validated config object.

    Raises
    ------
    FileNotFoundError
        If config file does not exist.
    pydantic.ValidationError
        If config values are invalid.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return TrackAnnotationConfig(**raw)
