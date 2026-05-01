"""
CLI entry point for the track annotation pipeline.

Usage
-----
    # Build annotation package from video
    python -m track_annotation.cli build-package \\
        --video data/videos/match.mp4 \\
        --config configs/person_tracking.yaml \\
        --output data/annotation_packages/match

    # Export annotated package to YOLO training format
    python -m track_annotation.cli export \\
        --package data/annotation_packages/match \\
        --format yolo \\
        --output data/yolo_dataset \\
        --single-class

    # Export to CVAT XML
    python -m track_annotation.cli export \\
        --package data/annotation_packages/match \\
        --format cvat \\
        --output data/cvat_annotations.xml
"""

from __future__ import annotations

from pathlib import Path

import click

from track_annotation.config import load_config
from track_annotation.exporters.cvat import export_to_cvat
from track_annotation.exporters.yolo import export_to_yolo
from track_annotation.pipeline.package_builder import build_package
from track_annotation.utils.logging import setup_logging, get_logger

log = get_logger(__name__)


@click.group()
def cli():
    """Bradford Bulls track annotation pipeline."""
    pass


# ============================================================
# build-package
# ============================================================


@cli.command("build-package")
@click.option("--video", required=True, type=click.Path(exists=True, dir_okay=False), help="Input video file")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True, dir_okay=False), help="YAML config")
@click.option("--output", required=True, type=click.Path(file_okay=False), help="Output package directory")
@click.option("--max-duration", type=float, default=None, help="Limit processing to N seconds (testing)")
def cmd_build_package(video: str, config_path: str, output: str, max_duration: float | None):
    """Build annotation package from a video."""
    cfg = load_config(config_path)
    if max_duration is not None:
        cfg.video.max_duration_s = int(max_duration)
    setup_logging(
        level=cfg.logging.level,
        log_to_file=cfg.logging.log_to_file,
        log_dir=cfg.logging.log_dir,
        run_name=Path(video).stem,
    )
    log.info(f"Loaded config: {config_path}")
    build_package(video_path=video, output_dir=output, config=cfg)


# ============================================================
# export
# ============================================================


@cli.command("export")
@click.option("--package", required=True, type=click.Path(exists=True, file_okay=False), help="Annotation package dir")
@click.option(
    "--format",
    "fmt",
    required=True,
    type=click.Choice(["yolo", "cvat"]),
    help="Export format",
)
@click.option("--output", required=True, type=click.Path(), help="Output path/dir (depends on format)")
@click.option("--val-ratio", type=float, default=0.15, help="Val split ratio (yolo only)")
@click.option("--seed", type=int, default=42, help="RNG seed (yolo only)")
@click.option(
    "--single-class",
    is_flag=True,
    default=False,
    help="Export as single 'logo' class for Stage A training (yolo only)",
)
@click.option("--label-name", default="logo", help="Label name in CVAT export (cvat only)")
def cmd_export(
    package: str,
    fmt: str,
    output: str,
    val_ratio: float,
    seed: int,
    single_class: bool,
    label_name: str,
):
    """Export annotation package to a downstream format."""
    setup_logging(level="INFO", log_to_file=False)
    if fmt == "yolo":
        export_to_yolo(
            package_dir=package,
            output_dir=output,
            val_ratio=val_ratio,
            seed=seed,
            single_class=single_class,
        )
    elif fmt == "cvat":
        export_to_cvat(
            package_dir=package,
            output_path=output,
            label_name=label_name,
        )


# ============================================================
# inspect
# ============================================================


@cli.command("inspect")
@click.option("--package", required=True, type=click.Path(exists=True, file_okay=False))
def cmd_inspect(package: str):
    """Print quick stats about an annotation package."""
    import json
    setup_logging(level="INFO", log_to_file=False)
    pkg = Path(package)
    manifest = json.loads((pkg / "manifest.json").read_text())
    click.echo(f"Package: {pkg}")
    click.echo(f"  video       : {manifest['video']['filename']} ({manifest['video']['duration_s']:.1f}s)")
    click.echo(f"  resolution  : {manifest['video']['width']}x{manifest['video']['height']}")
    click.echo(f"  num_tracks  : {manifest['stats']['num_tracks']}")
    click.echo(f"  total_dets  : {manifest['stats']['total_detections']}")
    click.echo(f"  mean_dur_s  : {manifest['stats']['mean_track_duration_s']:.2f}")
    ann_path = pkg / "annotations.jsonl"
    if ann_path.exists():
        n_ann = sum(1 for _ in ann_path.open())
        click.echo(f"  annotations : {n_ann} rows")


if __name__ == "__main__":
    cli()
