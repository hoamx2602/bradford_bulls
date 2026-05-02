"""
CLI entry point for the track annotation pipeline.

Usage
-----
    # Build annotation package from video — kit_context REQUIRED
    python -m track_annotation.cli build-package \\
        --video data/videos/match.mp4 \\
        --config configs/person_tracking.yaml \\
        --output data/annotation_packages/match \\
        --kit-context home

    # ... or load match metadata from a sidecar YAML
    python -m track_annotation.cli build-package \\
        --video data/videos/match.mp4 \\
        --config configs/person_tracking.yaml \\
        --output data/annotation_packages/match \\
        --match-meta data/videos/match.meta.yaml

    # Export to YOLO format (brand-level multi-class by default)
    python -m track_annotation.cli export \\
        --package data/annotation_packages/match \\
        --format yolo \\
        --output data/yolo_dataset \\
        --class-mode brand

    # Export to CVAT XML
    python -m track_annotation.cli export \\
        --package data/annotation_packages/match \\
        --format cvat \\
        --output data/cvat_annotations.xml

    # List brands & variants for a kit context
    python -m track_annotation.cli brands --kit-context home
"""

from __future__ import annotations

import json
from pathlib import Path

import click

from track_annotation.config import (
    MatchContext,
    load_brand_registry,
    load_config,
    load_match_context,
)
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
@click.option("--video", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--config", "config_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--output", required=True, type=click.Path(file_okay=False))
@click.option(
    "--kit-context",
    type=click.Choice(["home", "away", "special", "any"]),
    default=None,
    help="Match kit context. Required unless --match-meta provides one.",
)
@click.option(
    "--match-meta",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Sidecar YAML with MatchContext fields (overrides --kit-context if both given)",
)
@click.option("--max-duration", type=float, default=None, help="Limit to N seconds (testing)")
def cmd_build_package(
    video: str,
    config_path: str,
    output: str,
    kit_context: str | None,
    match_meta: str | None,
    max_duration: float | None,
):
    """Build annotation package from a video."""
    cfg = load_config(config_path)
    if max_duration is not None:
        cfg.video.max_duration_s = int(max_duration)

    # Resolve match context
    match_ctx = load_match_context(match_meta)
    if match_ctx is None:
        if kit_context is None:
            raise click.UsageError(
                "Must provide either --kit-context or --match-meta"
            )
        match_ctx = MatchContext(kit_context=kit_context)
    elif kit_context is not None and match_ctx.kit_context != kit_context:
        log.warning(
            f"--kit-context={kit_context} ignored; --match-meta has kit_context={match_ctx.kit_context}"
        )

    # Load brand registry
    registry = load_brand_registry(cfg.logo_templates.registry_path())

    setup_logging(
        level=cfg.logging.level,
        log_to_file=cfg.logging.log_to_file,
        log_dir=cfg.logging.log_dir,
        run_name=Path(video).stem,
    )
    log.info(f"Loaded config: {config_path}")
    log.info(f"Loaded registry: {len(registry.brands)} brands total")

    build_package(
        video_path=video,
        output_dir=output,
        config=cfg,
        registry=registry,
        match_context=match_ctx,
    )


# ============================================================
# export
# ============================================================


@cli.command("export")
@click.option("--package", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--format", "fmt", required=True, type=click.Choice(["yolo", "cvat"]))
@click.option("--output", required=True, type=click.Path())
@click.option("--val-ratio", type=float, default=0.15, help="Val split ratio (yolo only)")
@click.option("--seed", type=int, default=42)
@click.option(
    "--class-mode",
    type=click.Choice(["single", "brand", "variant"]),
    default="brand",
    help=(
        "YOLO class scheme: "
        "single = one 'logo' class for Stage A | "
        "brand = brand-level (recommended for most training) | "
        "variant = variant-level (for fine-grained models)"
    ),
)
@click.option("--label-name", default="logo", help="Label name in CVAT export")
def cmd_export(
    package: str,
    fmt: str,
    output: str,
    val_ratio: float,
    seed: int,
    class_mode: str,
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
            class_mode=class_mode,  # type: ignore[arg-type]
        )
    elif fmt == "cvat":
        export_to_cvat(
            package_dir=package,
            output_path=output,
            label_name=label_name,
        )


# ============================================================
# brands
# ============================================================


@cli.command("brands")
@click.option(
    "--registry",
    type=click.Path(exists=True, dir_okay=False),
    default="data/logo_templates/brands.yaml",
)
@click.option(
    "--kit-context",
    type=click.Choice(["home", "away", "special", "any"]),
    default=None,
    help="If set, only show variants active for this kit context",
)
def cmd_brands(registry: str, kit_context: str | None):
    """List brands and variants in a registry."""
    setup_logging(level="WARNING", log_to_file=False)
    reg = load_brand_registry(registry)
    if kit_context:
        click.echo(f"Active brands for kit_context={kit_context!r}:\n")
    else:
        click.echo(f"All brands ({len(reg.brands)} total):\n")

    for brand in reg.brands:
        if kit_context:
            variants = brand.active_variants(kit_context)
            if not variants:
                continue
        else:
            variants = brand.variants
        click.echo(f"  {brand.id}  ({brand.display_name})")
        for v in variants:
            ctx = ", ".join(v.kit_contexts)
            click.echo(f"    └─ {v.id}  [{ctx}]  →  {v.template_path}")


# ============================================================
# inspect
# ============================================================


@cli.command("inspect")
@click.option("--package", required=True, type=click.Path(exists=True, file_okay=False))
def cmd_inspect(package: str):
    """Print quick stats about an annotation package."""
    setup_logging(level="INFO", log_to_file=False)
    pkg = Path(package)
    manifest = json.loads((pkg / "manifest.json").read_text())
    click.echo(f"Package: {pkg}")
    click.echo(f"  schema_version : {manifest.get('schema_version', '1.0')}")
    click.echo(f"  video          : {manifest['video']['filename']} "
               f"({manifest['video']['duration_s']:.1f}s)")
    click.echo(f"  resolution     : {manifest['video']['width']}x{manifest['video']['height']}")
    mc = manifest.get("match_context", {})
    click.echo(f"  kit_context    : {mc.get('kit_context', 'N/A')}")
    if mc.get("opponent"):
        click.echo(f"  opponent       : {mc['opponent']}")
    click.echo(f"  active brands  : {len(manifest['logo_templates'].get('active_brands', []))}")
    click.echo(f"  num_tracks     : {manifest['stats']['num_tracks']}")
    click.echo(f"  total_dets     : {manifest['stats']['total_detections']}")
    click.echo(f"  mean_dur_s     : {manifest['stats']['mean_track_duration_s']:.2f}")
    ann_path = pkg / "annotations.jsonl"
    if ann_path.exists():
        n_ann = sum(1 for _ in ann_path.open())
        click.echo(f"  annotations    : {n_ann} rows")


if __name__ == "__main__":
    cli()
