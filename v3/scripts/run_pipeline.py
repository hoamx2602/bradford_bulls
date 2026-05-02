#!/usr/bin/env python3
"""
End-to-end pipeline runner: video -> annotation package.

Usage:
    # With explicit kit context
    python scripts/run_pipeline.py --video data/videos/match.mp4 \\
        --config configs/person_tracking.yaml \\
        --output data/annotation_packages/match \\
        --kit-context home

    # Or with sidecar match metadata YAML
    python scripts/run_pipeline.py --video data/videos/match.mp4 \\
        --config configs/person_tracking.yaml \\
        --output data/annotation_packages/match \\
        --match-meta data/videos/match.meta.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make src/ importable when running as a script
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from track_annotation.config import (  # noqa: E402
    MatchContext,
    load_brand_registry,
    load_config,
    load_match_context,
)
from track_annotation.pipeline.package_builder import build_package  # noqa: E402
from track_annotation.utils.logging import setup_logging, get_logger  # noqa: E402

log = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Build track annotation package from a video")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--kit-context",
        choices=["home", "away", "special", "any"],
        default=None,
        help="Match kit context (required unless --match-meta given)",
    )
    parser.add_argument(
        "--match-meta",
        type=Path,
        default=None,
        help="Sidecar YAML with MatchContext fields",
    )
    parser.add_argument("--max-duration", type=float, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.max_duration is not None:
        cfg.video.max_duration_s = int(args.max_duration)

    # Resolve match context (sidecar YAML wins over CLI flag)
    match_ctx = load_match_context(args.match_meta) if args.match_meta else None
    if match_ctx is None:
        if args.kit_context is None:
            parser.error("Must provide either --kit-context or --match-meta")
        match_ctx = MatchContext(kit_context=args.kit_context)

    registry = load_brand_registry(cfg.logo_templates.registry_path())

    setup_logging(
        level=cfg.logging.level,
        log_to_file=cfg.logging.log_to_file,
        log_dir=cfg.logging.log_dir,
        run_name=args.video.stem,
    )

    build_package(
        video_path=args.video,
        output_dir=args.output,
        config=cfg,
        registry=registry,
        match_context=match_ctx,
    )


if __name__ == "__main__":
    main()
