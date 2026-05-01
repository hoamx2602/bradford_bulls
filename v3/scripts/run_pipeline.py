#!/usr/bin/env python3
"""
End-to-end pipeline runner: video -> annotation package -> ready for reviewer.

Equivalent to:
    python -m track_annotation.cli build-package --video VIDEO --config CFG --output OUT

But provides a single-script entry point that's friendlier for cron / batch jobs.

Usage:
    python scripts/run_pipeline.py --video data/videos/match.mp4 \\
        --config configs/person_tracking.yaml \\
        --output data/annotation_packages/match
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make src/ importable when running as a script
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from track_annotation.config import load_config  # noqa: E402
from track_annotation.pipeline.package_builder import build_package  # noqa: E402
from track_annotation.utils.logging import setup_logging  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Build track annotation package from a video")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path, help="YAML config path")
    parser.add_argument("--output", required=True, type=Path, help="Output package directory")
    parser.add_argument("--max-duration", type=float, default=None, help="Limit to N seconds (testing)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.max_duration is not None:
        cfg.video.max_duration_s = int(args.max_duration)

    setup_logging(
        level=cfg.logging.level,
        log_to_file=cfg.logging.log_to_file,
        log_dir=cfg.logging.log_dir,
        run_name=args.video.stem,
    )

    build_package(video_path=args.video, output_dir=args.output, config=cfg)


if __name__ == "__main__":
    main()
