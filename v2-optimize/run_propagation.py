#!/usr/bin/env python3
"""
Bradford Bulls v2 — Temporal Annotation Propagation (Phase 1b).

After manually annotating frames on Roboflow and exporting YOLO format labels,
run this script to propagate annotations to neighboring frames.

This creates ~5x more training data with REAL motion blur.

Usage:
    python run_propagation.py \
        --video /path/to/video.mp4 \
        --frames-dir /path/to/exported/images \
        --labels-dir /path/to/exported/labels \
        --output-dir /path/to/output
"""

import sys
import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.propagation import run_propagation
from src import config


def main():
    parser = argparse.ArgumentParser(
        description="Bradford Bulls v2 — Annotation Propagation (Phase 1b)"
    )
    parser.add_argument("--video", type=str, required=True,
                        help="Path to the original video file")
    parser.add_argument("--frames-dir", type=str, required=True,
                        help="Directory with annotated frame images")
    parser.add_argument("--labels-dir", type=str, required=True,
                        help="Directory with YOLO .txt label files")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for propagated dataset")
    parser.add_argument("--radius", type=int, default=config.PROPAGATION_RADIUS,
                        help=f"Propagation radius ±N frames (default: {config.PROPAGATION_RADIUS})")
    parser.add_argument("--use-yolo", action="store_true",
                        help="Use YOLO for torso sharpness (slower but more accurate)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Bradford Bulls v2 — Annotation Propagation (Phase 1b)")
    print("=" * 60)

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"❌ Video not found: {video_path}")
        sys.exit(1)

    frames_dir = Path(args.frames_dir)
    labels_dir = Path(args.labels_dir)
    if not frames_dir.exists():
        print(f"❌ Frames directory not found: {frames_dir}")
        sys.exit(1)
    if not labels_dir.exists():
        print(f"❌ Labels directory not found: {labels_dir}")
        sys.exit(1)

    yolo_model = None
    device = "cpu"
    if args.use_yolo:
        import torch
        from ultralytics import YOLO
        yolo_model = YOLO(config.PERSON_MODEL)
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        print(f"  YOLO loaded on {device}")

    print(f"\n  Video:     {video_path.name}")
    print(f"  Frames:    {frames_dir}")
    print(f"  Labels:    {labels_dir}")
    print(f"  Output:    {args.output_dir}")
    print(f"  Radius:    ±{args.radius} frames")
    print(f"  YOLO:      {'Yes' if args.use_yolo else 'No (using Laplacian fallback)'}")

    stats = run_propagation(
        video_path=str(video_path),
        frames_dir=str(frames_dir),
        labels_dir=str(labels_dir),
        output_dir=str(args.output_dir),
        yolo_model=yolo_model,
        device=device,
        radius=args.radius,
    )

    if stats:
        print("\n🎉 Phase 1b complete! Next steps:")
        print(f"   1. Dataset ready at: {args.output_dir}")
        print(f"   2. Total frames: {stats['total_frames']}")
        print("   3. Use this dataset for crop-based YOLO training (Phase 2)")


if __name__ == "__main__":
    main()
