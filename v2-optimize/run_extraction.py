#!/usr/bin/env python3
"""
Bradford Bulls v2 — Smart Frame Extraction (Phase 1).

Standalone script for extracting high-quality training frames.
Run locally or on Google Colab (with Google Drive mount).

Key improvements over v1:
  - Torso-focused sharpness scoring (logo region, not whole player)
  - Tiered quality system (Gold/Silver/Bronze)
  - Better scoring that prioritizes logo-readable frames

Usage:
    python run_extraction.py --video /path/to/video.mp4
    python run_extraction.py --video /path/to/video.mp4 --target-frames 400 --test
"""

import os
import sys
import argparse
import getpass
from pathlib import Path
from datetime import datetime

import torch
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

# Add src to path
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.overlay import detect_static_overlays
from src.calibration import collect_samples, show_samples, build_calibration
from src.pipeline import pass1_fast_scan, pass2_extract
from src.selection import select_by_quota, auto_target_frames, print_selection_summary
from src import config


def main():
    parser = argparse.ArgumentParser(
        description="Bradford Bulls v2 — Smart Frame Extraction"
    )
    parser.add_argument("--member", type=str, help="Your name (e.g., edward)")
    parser.add_argument("--video", type=str, help="Path to video file")
    parser.add_argument("--output-dir", type=str, help="Output directory for frames")
    parser.add_argument("--test", action="store_true", help="Test mode (first 2000 frames)")
    parser.add_argument("--target-frames", type=int, help="Target frames (default: auto)")
    parser.add_argument("--no-roboflow", action="store_true", help="Skip Roboflow upload")
    args = parser.parse_args()

    print("=" * 60)
    print("  Bradford Bulls v2 — Smart Frame Extraction")
    print("  (Torso Sharpness + Tiered Quality)")
    print("=" * 60)

    # ── 1. Config ──
    MEMBER_NAME = args.member or input("Your name (e.g. edward): ").strip().lower()
    VIDEO_PATH = args.video or input("Video path: ").strip()
    VIDEO_PATH = Path(VIDEO_PATH)

    if not VIDEO_PATH.exists():
        print(f"\n❌ ERROR: Video not found at {VIDEO_PATH}")
        sys.exit(1)

    TEST_MODE = args.test
    if not args.video:
        test_input = input("Test mode? (y/n, default=y): ").strip().lower()
        TEST_MODE = test_input != "n"

    # Output directory
    MATCH_ID = VIDEO_PATH.stem
    if args.output_dir:
        OUTPUT_BASE = Path(args.output_dir)
    else:
        OUTPUT_BASE = SCRIPT_DIR / "output"

    FRAMES_DIR = OUTPUT_BASE / "frames" / MATCH_ID
    METADATA_DIR = OUTPUT_BASE / "metadata"
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    TARGET_FRAMES = args.target_frames

    print(f"\n--- Config ---")
    print(f"  Member:     {MEMBER_NAME}")
    print(f"  Video:      {VIDEO_PATH.name} ({VIDEO_PATH.stat().st_size/1e6:.0f} MB)")
    print(f"  Mode:       {'TEST' if TEST_MODE else 'PRODUCTION'}")
    print(f"  Output:     {OUTPUT_BASE}")

    # ── 2. Device setup ──
    if torch.backends.mps.is_available():
        DEVICE = "mps"
        print("🚀 Apple Silicon GPU (MPS) detected!")
    elif torch.cuda.is_available():
        DEVICE = "cuda"
        print("🔥 NVIDIA GPU (CUDA) detected!")
    else:
        DEVICE = "cpu"
        print("⚠️  No GPU detected. Running on CPU (will be slow).")
    print(f"  Device: {DEVICE}")

    # ── 3. Load Video Info + Model ──
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    FPS = cap.get(cv2.CAP_PROP_FPS)
    TOTAL_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    MAX_SCAN_FRAMES = 2000 if TEST_MODE else TOTAL_FRAMES

    if TARGET_FRAMES is None and not args.video:
        tf_input = input("\nTarget frames (Enter for AUTO, or type number): ").strip()
        TARGET_FRAMES = int(tf_input) if tf_input.isdigit() else None

    print(f"\n  Duration:   {TOTAL_FRAMES/FPS/60:.1f}min | {W}x{H} | {FPS:.0f}fps | {TOTAL_FRAMES:,} frames")
    print(f"  Scan limit: {MAX_SCAN_FRAMES:,} | Target: {TARGET_FRAMES or 'AUTO'}")

    print(f"\n  Loading {config.PERSON_MODEL}...")
    yolo_model = YOLO(config.PERSON_MODEL)
    print("  Model loaded ✓")

    # ── 4. Detect Static Overlays ──
    print("\n[1/5] Detecting static overlays...")
    overlay_mask, overlay_ratio = detect_static_overlays(VIDEO_PATH)
    print(f"  Overlay coverage: {overlay_ratio*100:.1f}% masked")

    # ── 5. Team Calibration ──
    print("\n[2/5] Collecting samples for team calibration...")
    sample_data = collect_samples(
        VIDEO_PATH, yolo_model, DEVICE,
        overlay_mask=overlay_mask,
        n_sample_frames=80,
        n_display=24,
    )

    print("\n>>> Look at the grid to identify your team's jerseys.")
    print(">>> Close the plot window when ready to input numbers.")
    show_samples(sample_data)

    while True:
        raw = input("\n✏️  Enter grid numbers for YOUR team (comma-separated, e.g. 0,3,5): ")
        MY_TEAM = [int(x.strip()) for x in raw.split(",") if x.strip().isdigit()]
        if len(MY_TEAM) >= 2:
            break
        print("  Please enter at least 2 valid numbers.")

    print(f"\n  Your team labels: {MY_TEAM}")
    calibration = build_calibration(sample_data, MY_TEAM)

    # ── 6. Pass 1 — Fast Scan ──
    print("\n[3/5] Pass 1: Fast Scan...")
    segments, zoom_timeline, video_info = pass1_fast_scan(
        VIDEO_PATH, yolo_model, DEVICE, max_frames=MAX_SCAN_FRAMES
    )
    print(f"  Scanned: {len(zoom_timeline):,} frames")
    print(f"  Quality segments: {len(segments)}")

    # ── 7. Pass 2 — Team-Aware Extraction (with Torso Sharpness) ──
    print("\n[4/5] Pass 2: Team-Aware Extraction (v2 — Torso Sharpness)...")
    candidates, pass2_stats = pass2_extract(
        VIDEO_PATH, segments, yolo_model, DEVICE,
        calibration, overlay_mask, video_info,
        max_frames=MAX_SCAN_FRAMES
    )
    print(f"  Candidates found: {len(candidates)}")
    print(f"  Tier breakdown: Gold={pass2_stats.get('tier_gold',0)}, "
          f"Silver={pass2_stats.get('tier_silver',0)}, "
          f"Bronze={pass2_stats.get('tier_bronze',0)}")
    print(f"  Skipped (blurry torso): {pass2_stats.get('skipped_blurry',0)}")

    # ── 8. Selection ──
    print("\n[5/5] Tier-Aware Quota Selection...")
    if TARGET_FRAMES is None:
        TARGET_FRAMES = auto_target_frames(
            candidates, video_duration_sec=TOTAL_FRAMES / FPS
        )

    selected, sel_stats = select_by_quota(candidates, TARGET_FRAMES)
    print_selection_summary(selected, sel_stats, len(candidates))

    # ── 9. Save Frames + Metadata ──
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    rows = []

    for idx, meta in enumerate(tqdm(selected, desc="Saving to disk")):
        cap.set(cv2.CAP_PROP_POS_FRAMES, meta["frame_num"])
        ret, frame = cap.read()
        if not ret:
            continue

        fname = f"{MATCH_ID}_{meta['frame_num']:06d}_{meta['timestamp_hms']}.jpg"
        cv2.imwrite(str(FRAMES_DIR / fname), frame,
                     [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])

        rows.append({
            "filename": fname,
            **meta,
            "match_id": MATCH_ID,
            "member": MEMBER_NAME,
            "extracted_at": datetime.now().isoformat(),
        })

    cap.release()

    df = pd.DataFrame(rows)
    csv_path = METADATA_DIR / f"{MATCH_ID}_v2_index.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Saved {len(df)} frames to: {FRAMES_DIR}")
    print(f"✅ Metadata saved to: {csv_path}")

    # ── 10. Roboflow Upload ──
    if not args.no_roboflow:
        do_upload = input("\nUpload to Roboflow? (y/n): ").strip().lower()
        if do_upload == 'y':
            try:
                from roboflow import Roboflow
                ROBOFLOW_API_KEY = getpass.getpass("Roboflow API Key: ")
                if ROBOFLOW_API_KEY.strip():
                    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
                    ROBOFLOW_PROJECT = input("Roboflow project name (e.g. kit-sponsor-logos): ").strip()
                    project = rf.workspace().project(ROBOFLOW_PROJECT)

                    for fp in tqdm(sorted(FRAMES_DIR.glob(f"{MATCH_ID}_*.jpg")), desc="Uploading"):
                        try:
                            project.upload(
                                image_path=str(fp), split="train",
                                tag_names=[MATCH_ID, MEMBER_NAME, "v2"]
                            )
                        except Exception as e:
                            print(f"  Error uploading {fp.name}: {e}")
                    print("✅ Upload complete!")
            except ImportError:
                print("roboflow not found. Install: pip install roboflow")

    print("\n🎉 Phase 1 complete! Next steps:")
    print("   1. Annotate frames on Roboflow (manual)")
    print("   2. Export annotations (YOLO format)")
    print("   3. Run Phase 1b: python run_propagation.py")


if __name__ == "__main__":
    main()
