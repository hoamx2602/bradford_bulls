#!/usr/bin/env python3
"""
Bradford Bulls — Team-Aware Frame Extraction v3.0 (Standalone Script)

This is the Python script equivalent of the 02_team_aware_extraction.ipynb notebook.
It runs locally and does not depend on Google Colab or Google Drive.
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
import matplotlib.pyplot as plt
from tqdm import tqdm
from ultralytics import YOLO

# Ensure src is in the python path
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from frame_extraction.overlay import detect_static_overlays, visualize_overlay
    from frame_extraction.calibration import collect_samples, show_samples, build_calibration
    from frame_extraction.pipeline import pass1_fast_scan, pass2_extract
    from frame_extraction.selection import select_by_quota, auto_target_frames, print_selection_summary
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please make sure you are running this script from the project root or src is in your PYTHONPATH.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Team-Aware Frame Extraction")
    parser.add_argument("--member", type=str, help="Your name (e.g., edward)")
    parser.add_argument("--video", type=str, help="Video filename inside videos/ (e.g., M06_black_1080p.mp4)")
    parser.add_argument("--test", action="store_true", help="Run in test mode (scan only first 2000 frames)")
    parser.add_argument("--target-frames", type=int, help="Target frames to extract (default is auto)")
    parser.add_argument("--no-roboflow", action="store_true", help="Skip Roboflow upload completely")
    args = parser.parse_args()

    print("=== Bradford Bulls Frame Extraction ===")
    
    # 1. Config
    MEMBER_NAME = args.member if args.member else input("Your name (e.g. edward): ").strip().lower()
    VIDEO_FILENAME = args.video if args.video else input("Video filename (e.g. M06_black_1080p.mp4): ").strip()
    
    TEST_MODE = args.test
    if not args.member and not args.video: # interactive
        test_input = input("Test mode? (y/n, default=y): ").strip().lower()
        TEST_MODE = test_input != "n"
        
    # Directories
    VIDEOS_DIR = PROJECT_ROOT / "videos"
    FRAMES_DIR = PROJECT_ROOT / "frames"
    METADATA_DIR = PROJECT_ROOT / "metadata"
    
    for d in [VIDEOS_DIR, FRAMES_DIR, METADATA_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    VIDEO_PATH = VIDEOS_DIR / VIDEO_FILENAME
    if not VIDEO_PATH.exists():
        print(f"\nERROR: Video not found at {VIDEO_PATH}")
        print("Please ensure the video is placed inside the 'videos' folder.")
        sys.exit(1)

    MATCH_ID = VIDEO_PATH.stem
    FRAMES_OUTPUT_DIR = FRAMES_DIR / MATCH_ID
    FRAMES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"\n--- Config ---")
    print(f"  Member:   {MEMBER_NAME}")
    print(f"  Video:    {VIDEO_PATH.name} ({VIDEO_PATH.stat().st_size/1e6:.0f} MB)")
    print(f"  Mode:     {'TEST' if TEST_MODE else 'PRODUCTION'}")

    # 2. Setup
    # Use Apple Silicon GPU (MPS) if available, otherwise CUDA, otherwise CPU
    if torch.backends.mps.is_available():
        DEVICE = "mps"
        print("🚀 Apple Silicon GPU (MPS) detected! Using M4 Pro acceleration.")
    elif torch.cuda.is_available():
        DEVICE = "cuda"
        print("🔥 NVIDIA GPU (CUDA) detected!")
    else:
        DEVICE = "cpu"
        print("⚠️ Warning: No GPU detected. Running on CPU (will be slow).")
        
    print(f"Device set to: {DEVICE}")

    # 3. Load Video & Model
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    FPS = cap.get(cv2.CAP_PROP_FPS)
    TOTAL_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    PERSON_DETECTION_MODEL = "yolo11l.pt"
    JPEG_QUALITY = 95
    MAX_SCAN_FRAMES = 2000 if TEST_MODE else TOTAL_FRAMES
    
    TARGET_FRAMES = args.target_frames
    if TARGET_FRAMES is None and not args.video: # interactive
        tf_input = input("\nTarget frames (press Enter for AUTO, or type a number): ").strip()
        TARGET_FRAMES = int(tf_input) if tf_input.isdigit() else None

    print(f"\nDuration: {TOTAL_FRAMES/FPS/60:.1f}min | {W}x{H} | {FPS:.0f}fps | {TOTAL_FRAMES:,} frames")
    print(f"Scan limit: {MAX_SCAN_FRAMES:,} | Target frames: {TARGET_FRAMES or 'AUTO'}")

    print(f"\nLoading {PERSON_DETECTION_MODEL}...")
    yolo_model = YOLO(PERSON_DETECTION_MODEL)
    print("Model loaded.")

    # 4. Detect Static Overlays
    print("\n[1/5] Detecting static overlays...")
    overlay_mask, overlay_ratio = detect_static_overlays(VIDEO_PATH)
    print(f"Overlay coverage: {overlay_ratio*100:.1f}% masked")
    
    # 5. Team Calibration
    print("\n[2/5] Collecting samples for team calibration...")
    sample_data = collect_samples(
        VIDEO_PATH, yolo_model, DEVICE,
        overlay_mask=overlay_mask,
        n_sample_frames=80,
        n_display=24,
    )
    
    print("\n>>> Look at the windows popping up to see the sample grid.")
    print(">>> Close the plot window when you are ready to input your team numbers in the terminal.")
    show_samples(sample_data)
    
    while True:
        raw = input("\n✏️  Enter grid numbers for YOUR team (comma-separated, e.g. 0,3,5): ")
        MY_TEAM = [int(x.strip()) for x in raw.split(",") if x.strip().isdigit()]
        if len(MY_TEAM) >= 2:
            break
        print("Please enter at least 2 valid numbers.")
        
    print(f"\nYour team labels: {MY_TEAM}")
    calibration = build_calibration(sample_data, MY_TEAM)
    
    # 6. Pass 1 - Fast Scan
    print("\n[3/5] Pass 1: Fast Scan...")
    segments, zoom_timeline, video_info = pass1_fast_scan(
        VIDEO_PATH, yolo_model, DEVICE, max_frames=MAX_SCAN_FRAMES
    )
    
    print(f"Scanned: {len(zoom_timeline):,} frames")
    print(f"Quality segments: {len(segments)}")

    # 7. Pass 2 - Extract
    print("\n[4/5] Pass 2: Team-Aware Extraction...")
    candidates, pass2_stats = pass2_extract(
        VIDEO_PATH, segments, yolo_model, DEVICE,
        calibration, overlay_mask, video_info,
        max_frames=MAX_SCAN_FRAMES
    )
    print(f"Candidates found: {len(candidates)}")
    
    # 8. Selection
    print("\n[5/5] Quota Selection and Saving...")
    if TARGET_FRAMES is None:
        TARGET_FRAMES = auto_target_frames(candidates, video_duration_sec=TOTAL_FRAMES / FPS)
        
    selected, sel_stats = select_by_quota(candidates, TARGET_FRAMES)
    print_selection_summary(selected, sel_stats, len(candidates))

    # 9. Save Frames & Metadata
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    rows = []
    
    for idx, meta in enumerate(tqdm(selected, desc="Saving to disk")):
        cap.set(cv2.CAP_PROP_POS_FRAMES, meta["frame_num"])
        ret, frame = cap.read()
        if not ret: continue

        fname = f"{MATCH_ID}_{meta['frame_num']:06d}_{meta['timestamp_hms']}.jpg"
        cv2.imwrite(str(FRAMES_OUTPUT_DIR / fname), frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

        rows.append({"filename": fname, **meta, "match_id": MATCH_ID,
                     "member": MEMBER_NAME, "extracted_at": datetime.now().isoformat()})

    cap.release()
    
    df = pd.DataFrame(rows)
    csv_path = METADATA_DIR / f"{MATCH_ID}_v3_index.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Saved {len(df)} frames to: {FRAMES_OUTPUT_DIR}")
    print(f"✅ Metadata saved to: {csv_path}")

    # 10. Roboflow Upload
    if not args.no_roboflow:
        do_upload = input("\nDo you want to upload these frames to Roboflow? (y/n): ").strip().lower()
        if do_upload == 'y':
            try:
                from roboflow import Roboflow
                ROBOFLOW_API_KEY = getpass.getpass("Roboflow API Key: ")
                if ROBOFLOW_API_KEY.strip():
                    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
                    ROBOFLOW_PROJECT = "kit-sponsor-logos"
                    project = rf.workspace().project(ROBOFLOW_PROJECT)
                    
                    for fp in tqdm(sorted(FRAMES_OUTPUT_DIR.glob(f"{MATCH_ID}_*.jpg")), desc="Uploading"):
                        try:
                            project.upload(image_path=str(fp), split="train", tag_names=[f"{MATCH_ID}", f"{MEMBER_NAME}", "v3"])
                        except Exception as e:
                            print(f"Error uploading {fp.name}: {e}")
                    print("✅ Upload complete.")
            except ImportError:
                print("roboflow limit found. To upload, install tool first: pip install roboflow")

if __name__ == "__main__":
    main()
