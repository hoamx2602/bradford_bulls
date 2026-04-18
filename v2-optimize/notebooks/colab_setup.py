"""
Bradford Bulls v2 — Google Colab Setup & Run

Copy each "# ── CELL N ──" section into a SEPARATE Colab notebook cell.
All code runs DIRECTLY in notebook (no subprocess) so you see all output + errors.
"""


# ══════════════════════════════════════════════════════════════════════
# ── CELL 1: Mount Drive + Clone Repo + Install ──
# ══════════════════════════════════════════════════════════════════════

from google.colab import drive
drive.mount('/content/drive')

import os, subprocess

REPO_URL = "https://github.com/YOUR_USERNAME/BRADFORD_BULLS_PROJECT.git"  # ← UPDATE THIS
REPO_DIR = "/content/BRADFORD_BULLS_PROJECT"
V2_DIR = f"{REPO_DIR}/v2-optimize"

if not os.path.exists(REPO_DIR):
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
    print(f"✅ Cloned repo")
else:
    subprocess.run(["git", "-C", REPO_DIR, "pull"], check=True)
    print(f"✅ Repo updated")

subprocess.run(["pip", "install", "-q", "-r", f"{V2_DIR}/requirements.txt"], check=True)
print("✅ Dependencies installed")


# ══════════════════════════════════════════════════════════════════════
# ── CELL 2: Import v2 modules + setup paths ──
# ══════════════════════════════════════════════════════════════════════

import sys
sys.path.insert(0, V2_DIR)

import torch
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from ultralytics import YOLO

from src import config
from src.overlay import detect_static_overlays, visualize_overlay
from src.calibration import collect_samples, show_samples, confirm_selection, build_calibration
from src.pipeline import pass1_fast_scan, pass2_extract
from src.selection import select_by_quota, auto_target_frames, print_selection_summary

# ── Paths (UPDATE these) ──
DRIVE_BASE = "/content/drive/MyDrive/Bradford_Bulls"
VIDEOS_DIR = f"{DRIVE_BASE}/videos"
OUTPUT_DIR = f"{DRIVE_BASE}/v2_output"

# Device
if torch.cuda.is_available():
    DEVICE = "cuda"
    print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    DEVICE = "mps"
    print("🚀 Apple Silicon GPU")
else:
    DEVICE = "cpu"
    print("⚠️ CPU only")

# List videos
videos = sorted(glob.glob(f"{VIDEOS_DIR}/*.mp4") + glob.glob(f"{VIDEOS_DIR}/*.MP4"))
print(f"\n📹 Available videos ({len(videos)}):")
for i, v in enumerate(videos):
    size_mb = os.path.getsize(v) / 1e6
    print(f"  [{i}] {os.path.basename(v)} ({size_mb:.0f} MB)")

print(f"\n✅ Ready! Set VIDEO_INDEX in next cell.")


# ══════════════════════════════════════════════════════════════════════
# ── CELL 3: Select Video + Load Model ──
# ══════════════════════════════════════════════════════════════════════

# ← CHANGE THESE
VIDEO_INDEX = 0
MEMBER_NAME = "your_name"
TEST_MODE = True   # True = scan first 2000 frames only

# Resolve
VIDEO_PATH = Path(videos[VIDEO_INDEX])
MATCH_ID = VIDEO_PATH.stem
FRAMES_DIR = Path(OUTPUT_DIR) / "frames" / MATCH_ID
METADATA_DIR = Path(OUTPUT_DIR) / "metadata"
FRAMES_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)

# Video info
cap = cv2.VideoCapture(str(VIDEO_PATH))
FPS = cap.get(cv2.CAP_PROP_FPS)
TOTAL_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

MAX_SCAN = 2000 if TEST_MODE else TOTAL_FRAMES

print(f"🎬 Video: {VIDEO_PATH.name}")
print(f"   Duration: {TOTAL_FRAMES/FPS/60:.1f} min | {W}×{H} | {FPS:.0f} fps | {TOTAL_FRAMES:,} frames")
print(f"   Mode: {'TEST (2000 frames)' if TEST_MODE else 'PRODUCTION (full)'}")
print(f"   Scan limit: {MAX_SCAN:,}")

# Load YOLO
print(f"\n   Loading {config.PERSON_MODEL}...")
yolo_model = YOLO(config.PERSON_MODEL)
print(f"   ✅ Model loaded")


# ══════════════════════════════════════════════════════════════════════
# ── CELL 4: Detect Overlays ──
# ══════════════════════════════════════════════════════════════════════

print("[1/5] Detecting static overlays...")
overlay_mask, overlay_ratio = detect_static_overlays(VIDEO_PATH)
print(f"  Overlay coverage: {overlay_ratio*100:.1f}% masked")

# Visualize (optional — shows which areas are detected as scoreboard/watermark)
cap = cv2.VideoCapture(str(VIDEO_PATH))
cap.set(cv2.CAP_PROP_POS_FRAMES, int(TOTAL_FRAMES * 0.3))
ret, sample_frame = cap.read()
cap.release()
if ret:
    visualize_overlay(sample_frame, overlay_mask)


# ══════════════════════════════════════════════════════════════════════
# ── CELL 5: Team Calibration — Sample Collection ──
# ══════════════════════════════════════════════════════════════════════

print("[2/5] Collecting samples for team calibration...")
sample_data = collect_samples(
    VIDEO_PATH, yolo_model, DEVICE,
    overlay_mask=overlay_mask,
    n_sample_frames=80,
    n_display=24,
)

show_samples(sample_data)
print("\n👆 Look at the grid above. Note the numbers of YOUR team's jerseys.")


# ══════════════════════════════════════════════════════════════════════
# ── CELL 6a: Team Calibration — Input Your Team ──
# ══════════════════════════════════════════════════════════════════════

# ← TYPE YOUR TEAM'S GRID NUMBERS HERE (from the grid above)
MY_TEAM = [0, 3, 5]  # ← UPDATE THIS

# This shows your selections at large size with color analysis.
# If it warns about mixed LIGHT/DARK, you probably selected opponent crops by mistake.
# → Remove the wrong numbers from MY_TEAM and re-run this cell.
MY_TEAM = confirm_selection(sample_data, MY_TEAM)


# ══════════════════════════════════════════════════════════════════════
# ── CELL 6b: Build Calibration (only after confirming above) ──
# ══════════════════════════════════════════════════════════════════════

calibration = build_calibration(sample_data, MY_TEAM)
print("\n✅ Calibration complete!")


# ══════════════════════════════════════════════════════════════════════
# ── CELL 7: Pass 1 — Fast Scan ──
# ══════════════════════════════════════════════════════════════════════

print("[3/5] Pass 1: Fast Scan...")
segments, zoom_timeline, video_info = pass1_fast_scan(
    VIDEO_PATH, yolo_model, DEVICE, max_frames=MAX_SCAN
)
print(f"  Scanned: {len(zoom_timeline):,} frames")
print(f"  Quality segments: {len(segments)}")

# Quick zoom timeline visualization
if zoom_timeline:
    fns = [z[0] for z in zoom_timeline]
    ratios = [z[1] for z in zoom_timeline]
    plt.figure(figsize=(15, 3))
    plt.fill_between(fns, ratios, alpha=0.3, color='#3498db')
    plt.plot(fns, ratios, linewidth=0.5, color='#2c3e50')
    plt.axhline(y=config.MIN_MAX_PERSON_AREA_RATIO, color='red',
                linestyle='--', alpha=0.5, label='Min threshold')
    plt.xlabel('Frame')
    plt.ylabel('Max Person Area Ratio')
    plt.title('Zoom Timeline — Player Size Over Time')
    plt.legend()
    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════════════
# ── CELL 8: Pass 2 — Team-Aware Extraction (Torso Sharpness) ──
# ══════════════════════════════════════════════════════════════════════

print("[4/5] Pass 2: Team-Aware Extraction (v2 — Torso Sharpness)...")
candidates, pass2_stats = pass2_extract(
    VIDEO_PATH, segments, yolo_model, DEVICE,
    calibration, overlay_mask, video_info,
    max_frames=MAX_SCAN
)

print(f"\n  ✅ Candidates found: {len(candidates)}")
print(f"  Tiers: 🥇 Gold={pass2_stats.get('tier_gold',0)}, "
      f"🥈 Silver={pass2_stats.get('tier_silver',0)}, "
      f"🥉 Bronze={pass2_stats.get('tier_bronze',0)}")
print(f"  Skipped (blurry torso): {pass2_stats.get('skipped_blurry',0)}")
print(f"  Skipped (no pitch): {pass2_stats.get('skipped_pitch',0)}")


# ══════════════════════════════════════════════════════════════════════
# ── CELL 9: Quota Selection + Save ──
# ══════════════════════════════════════════════════════════════════════

TARGET_FRAMES = 0  # ← Set to 0 for AUTO, or type a number like 400

if TARGET_FRAMES <= 0:
    TARGET_FRAMES = auto_target_frames(
        candidates, video_duration_sec=TOTAL_FRAMES / FPS
    )

selected, sel_stats = select_by_quota(candidates, TARGET_FRAMES)
print_selection_summary(selected, sel_stats, len(candidates))

# ── Save frames to disk ──
cap = cv2.VideoCapture(str(VIDEO_PATH))
rows = []

for idx, meta in enumerate(tqdm(selected, desc="Saving frames")):
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

print(f"\n✅ Saved {len(df)} frames → {FRAMES_DIR}")
print(f"✅ Metadata → {csv_path}")


# ══════════════════════════════════════════════════════════════════════
# ── CELL 10: Preview Results ──
# ══════════════════════════════════════════════════════════════════════

print(f"📊 Extraction Summary ({MATCH_ID}):")
print(f"  Total frames: {len(df)}")
print(f"\n  Tier breakdown:")
print(df['sharpness_tier'].value_counts().to_string())
print(f"\n  Category breakdown:")
print(df['category'].value_counts().to_string())

# Show sample frames per tier
fig, axes = plt.subplots(3, 5, figsize=(20, 12))
for row, tier in enumerate(["gold", "silver", "bronze"]):
    tier_df = df[df['sharpness_tier'] == tier].head(5)
    for col in range(5):
        ax = axes[row, col]
        if col < len(tier_df):
            fname = tier_df.iloc[col]['filename']
            img_path = str(FRAMES_DIR / fname)
            img = cv2.imread(img_path)
            if img is not None:
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                ts = tier_df.iloc[col]['torso_sharpness']
                ax.set_title(f"Torso={ts:.3f}", fontsize=10)
        ax.axis('off')
        if col == 0:
            icons = {"gold": "🥇", "silver": "🥈", "bronze": "🥉"}
            ax.set_ylabel(f"{icons.get(tier,'')} {tier.upper()}",
                         fontsize=14, rotation=0, labelpad=60)

plt.suptitle(f"Frame Quality Tiers — {MATCH_ID}", fontsize=16)
plt.tight_layout()
plt.show()


# ══════════════════════════════════════════════════════════════════════
# ── CELL 11 (OPTIONAL): Upload to Roboflow ──
# ══════════════════════════════════════════════════════════════════════

UPLOAD = False  # ← Set True when ready

if UPLOAD:
    import getpass
    from roboflow import Roboflow

    api_key = getpass.getpass("Roboflow API Key: ")
    project_name = input("Project name (e.g. kit-sponsor-logos): ").strip()

    rf = Roboflow(api_key=api_key)
    project = rf.workspace().project(project_name)

    frame_files = sorted(FRAMES_DIR.glob(f"{MATCH_ID}_*.jpg"))
    print(f"\n📤 Uploading {len(frame_files)} frames...")

    success, fail = 0, 0
    for fp in tqdm(frame_files):
        try:
            project.upload(
                image_path=str(fp), split="train",
                tag_names=[MATCH_ID, MEMBER_NAME, "v2"]
            )
            success += 1
        except Exception as e:
            fail += 1
            if fail <= 3:
                print(f"  ⚠️ {fp.name}: {e}")

    print(f"\n✅ Uploaded: {success} | Failed: {fail}")


# ══════════════════════════════════════════════════════════════════════
# ── CELL 12: Phase 1b — Annotation Propagation ──
# (Run AFTER annotating on Roboflow + exporting YOLO format)
# ══════════════════════════════════════════════════════════════════════

RUN_PROPAGATION = False  # ← Set True AFTER you finish annotation

if RUN_PROPAGATION:
    from src.propagation import run_propagation

    # ← UPDATE these paths to your exported Roboflow dataset
    EXPORTED_IMAGES = f"{DRIVE_BASE}/roboflow_export/train/images"
    EXPORTED_LABELS = f"{DRIVE_BASE}/roboflow_export/train/labels"
    PROPAGATED_OUTPUT = f"{DRIVE_BASE}/v2_propagated"

    print("Running annotation propagation...")
    print(f"  Images: {EXPORTED_IMAGES}")
    print(f"  Labels: {EXPORTED_LABELS}")
    print(f"  Output: {PROPAGATED_OUTPUT}")

    stats = run_propagation(
        video_path=str(VIDEO_PATH),
        frames_dir=EXPORTED_IMAGES,
        labels_dir=EXPORTED_LABELS,
        output_dir=PROPAGATED_OUTPUT,
        yolo_model=yolo_model,
        device=DEVICE,
        radius=3,
    )

    if stats:
        print(f"\n🎉 Done! Dataset amplified: {stats['original_frames']} → {stats['total_frames']} frames")
