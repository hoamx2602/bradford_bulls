"""
Bradford Bulls v2 — Google Colab Setup & Run

Copy this entire file into a Colab notebook cell and run it.
Or split at the "# ── CELL N ──" markers into separate cells.

Prerequisites:
  - Google Drive mounted at /content/drive
  - Video files in /content/drive/MyDrive/Bradford_Bulls/videos/
"""

# ══════════════════════════════════════════════════════════════════════
# ── CELL 1: Mount Drive + Clone Repo ──
# ══════════════════════════════════════════════════════════════════════

from google.colab import drive
drive.mount('/content/drive')

import os
REPO_URL = "https://github.com/YOUR_USERNAME/BRADFORD_BULLS_PROJECT.git"  # ← UPDATE THIS
REPO_DIR = "/content/BRADFORD_BULLS_PROJECT"
V2_DIR = f"{REPO_DIR}/v2-optimize"

if not os.path.exists(REPO_DIR):
    os.system(f"git clone {REPO_URL} {REPO_DIR}")
else:
    os.system(f"cd {REPO_DIR} && git pull")

print(f"✅ Repo at: {REPO_DIR}")
print(f"✅ v2 code at: {V2_DIR}")


# ══════════════════════════════════════════════════════════════════════
# ── CELL 2: Install Dependencies ──
# ══════════════════════════════════════════════════════════════════════

os.system(f"pip install -q -r {V2_DIR}/requirements.txt")
print("✅ Dependencies installed")


# ══════════════════════════════════════════════════════════════════════
# ── CELL 3: Configure Paths ──
# ══════════════════════════════════════════════════════════════════════

# ← UPDATE these paths to match your Google Drive structure
DRIVE_BASE = "/content/drive/MyDrive/Bradford_Bulls"
VIDEOS_DIR = f"{DRIVE_BASE}/videos"
OUTPUT_DIR = f"{DRIVE_BASE}/v2_output"

# List available videos
import glob
videos = glob.glob(f"{VIDEOS_DIR}/*.mp4") + glob.glob(f"{VIDEOS_DIR}/*.MP4")
print(f"\n📹 Available videos ({len(videos)}):")
for i, v in enumerate(sorted(videos)):
    size_mb = os.path.getsize(v) / 1e6
    print(f"  [{i}] {os.path.basename(v)} ({size_mb:.0f} MB)")


# ══════════════════════════════════════════════════════════════════════
# ── CELL 4: Select Video & Run Phase 1 ──
# ══════════════════════════════════════════════════════════════════════

#@markdown ### Video Selection
VIDEO_INDEX = 0  #@param {type:"integer"}
MEMBER_NAME = "your_name"  #@param {type:"string"}
TEST_MODE = True  #@param {type:"boolean"}
TARGET_FRAMES = 0  #@param {type:"integer"}

# Resolve paths
video_path = sorted(videos)[VIDEO_INDEX]
video_name = os.path.basename(video_path)
print(f"\n🎬 Selected: {video_name}")

# Build command
cmd = f"python {V2_DIR}/run_extraction.py"
cmd += f" --video '{video_path}'"
cmd += f" --member {MEMBER_NAME}"
cmd += f" --output-dir '{OUTPUT_DIR}'"
cmd += " --no-roboflow"  # Upload separately
if TEST_MODE:
    cmd += " --test"
if TARGET_FRAMES > 0:
    cmd += f" --target-frames {TARGET_FRAMES}"

print(f"\n🚀 Running: {cmd}\n")
os.system(cmd)


# ══════════════════════════════════════════════════════════════════════
# ── CELL 5: Preview Extracted Frames ──
# ══════════════════════════════════════════════════════════════════════

import matplotlib.pyplot as plt
import cv2
import pandas as pd

match_id = os.path.splitext(video_name)[0]
frames_dir = f"{OUTPUT_DIR}/frames/{match_id}"
metadata_csv = f"{OUTPUT_DIR}/metadata/{match_id}_v2_index.csv"

if os.path.exists(metadata_csv):
    df = pd.read_csv(metadata_csv)
    print(f"\n📊 Extraction Summary:")
    print(f"  Total frames: {len(df)}")
    print(f"  Tier breakdown:")
    print(df['sharpness_tier'].value_counts().to_string(header=False))
    print(f"\n  Category breakdown:")
    print(df['category'].value_counts().to_string(header=False))

    # Show sample frames per tier
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    for row, tier in enumerate(["gold", "silver", "bronze"]):
        tier_df = df[df['sharpness_tier'] == tier].head(5)
        for col in range(5):
            ax = axes[row, col]
            if col < len(tier_df):
                fname = tier_df.iloc[col]['filename']
                img = cv2.imread(f"{frames_dir}/{fname}")
                if img is not None:
                    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    score = tier_df.iloc[col]['torso_sharpness']
                    ax.set_title(f"T={score:.3f}", fontsize=10)
            ax.axis('off')
            if col == 0:
                tier_icon = {"gold": "🥇", "silver": "🥈", "bronze": "🥉"}
                ax.set_ylabel(f"{tier_icon.get(tier, '')} {tier.upper()}",
                             fontsize=14, rotation=0, labelpad=60)

    plt.suptitle("Frame Quality Tiers — Torso Sharpness", fontsize=16)
    plt.tight_layout()
    plt.show()
else:
    print(f"⚠️  Metadata not found at {metadata_csv}")
    print("   Run the extraction first (Cell 4)")


# ══════════════════════════════════════════════════════════════════════
# ── CELL 6 (OPTIONAL): Upload to Roboflow ──
# ══════════════════════════════════════════════════════════════════════

UPLOAD_TO_ROBOFLOW = False  #@param {type:"boolean"}
ROBOFLOW_API_KEY = ""  #@param {type:"string"}
ROBOFLOW_PROJECT = "kit-sponsor-logos"  #@param {type:"string"}

if UPLOAD_TO_ROBOFLOW and ROBOFLOW_API_KEY:
    from roboflow import Roboflow
    from tqdm import tqdm

    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace().project(ROBOFLOW_PROJECT)

    frame_files = sorted(glob.glob(f"{frames_dir}/{match_id}_*.jpg"))
    print(f"\n📤 Uploading {len(frame_files)} frames to Roboflow...")

    for fp in tqdm(frame_files):
        try:
            project.upload(
                image_path=fp, split="train",
                tag_names=[match_id, MEMBER_NAME, "v2"]
            )
        except Exception as e:
            print(f"  Error: {e}")

    print("✅ Upload complete!")


# ══════════════════════════════════════════════════════════════════════
# ── CELL 7: Phase 1b — Annotation Propagation ──
# (Run this AFTER annotating on Roboflow and exporting YOLO format)
# ══════════════════════════════════════════════════════════════════════

RUN_PROPAGATION = False  #@param {type:"boolean"}

# Paths to exported Roboflow dataset (YOLO format)
EXPORTED_IMAGES = f"{DRIVE_BASE}/roboflow_export/train/images"  #@param {type:"string"}
EXPORTED_LABELS = f"{DRIVE_BASE}/roboflow_export/train/labels"  #@param {type:"string"}
PROPAGATED_OUTPUT = f"{DRIVE_BASE}/v2_propagated"  #@param {type:"string"}

if RUN_PROPAGATION:
    cmd = f"python {V2_DIR}/run_propagation.py"
    cmd += f" --video '{video_path}'"
    cmd += f" --frames-dir '{EXPORTED_IMAGES}'"
    cmd += f" --labels-dir '{EXPORTED_LABELS}'"
    cmd += f" --output-dir '{PROPAGATED_OUTPUT}'"
    cmd += " --use-yolo"  # Use YOLO for accurate torso sharpness
    cmd += f" --radius 3"

    print(f"\n🚀 Running propagation: {cmd}\n")
    os.system(cmd)
