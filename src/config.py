"""
Bradford Bulls - AI Sponsorship Exposure Valuation System
Configuration & Device Detection
"""

import os
from pathlib import Path

import torch

# ============================================================
# PROJECT PATHS
# ============================================================
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
VIDEOS_DIR = OUTPUT_DIR / "videos"
FRAMES_DIR = OUTPUT_DIR / "frames"
FRAMES_CLEAR_DIR = OUTPUT_DIR / "frames_clear"
METADATA_DIR = OUTPUT_DIR / "metadata"
KIT_SPONSORS_DIR = PROJECT_ROOT / "Kit Sponsors" / "Kit Sponsors"

# Ensure directories exist
for d in [VIDEOS_DIR, FRAMES_DIR, FRAMES_CLEAR_DIR, METADATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# DEVICE DETECTION (MPS / CUDA / CPU)
# ============================================================

def get_device() -> str:
    if torch.cuda.is_available():
        device = "cuda"
        name = torch.cuda.get_device_name(0)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        name = "Apple Silicon (Metal)"
    else:
        device = "cpu"
        name = "CPU"
    print(f"[Device] Using: {device} ({name})")
    return device

DEVICE = get_device()

# ============================================================
# FRAME SAMPLING CONFIG
# ============================================================

# L1 - Temporal Sampling
TARGET_FPS = 2  # Extract 2 frames per second (from 30/60 FPS original)

# L2 - Scene Change Detection
PHASH_THRESHOLD = 10       # Hamming distance threshold for perceptual hash
SSIM_THRESHOLD = 0.92      # Frames with SSIM > this are considered "same scene"

# L3 - Player Presence Filter
PERSON_CONFIDENCE = 0.5    # Min confidence for YOLOv8 person detection
MIN_PERSONS_IN_FRAME = 1   # At least 1 person must be detected

# ============================================================
# PLAYER VISIBILITY FILTER CONFIG
# ============================================================

# Min player bounding box area as % of frame area
MIN_PLAYER_AREA_RATIO = 0.03  # 3% of frame (slightly relaxed from 5%)

# Min sharpness (Laplacian variance) - frames below this are too blurry
MIN_SHARPNESS = 50.0

# Min player visibility (not occluded) - 0 to 1
MIN_VISIBILITY = 0.70

# ============================================================
# VIDEO DOWNLOAD CONFIG
# ============================================================

# yt-dlp: prefer merged video+audio up to 1080p.
# Do not force video ext=mp4 only — on YouTube the best MP4 is often 720p while 1080p is VP9/WebM.
# See README.md to raise the cap (e.g. 4K) or pin a different format string.
YT_DLP_FORMAT = (
    "bestvideo[height<=1080]+bestaudio/"
    "bestvideo+bestaudio/"
    "best[height<=1080]/"
    "best"
)

# ============================================================
# SPONSOR LABELS (from Kit Sponsors directory)
# ============================================================

SPONSOR_LABELS = {
    "aon": "AON",
    "atm_hospitality": "ATM Hospitality",
    "cch_cedar_court": "Cedar Court Hotels",
    "chadlaw": "ChadLaw",
    "em_workwear": "EM Workwear",
    "fairway_flooring": "Fairway Flooring",
    "klg": "KLG",
    "mcp": "MCP",
    "mna_cladding": "MNA Cladding",
    "mna_support": "MNA Support Services",
    "bartercard": "Bartercard",
    "top_notch": "Top Notch",
    "romantica_beds": "Romantica Beds",
}

# Mapping: sponsor → jersey position (from pricing CSV)
SPONSOR_POSITIONS = {
    "aon": "Main Sponsor",
    "atm_hospitality": "Collar Bone",
    "cch_cedar_court": "Chest (opp Badge)",
    "chadlaw": "Sleeve 1",
    "em_workwear": "Sleeve 3",
    "fairway_flooring": "Sleeve 2",
    "klg": "Top Back",
    "mcp": "Sleeve 2",
    "mna_cladding": "Shorts Front",
    "mna_support": "Shorts Back 1",
    "bartercard": "Bottom Back",
    "top_notch": "Socks",
    "romantica_beds": "Shorts Back 2",
}

# Pricing percentages (from Bradford Bulls Current Pricing.csv)
POSITION_PRICING = {
    "Main Sponsor": 0.26,
    "Collar Back": 0.08,
    "Collar Bone": 0.08,
    "Chest (opp Badge)": 0.07,
    "Sleeve 1": 0.04,
    "Sleeve 2": 0.11,
    "Sleeve 3": 0.04,
    "Top Back": 0.05,
    "Nape Neck": 0.03,
    "Bottom Back": 0.03,
    "Top Back Shorts": 0.05,
    "Shorts Front": 0.03,
    "Shorts Back 1": 0.03,
    "Shorts Back 2": 0.03,
    "Socks": 0.01,
}
