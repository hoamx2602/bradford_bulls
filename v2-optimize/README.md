# Bradford Bulls v2 — Smart Frame Extraction Pipeline

## Overview
Optimized frame extraction pipeline with **torso-based sharpness scoring** and **tiered quality system** for logo exposure detection training data.

## Key Improvements over v1

| Feature | v1 | v2 |
|---------|----|----|
| Sharpness target | Whole player bbox | **Torso region only** (15-65% height) |
| Quality tiers | Binary (pass/fail) | **Gold / Silver / Bronze** |
| Scoring | Player sharpness × 0.25 | Torso sharpness × 0.20 + **tier bonus** |
| Selection | Score-based only | **Tier-first, score-second** |
| Annotation boost | N/A | **Temporal propagation (5x data)** |

## Quick Start

### Local (macOS/Linux)
```bash
cd v2-optimize
pip install -r requirements.txt
python run_extraction.py --video /path/to/video.mp4 --member yourname
```

### Google Colab
1. Open `notebooks/colab_setup.py`
2. Copy each CELL section into a Colab notebook cell
3. Update paths (`REPO_URL`, `DRIVE_BASE`)
4. Run cells sequentially

## Pipeline Phases

### Phase 1: Smart Frame Extraction (`run_extraction.py`)
Extracts 300-400 high-quality frames with torso-focused sharpness scoring.

### Phase 1b: Annotation Propagation (`run_propagation.py`)
After manual annotation on Roboflow, propagates annotations to ±3 neighbor frames.
Creates ~5x more training data with REAL motion blur.

## File Structure
```
v2-optimize/
├── src/
│   ├── __init__.py
│   ├── config.py          # All tunable parameters
│   ├── helpers.py          # Sharpness, detection, utilities
│   ├── overlay.py          # Static overlay detection
│   ├── calibration.py      # Team color calibration
│   ├── pipeline.py         # Pass 1 + Pass 2 extraction
│   ├── selection.py        # Tier-aware quota selection
│   └── propagation.py      # Temporal annotation propagation
├── run_extraction.py       # Phase 1 entry point
├── run_propagation.py      # Phase 1b entry point
├── notebooks/
│   └── colab_setup.py      # Colab notebook template
├── requirements.txt
└── README.md
```

## Configuration
All parameters in `src/config.py`:
- `SHARPNESS_TIER_GOLD = 0.20` — Logo clearly readable
- `SHARPNESS_TIER_SILVER = 0.12` — Slightly soft, logo identifiable
- `SHARPNESS_TIER_BRONZE = 0.06` — Mild blur, logo partially readable

## Workflow
```
1. Run extraction → 300-400 frames (Phase 1)
2. Upload to Roboflow → Manual annotation (21 classes)
3. Export YOLO format from Roboflow
4. Run propagation → ~2000 frames with annotations (Phase 1b)
5. [Future] Crop-based YOLO training (Phase 2)
```
