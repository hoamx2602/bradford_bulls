# Bradford Bulls — Team-Aware Frame Extraction Technical Summary

This document outlines the technical strategies, algorithms, and models implemented in the `02_team_aware_extraction.ipynb` pipeline for high-quality logo training data extraction.

---

## Tech Stack & Dependencies

| Category | Library / Tool | Version | Role |
|---|---|---|---|
| **Object Detection** | `ultralytics` (YOLOv8) | `>=8.3.0` | Person detection in every frame |
| **Computer Vision** | `opencv-python` (cv2) | `>=4.9.0` | Frame reading, HSV conversion, morphological ops, Laplacian sharpness |
| **Deep Learning Runtime** | `torch` / `torchvision` | `>=2.2.0` / `>=0.17.0` | YOLO inference backend (CPU/GPU) |
| **ML / Clustering** | `scikit-learn` | `>=1.4.0` | `KMeans` (diverse sample selection), `StandardScaler` (feature normalisation) |
| **Frame Deduplication** | `imagehash` | `>=4.3.1` | Perceptual hashing (pHash) for near-duplicate detection |
| **Frame Similarity** | `scikit-image` | `>=0.22.0` | SSIM (Structural Similarity Index) |
| **Numerical Computing** | `numpy` | `>=1.26.0` | All array/matrix operations, histogramdd, distance metrics |
| **Visualisation** | `matplotlib` | `>=3.8.0` | Calibration grids, verification plots, verification overlays |
| **Video Download** | `yt-dlp` | `>=2024.12.0` | Downloading match footage from YouTube |
| **Auto-Annotation** | `autodistill-grounding-dino` | `>=0.1.0` | Zero-shot logo detection and labelling |
| **Annotation Management** | `roboflow` + `supervision` | `>=1.1.0` / `>=0.18.0` | Dataset upload, bounding box management |
| **Progress Bars** | `tqdm` | `>=4.66.0` | Pass 1 / Pass 2 scan progress |
| **Notebook UI** | `ipywidgets` + `gradio` | `>=8.1.0` / `>=4.44.0` | Interactive review and annotation UI |
| **Data** | `pandas` | `>=2.2.0` | Candidate metadata and selection statistics |

---

## Technical Models Used

### 1. YOLOv8 — Person Detection (`ultralytics`)

**Model**: `YOLOv8n` / `YOLOv8s` (configurable, `class=0` = person)

- Used in **both** Pass 1 (fast scan) and Pass 2 (detailed extraction).
- Runs in batch mode (`batch_size=32`) during Pass 1 for efficiency.
- Confidence threshold configurable via `PERSON_CONFIDENCE` (default `0.45`).
- GPU accelerated via PyTorch (`device` param passed through to `.predict()`).
- Only class `0` (person) is predicted — all other COCO classes are ignored.

---

### 2. Grounding DINO — Zero-Shot Logo Detection (`autodistill-grounding-dino`)

**Model**: `GroundingDINO` (transformer-based, open-vocabulary detection)

- Used in the downstream **auto-annotation** phase.
- Accepts natural language prompts (e.g., `"Bradford Bulls logo"`) to detect logos without labelled training data.
- Produces bounding box predictions exported to Roboflow for manual review.
- Requires `transformers<4.40.0` to avoid `BertModel.get_head_mask` compatibility errors.

---

## Techniques & Algorithms

### 1. Static Overlay Detection — `overlay.py` (Phase 0A)
To prevent the system from getting "confused" by scoreboards, logos, or captions that cover players, we use **Temporal Variance Analysis**.
- **Technique**: We sample frames evenly across the video and compute the variance of every pixel across time.
- **Logic**: Pixels that stay constant (low variance) are identified as static overlays.
- **Threshold**: Adaptive — bottom 3rd percentile of variance, floored at `8.0`.
- **Filtering**: We only apply this to the edges/corners (where broadcast graphics live) and use morphological operations (`MORPH_CLOSE` / `MORPH_OPEN`, 11×11 kernel) to create a clean binary mask.
- **Result**: A `white-list` mask where players are visible and graphics are blocked.

### 2. Semi-Supervised Team Calibration — `calibration.py` (Phase 0B)
Instead of unreliable unsupervised clustering (K-Means alone), we use a **Human-in-the-Loop** approach.

#### Feature Engineering
- **Weighted HSV Histograms**: We extract player torso crops (10–40% of bounding box height) and compute 3D HSV histograms (`bins=[12, 5, 5]`, 300-dim feature vector).
- **Gaussian Weighting**: A 2D Gaussian (σ=0.4) centered on the crop emphasises jersey centre pixels and suppresses edge pixels (background/limbs).
- **Grass Masking**: Green pixels (HSV range `[35–85, 40–255, 40–255]`) are masked out before computing the histogram to ensure only the jersey colour is captured.
- **Overlay Masking**: Pixels flagged by the static overlay mask are also zeroed out to prevent scoreboard text from contaminating the jersey colour signature.

#### Diverse Sampling for Display
- **Mini K-Means Centroid Selection** (`sklearn.cluster.KMeans`, `n_clusters=24`, `n_init=5`): We sample hundreds of players but use K-Means to select the 24 most *different* looking jerseys for the user to label.
- `StandardScaler` normalises features before clustering to prevent any histogram bin from dominating.

#### Calibration Model (k-NN)
1. User labels 3–5 grid numbers as "your team".
2. **Semi-supervised expansion**: Unlabelled samples within `median(intra-target distances)` of any labeled sample are added to the target set.
3. **Opponent reference set**: The farthest 70% of non-target samples become opponent references.
4. **Runtime classification**: `classify_person()` uses **k-NN** — minimum distance to any target reference vs. minimum distance to any opponent reference. Falls back to single centroid distance for backward compatibility.
5. **Ambiguity threshold**: A confidence below `0.53` returns `"ambiguous"`.

### 3. Two-Pass Extraction Pipeline — `pipeline.py` (Phases 1 & 2)
To handle long videos efficiently, we split the logic into two passes.

#### Pass 1: Fast Scan (Zoom Discovery)
- **Goal**: Identify only the parts of the video where the camera is actually zoomed into players (likely to have visible logos).
- **Optimisation**: Processes every `SCAN_INTERVAL` (default 5) frames; uses batched YOLO inference (`batch_size=32`).
- **Logic**: Records `max_person_area_ratio`. If a person takes up >1.5% of the frame area, we flag that segment for Pass 2.
- **Segment detection**: Consecutive qualifying frames are merged into segments (with a `SEGMENT_GAP_TOLERANCE=3` frame tolerance) and only segments ≥ `MIN_SEGMENT_FRAMES=2` are kept.

#### Pass 2: Detailed Extraction
Within "zoomed" segments, we process every frame with deep filters:
- **Smart Pitch Filter**: Checks for green grass (HSV: `[35–95, 40–255, 40–255]`) in the bottom 60% of frame. Bypassed for extreme close-ups (`max_person_area_ratio > 6%`) where the pitch may be hidden. Target team frames are always kept regardless of pitch visibility.
- **Foreground Filter**: Removes crowd/background detections by keeping only persons with area > 25% of median detected area and height > 5% of frame height.
- **Overlay-Masked Sharpness**: Computed via **Laplacian variance** (55% weight) and **Tenengrad** (Sobel gradient magnitude, 45% weight), evaluated only on player bounding box regions and masked against static overlays to avoid fake sharpness from high-contrast scoreboard text.

### 4. Team-Aware Scoring — `pipeline.py`
Every candidate frame is assigned a quality score (0.0–1.0) based on a weighted formula that varies by frame category:

| Category | Sharpness | Target Player Size | Team Count | Player Size | Category Bonus | Pitch Bonus |
|---|---|---|---|---|---|---|
| `target_*` | 25% | 25% | 20% | 15% | 10% | ~5% |
| `mixed` | 30% | — | 15% | 25% | 5% | ~5% |
| `opponent_*` / `ambiguous_*` | 40% | — | 15% | 30% | — | ~5% |

- **Shot type** (`closeup` / `medium` / `wide`) is determined by `max_person_area_ratio` (>12% = closeup, >5% = medium, else wide).

### 5. Quota-Based Selection & Diversity — `selection.py`
To prevent the dataset from being 90% "Player X standing still", we use **Selection Quotas**.

| Category | Default Quota |
|---|---|
| `target_closeup` | 30% |
| `target_medium` | 25% |
| `mixed` | 20% |
| `opponent_closeup` | 8% |
| `opponent_medium` | 7% |
| `target_wide` | 5% |
| `opponent_wide` | 5% |

- **Temporal Diversity**: Enforces `min_time_gap=1.5s` between selected frames to prevent bursts of nearly identical frames from a single play.
- **Perceptual Deduplication**: Uses `imagehash.phash()` with a configurable Hamming distance threshold (`dedup_hash_thresh=8`) to remove near-duplicate frames.
- **Auto-Targeting**: `auto_target_frames()` automatically computes target frame count as `video_duration_min × 5 frames/min`, capped to 60% of candidate pool and bounded between `[50, 600]` frames.

### 6. Multi-Metric Sharpness — `helpers.py`
Rather than relying on a single metric, sharpness is computed as a blended score:

```
sharpness = Laplacian_variance_score × 0.55 + Tenengrad_score × 0.45
```

- **Laplacian variance** (`cv2.Laplacian`, `CV_64F`): Sensitive to all edges; normalised to `[0, 1]` against a cap of 300.
- **Tenengrad** (Sobel gradients `gx² + gy²`): More robust to noise; normalised to `[0, 1]` against a cap of 5000.
- Hard blur cutoff: frames below `MOTION_BLUR_SHARPNESS_MIN=0.06` are discarded entirely.
