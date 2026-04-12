# Bradford Bulls — Team-Aware Frame Extraction Technical Summary

This document outlines the technical strategies and algorithms implemented in the `02_team_aware_extraction.ipynb` pipeline for high-quality logo training data extraction.

## 1. Static Overlay Detection (Phase 0A)
To prevent the system from getting "confused" by scoreboards, logos, or captions that cover players, we use **Temporal Variance Analysis**.
- **Technique**: We sample frames evenly across the video and compute the variance of every pixel across time.
- **Logic**: Pixels that stay constant (low variance) are identified as static overlays.
- **Filtering**: We only apply this to the edges/corners (where broadcast graphics live) and use morphological operations (Closing/Opening) to create a clean binary mask.
- **Result**: A `white-list` mask where players are visible and graphics are blocked.

## 2. Semi-Supervised Team Calibration (Phase 0B)
Instead of unreliable unsupervised clustering (K-Means), we use a **Human-in-the-Loop** approach.
- **Feature Engineering**:
    - **Weighted HSV Histograms**: We extract player torso crops and compute HSV histograms.
    - **Gaussian Weighting**: Center pixels (jersey area) are weighted higher than edges (background/limbs).
    - **Grass Masking**: Green pixels (pitch) are masked out before computing the histogram to ensure we only see the jersey color.
- **Diverse Sampling**: We sample hundreds of players but use **K-Means Centroid selection** to show the user the 24 most *different* looking jerseys for labeling.
- **Classification**: Once the user labels 3-5 jerseys (e.g., "Bradford Bulls are #1, #4, #7"), the system computes a **Target Centroid** and an **Opponent Centroid** in feature space. Every future detection is classified by its distance to these centroids.

## 3. Two-Pass Extraction Pipeline (Phases 1 & 2)
To handle long videos efficiently, we split the logic into two passes.

### Pass 1: Fast Scan (Zoom Discovery)
- **Goal**: Identify only the parts of the video where the camera is actually zoomed into players (likely to have visible logos).
- **Optimization**: We skip every 5 frames and skip the expensive full detection.
- **Logic**: We record the `max_person_area_ratio`. If a person takes up $>1.5\%$ of the frame area, we flag that segment for Pass 2.

### Pass 2: Detailed Extraction
Within the "zoomed" segments, we process every frame with deep filters:
- **Smart Pitch Filter**: We check for "green grass" in the bottom of the frame to skip crowd/coaches/graphics. This is bypassed for extreme player close-ups where the pitch might be hidden.
- **Foreground Filter**: We filter out tiny background people (noise) and focus only on the main players.
- **Overlay-Masked Sharpness**: We compute Laplacian variance (sharpness) ONLY on the player regions, masking out static graphics to avoid "fake" sharpness from high-contrast scoreboard text.

## 4. Team-Aware Scoring
Every candidate frame is assigned a quality score (0.0 to 1.0) based on:
- **Sharpness**: Prioritizing frames with no motion blur.
- **Team Presence**: Higher scores for frames containing more "Target Team" (Bradford Bulls) players.
- **Size**: Bonus points for larger players (more pixel detail for logo training).
- **Dominance**: A "Target Closeup" is scored higher than a "Mixed Wide" shot.

## 5. Quota-Based Selection & Diversity
To prevent the dataset from being 90% "Player X standing still", we use **Selection Quotas**.
- **Balanced Mix**: We aim for a specific distribution (e.g., 65% Target Team, 20% Mixed, 15% Opponent) to ensure the AI learns to distinguish the team and sees the logo in various contexts.
- **Temporal Diversity**: We enforce a minimum time gap (e.g., 1.5 seconds) between selected frames. This prevents "bursts" of nearly identical frames from a single play.
- **Auto-Targeting**: The system automatically determines the target number of frames based on video duration (e.g., 5 frames per minute) to maintain consistent data density across different match lengths.
