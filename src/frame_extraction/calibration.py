"""
Semi-supervised team color calibration.

Instead of relying on unsupervised K-Means (which often produces impure clusters),
this module uses a HUMAN-IN-THE-LOOP approach:

  1. Sample diverse torso crops from the video
  2. Display a numbered grid for the user to inspect
  3. User labels a few examples of their team (by number)
  4. System learns the team color signature and classifies all future detections

This is fundamentally more robust because:
  - Human provides ground truth → no ambiguity
  - Works for ANY jersey design (solid, striped, patterned)
  - Only needs 3-5 labeled examples per team
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# ═══════════════════════════════════════════════════════════════════════════
# Feature extraction
# ═══════════════════════════════════════════════════════════════════════════

def _build_green_mask(hsv_img):
    """Binary mask: 1 = non-green pixel, 0 = grass/green."""
    green = cv2.inRange(hsv_img, (35, 40, 40), (85, 255, 255))
    return (green == 0).astype(np.float32)


def _build_gaussian_weights(h, w, sigma=0.4):
    """2D Gaussian centered on crop — emphasizes jersey center, suppresses edges."""
    y = np.linspace(-1, 1, h)
    x = np.linspace(-1, 1, w)
    yy, xx = np.meshgrid(y, x, indexing='ij')
    return np.exp(-0.5 * (xx**2 + yy**2) / sigma**2).astype(np.float32)


def extract_torso_features(torso_crop):
    """
    Compute a 300-dim weighted HSV histogram for the torso crop.
    
    Green pixels (grass) are masked out, and center pixels are weighted
    higher than edges (Gaussian weighting).
    """
    if torso_crop is None or torso_crop.size < 200:
        return None

    hsv = cv2.cvtColor(torso_crop, cv2.COLOR_BGR2HSV)
    h, w = torso_crop.shape[:2]

    # Combined weight: center emphasis × green exclusion
    weights = _build_gaussian_weights(h, w, 0.4) * _build_green_mask(hsv)

    if weights.sum() < 50:
        return None

    pixels = hsv.reshape(-1, 3).astype(np.float64)
    w_flat = weights.flatten()

    hist, _ = np.histogramdd(
        pixels, bins=[12, 5, 5],
        range=[(0, 180), (0, 256), (0, 256)],
        weights=w_flat
    )
    hist = hist.flatten()
    norm = np.linalg.norm(hist)
    if norm > 0:
        hist /= norm
    return hist.astype(np.float64)


# ═══════════════════════════════════════════════════════════════════════════
# Torso crop extraction
# ═══════════════════════════════════════════════════════════════════════════

def extract_torso_crop(frame, bbox, overlay_mask=None, strict=False):
    """
    Extract the jersey region from a YOLO person bounding box.

    Filters applied:
      - Too small (bbox < 40×60 px)
      - Close-up / staff (bbox > 20% of frame = likely coach/commentator)
      - Bad aspect ratio (width > height = not a standing player)
      - Overlay contaminated (scoreboard/graphics region)
      - Mostly skin (shirtless or wrong crop)
      - Mostly grass (misaligned crop)
      - Too blurry (only in strict/calibration mode)

    Args:
        strict: True during calibration (rejects blurry); False at runtime
    Returns:
        (torso_crop, status_string)
    """
    fh, fw = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    bw, bh = x2 - x1, y2 - y1

    if bw < 40 or bh < 60:
        return None, "too_small"

    # Close-up filter: large bbox = likely TV close-up of staff/coach
    if (bw * bh) / (fh * fw) > 0.20:
        return None, "close_up"

    # Standing player check: height should be > width
    if bh / max(bw, 1) < 1.0:
        return None, "bad_aspect"

    # Jersey ROI: 10-40% of person height
    ty1 = max(0, y1 + int(bh * 0.10))
    ty2 = min(fh, y1 + int(bh * 0.40))
    tx1 = max(0, x1 + int(bw * 0.10))
    tx2 = min(fw, x2 - int(bw * 0.10))

    if ty2 - ty1 < 15 or tx2 - tx1 < 15:
        return None, "crop_too_small"

    torso = frame[ty1:ty2, tx1:tx2]

    # Overlay check
    if overlay_mask is not None:
        if overlay_mask[ty1:ty2, tx1:tx2].mean() < 0.5:
            return None, "overlay"

    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)

    # Skin check
    skin = cv2.inRange(hsv, (0, 30, 60), (25, 180, 255))
    if skin.mean() / 255 > 0.6:
        return None, "mostly_skin"

    # Grass check
    grass = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    if grass.mean() / 255 > 0.5:
        return None, "mostly_grass"

    # Sharpness (calibration only)
    if strict:
        gray = cv2.cvtColor(torso, cv2.COLOR_BGR2GRAY)
        if cv2.Laplacian(gray, cv2.CV_64F).var() < 100:
            return None, "blurry"

    return torso, "ok"


# ═══════════════════════════════════════════════════════════════════════════
# Step 1: Collect diverse samples
# ═══════════════════════════════════════════════════════════════════════════

def collect_samples(video_path, yolo_model, device, overlay_mask=None,
                    n_sample_frames=80, n_display=24):
    """
    Collect torso crops from the video and select a diverse subset for labeling.
    
    Returns a dict containing:
      - 'all_crops': list of all valid torso crops (64×64 BGR)
      - 'all_features': list of feature vectors
      - 'display_indices': indices of the diverse subset to show the user
      - 'display_crops': the crops to display (ordered by diversity)
    """
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_area = fw * fh

    start = int(total * 0.05)
    end = int(total * 0.95)
    indices = np.linspace(start, end, n_sample_frames, dtype=int)

    all_crops, all_features = [], []
    reject_counts = {}

    print(f"Sampling {n_sample_frames} frames for calibration...")
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        results = yolo_model.predict(frame, classes=[0], conf=0.5,
                                     device=device, verbose=False)
        if not results or results[0].boxes is None:
            continue

        for box in results[0].boxes:
            bbox = box.xyxy[0].cpu().numpy()
            area = float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
            if area / frame_area < 0.02:
                continue

            torso, status = extract_torso_crop(frame, bbox, overlay_mask, strict=True)
            if status != "ok":
                reject_counts[status] = reject_counts.get(status, 0) + 1
                continue

            feat = extract_torso_features(torso)
            if feat is None:
                continue

            all_crops.append(cv2.resize(torso, (64, 64)))
            all_features.append(feat)

    cap.release()
    n_total = len(all_crops)
    print(f"Collected {n_total} valid torso crops.")
    if reject_counts:
        print(f"  Filtered out: {reject_counts}")

    if n_total < 10:
        print("ERROR: Too few crops. Try lowering YOLO confidence or min_person_area_ratio.")
        return None

    # ── Select diverse subset for display ──
    # Use mini K-Means to pick maximally diverse samples
    X = np.array(all_features)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_show = min(n_display, n_total)
    if n_total <= n_display:
        display_indices = list(range(n_total))
    else:
        # Pick diverse samples: run K-Means with K=n_display, 
        # then pick the sample closest to each centroid
        km = KMeans(n_clusters=n_show, random_state=42, n_init=5)
        km.fit(X_scaled)
        display_indices = []
        for i in range(n_show):
            cluster_members = np.where(km.labels_ == i)[0]
            center = km.cluster_centers_[i]
            dists = np.linalg.norm(X_scaled[cluster_members] - center, axis=1)
            display_indices.append(int(cluster_members[dists.argmin()]))

    return {
        "all_crops": all_crops,
        "all_features": all_features,
        "scaler": scaler,
        "X_scaled": X_scaled,
        "display_indices": display_indices,
        "n_total": n_total,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Step 2: Show numbered grid
# ═══════════════════════════════════════════════════════════════════════════

def show_samples(sample_data):
    """
    Display a numbered grid of diverse torso crop samples.
    The user will pick which numbers belong to their team.
    """
    if sample_data is None:
        print("ERROR: No sample data. Run collect_samples() first.")
        return

    indices = sample_data["display_indices"]
    crops = sample_data["all_crops"]
    n = len(indices)

    # 4 columns layout
    cols = min(6, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.5 * rows))
    if rows == 1:
        axes = axes[np.newaxis, :]
    if cols == 1:
        axes = axes[:, np.newaxis]

    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        if i < n:
            idx = indices[i]
            crop = crops[idx]
            ax.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            # Big clear number
            ax.set_title(f"#{i}", fontsize=16, fontweight='bold',
                        color='white',
                        bbox=dict(boxstyle='round,pad=0.3', 
                                  facecolor='black', alpha=0.8))
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle(
        f"PICK YOUR TEAM — Write down the numbers of YOUR team's jerseys\n"
        f"(Total {sample_data['n_total']} crops sampled from video)",
        fontsize=15, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 60)
    print("  📋 In the NEXT cell, set MY_TEAM to a list of numbers")
    print("     that show YOUR team's jersey. Example:")
    print("     MY_TEAM = [0, 3, 5, 8, 11]")
    print("=" * 60)


# ═══════════════════════════════════════════════════════════════════════════
# Step 3: Build calibration from user labels
# ═══════════════════════════════════════════════════════════════════════════

def build_calibration(sample_data, my_team_indices):
    """
    Build a calibration model from user-labeled samples.
    
    Uses the labeled examples as seeds to compute team centroids,
    then classifies ALL crops to verify the separation quality.
    
    Args:
        sample_data: dict from collect_samples()
        my_team_indices: list of grid numbers the user identified as their team
                         (numbers from the displayed grid, NOT raw crop indices)
    
    Returns:
        calibration dict for use in the pipeline
    """
    if sample_data is None:
        print("ERROR: No sample data.")
        return None

    display_indices = sample_data["display_indices"]
    X_scaled = sample_data["X_scaled"]
    scaler = sample_data["scaler"]
    crops = sample_data["all_crops"]
    n_total = sample_data["n_total"]

    # Convert grid numbers → actual crop indices
    target_crop_indices = set()
    for grid_num in my_team_indices:
        if 0 <= grid_num < len(display_indices):
            target_crop_indices.add(display_indices[grid_num])
        else:
            print(f"  ⚠️ Grid number {grid_num} out of range, skipping")

    if len(target_crop_indices) < 2:
        print("ERROR: Need at least 2 labeled samples. Try again.")
        return None

    # ── Compute centroids ──
    target_features = X_scaled[list(target_crop_indices)]
    target_centroid = target_features.mean(axis=0)

    # Everything not labeled as target → opponent seed
    opponent_indices = [i for i in range(n_total) if i not in target_crop_indices]
    
    # But we also want to be smart: not everything is opponent.
    # Some unlabeled might actually be target team.
    # So we first classify all unlabeled by distance to target centroid,
    # then use the FARTHEST ones as opponent seeds.
    
    # Compute distances of all samples to target centroid
    all_dists_to_target = np.linalg.norm(X_scaled - target_centroid, axis=1)
    
    # Opponent seeds: top 50% farthest from target centroid (among unlabeled)
    unlabeled_dists = [(i, all_dists_to_target[i]) for i in opponent_indices]
    unlabeled_dists.sort(key=lambda x: x[1], reverse=True)
    n_opp_seeds = max(3, len(unlabeled_dists) // 2)
    opponent_seed_indices = [i for i, _ in unlabeled_dists[:n_opp_seeds]]
    
    opponent_centroid = X_scaled[opponent_seed_indices].mean(axis=0)

    # ── Classify all samples ──
    labels = np.zeros(n_total, dtype=int)  # 0 = target, 1 = opponent
    for i in range(n_total):
        d_target = np.linalg.norm(X_scaled[i] - target_centroid)
        d_opponent = np.linalg.norm(X_scaled[i] - opponent_centroid)
        labels[i] = 0 if d_target < d_opponent else 1

    n_target = (labels == 0).sum()
    n_opponent = (labels == 1).sum()

    # ── Show verification grid ──
    _show_verification(crops, labels, target_crop_indices)

    print(f"\n✅ Calibration built from {len(target_crop_indices)} labeled samples!")
    print(f"   Target team:  {n_target} crops ({n_target/n_total*100:.0f}%)")
    print(f"   Opponent:     {n_opponent} crops ({n_opponent/n_total*100:.0f}%)")

    return {
        "target_centroid": target_centroid,
        "opponent_centroid": opponent_centroid,
        "scaler": scaler,
        "target_cluster": 0,  # 0 = target by convention
        "n_clusters": 2,
        "cluster_sizes": {0: int(n_target), 1: int(n_opponent)},
        "n_crops_total": n_total,
    }


def _show_verification(crops, labels, labeled_indices):
    """Show a verification grid: target vs opponent rows with labeled samples highlighted."""
    fig, axes = plt.subplots(2, 10, figsize=(25, 6))

    for row, (team_name, team_label, color) in enumerate([
        ("YOUR TEAM (target)", 0, '#2ECC71'),
        ("OPPONENT", 1, '#E74C3C'),
    ]):
        team_indices = np.where(labels == team_label)[0]
        n_show = min(10, len(team_indices))
        if n_show > 0:
            show = team_indices[np.linspace(0, len(team_indices)-1, n_show, dtype=int)]
        else:
            show = []

        for j in range(10):
            ax = axes[row, j]
            if j < len(show):
                idx = show[j]
                crop = crops[idx]
                ax.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

                # Highlight user-labeled samples with a gold border
                if idx in labeled_indices:
                    for spine in ax.spines.values():
                        spine.set_edgecolor('gold')
                        spine.set_linewidth(4)
                else:
                    for spine in ax.spines.values():
                        spine.set_edgecolor(color)
                        spine.set_linewidth(2)
            else:
                ax.set_facecolor('#f0f0f0')

            ax.set_xticks([])
            ax.set_yticks([])

            if j == 0:
                ax.set_title(team_name, fontsize=13, fontweight='bold',
                            color=color, pad=8)

    plt.suptitle(
        "VERIFICATION — Gold borders = your labeled samples\n"
        "Check that team separation looks correct",
        fontsize=14, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════
# Runtime classification (used by pipeline.py)
# ═══════════════════════════════════════════════════════════════════════════

def classify_person(torso_crop, calibration):
    """
    Classify a person as target team, opponent, or ambiguous.
    
    Uses centroid distance — the torso is compared to the learned
    target and opponent color signatures.
    
    Returns: (role, confidence)
      - role: "target", "opponent", or "ambiguous"
      - confidence: 0.0 to 1.0
    """
    if calibration is None:
        return "ambiguous", 0.0

    features = extract_torso_features(torso_crop)
    if features is None:
        return "ambiguous", 0.0

    X = calibration["scaler"].transform([features])[0]

    # Distance to each centroid
    d_target = np.linalg.norm(X - calibration["target_centroid"])
    d_opponent = np.linalg.norm(X - calibration["opponent_centroid"])

    # Confidence: how much closer to one centroid vs the other
    total = d_target + d_opponent + 1e-9
    if d_target < d_opponent:
        confidence = 1.0 - (d_target / total)
        role = "target"
    else:
        confidence = 1.0 - (d_opponent / total)
        role = "opponent"

    # Low confidence → ambiguous
    if confidence < 0.55:
        return "ambiguous", round(confidence, 3)

    return role, round(confidence, 3)


# ═══════════════════════════════════════════════════════════════════════════
# Legacy compatibility
# ═══════════════════════════════════════════════════════════════════════════

def discover_clusters(*args, **kwargs):
    """Deprecated. Use collect_samples() + show_samples() instead."""
    print("⚠️  discover_clusters() is deprecated.")
    print("   Use the new 3-step flow:")
    print("   1. sample_data = collect_samples(...)")
    print("   2. show_samples(sample_data)")
    print("   3. calibration = build_calibration(sample_data, MY_TEAM)")
    return None


def finalize_calibration(*args, **kwargs):
    """Deprecated. Use build_calibration() instead."""
    print("⚠️  finalize_calibration() is deprecated. Use build_calibration().")
    return None
