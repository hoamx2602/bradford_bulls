"""
Auto-calibration for team color classification.

Uses K-Means clustering on torso color histograms to automatically
discover team colors. Works with ANY jersey color — no hardcoded palettes.

Key improvements:
- Green masking: removes grass/pitch pixels before computing histograms
- Gaussian weighting: prioritizes center of torso (jersey, not edges)
- Outlier rejection: removes misclassified samples after initial clustering
- Close-up filtering: rejects extreme close-ups (coaches, staff)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


# ---------------------------------------------------------------------------
# Torso feature extraction
# ---------------------------------------------------------------------------

def _build_green_mask(hsv_img):
    """
    Create a binary mask that is 1 for NON-green pixels and 0 for green (grass).
    Green in HSV: H≈35-85, S>40, V>40.
    """
    lower_green = np.array([35, 40, 40], dtype=np.uint8)
    upper_green = np.array([85, 255, 255], dtype=np.uint8)
    green = cv2.inRange(hsv_img, lower_green, upper_green)
    return (green == 0).astype(np.float32)  # 1 = not green


def _build_gaussian_weights(h, w, sigma=0.4):
    """
    Create a 2D Gaussian weight map that emphasizes the center of the crop.
    sigma controls how much the edges are suppressed (smaller = more focused).
    """
    y = np.linspace(-1, 1, h)
    x = np.linspace(-1, 1, w)
    yy, xx = np.meshgrid(y, x, indexing='ij')
    weights = np.exp(-0.5 * (xx**2 + yy**2) / (sigma**2))
    return weights.astype(np.float32)


def extract_torso_features(torso_crop):
    """
    Extract color histogram features from a torso crop.
    
    Improvements over basic histogram:
    1. Green masking: grass pixels are excluded from the histogram
    2. Gaussian weighting: center pixels contribute more than edge pixels
    3. Uses HSV for robust color separation under varying lighting
    
    Returns: 300-dim feature vector (12×5×5 HSV histogram), or None
    """
    if torso_crop is None or torso_crop.size < 200:
        return None

    hsv = cv2.cvtColor(torso_crop, cv2.COLOR_BGR2HSV)
    h, w = torso_crop.shape[:2]

    # Build combined weight map: Gaussian center weight × green exclusion
    gaussian_w = _build_gaussian_weights(h, w, sigma=0.4)
    green_mask = _build_green_mask(hsv)
    combined_weights = gaussian_w * green_mask

    # Check: if too few valid pixels after masking, skip
    if combined_weights.sum() < 50:
        return None

    # Compute weighted histogram using numpy
    pixels = hsv.reshape(-1, 3).astype(np.float64)
    weights_flat = combined_weights.flatten()

    # Bin edges for H(0-180), S(0-256), V(0-256)
    bins = [12, 5, 5]
    ranges = [(0, 180), (0, 256), (0, 256)]

    hist, _ = np.histogramdd(
        pixels, bins=bins, range=ranges, weights=weights_flat
    )

    hist = hist.flatten()
    norm = np.linalg.norm(hist)
    if norm > 0:
        hist /= norm

    return hist.astype(np.float64)


# ---------------------------------------------------------------------------
# Torso crop extraction
# ---------------------------------------------------------------------------

def extract_torso_crop(frame, bbox, overlay_mask=None, strict=False):
    """
    Extract torso region from a person bounding box.
    
    Args:
        strict: If True (calibration), reject blurry crops. 
                If False (runtime), accept blurry crops for color matching.
    
    Returns: (torso_crop, status) where status is "ok" or error reason
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    bw, bh = x2 - x1, y2 - y1

    # Skip tiny detections
    if bw < 40 or bh < 60:
        return None, "too_small"

    # ---- Close-up / non-player filter ----
    # Coaches and staff in close-ups have large bboxes (>25% of frame area)
    # Real on-pitch players are much smaller
    frame_area = h * w
    bbox_area = bw * bh
    if bbox_area / frame_area > 0.25:
        return None, "close_up_likely_staff"

    # Players on pitch typically have height > 1.5× width (standing)
    # Close-up face shots or seated staff tend to be wider
    aspect = bh / bw if bw > 0 else 0
    if aspect < 1.0:
        return None, "bad_aspect_ratio"

    # Torso/Jersey ROI: 10-40% of person height = upper body (jersey area)
    ty1 = y1 + int(bh * 0.10)
    ty2 = y1 + int(bh * 0.40)
    tx1 = x1 + int(bw * 0.10)
    tx2 = x2 - int(bw * 0.10)

    # Clamp to frame bounds
    ty1, ty2 = max(0, ty1), min(h, ty2)
    tx1, tx2 = max(0, tx1), min(w, tx2)

    if ty2 - ty1 < 15 or tx2 - tx1 < 15:
        return None, "crop_too_small"

    torso = frame[ty1:ty2, tx1:tx2]

    # Check overlay contamination
    if overlay_mask is not None:
        mask_crop = overlay_mask[ty1:ty2, tx1:tx2]
        if mask_crop.mean() < 0.5:  # >50% overlay
            return None, "overlay_contaminated"

    # Skin check: reject if torso is mostly skin (shirtless/wrong crop)
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    skin_mask = cv2.inRange(hsv, (0, 30, 60), (25, 180, 255))
    if skin_mask.mean() / 255 > 0.6:
        return None, "mostly_skin"

    # Green check: reject if torso is mostly grass (bad crop alignment)
    green_mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    if green_mask.mean() / 255 > 0.5:
        return None, "mostly_grass"

    # Sharpness check: ONLY during calibration (strict mode)
    if strict:
        gray = cv2.cvtColor(torso, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if lap_var < 100:
            return None, "too_blurry"

    return torso, "ok"


# ---------------------------------------------------------------------------
# Cluster discovery
# ---------------------------------------------------------------------------

def _remove_outliers(X_scaled, labels, n_clusters, max_std=2.0):
    """
    Remove samples that are far from their cluster center (likely misclassified).
    
    Args:
        max_std: samples beyond this many standard deviations from their 
                 cluster center are flagged as outliers
    
    Returns: boolean mask (True = keep, False = outlier)
    """
    keep = np.ones(len(labels), dtype=bool)

    for c in range(n_clusters):
        mask = labels == c
        if mask.sum() < 3:
            continue

        cluster_points = X_scaled[mask]
        center = cluster_points.mean(axis=0)
        dists = np.linalg.norm(cluster_points - center, axis=1)

        threshold = dists.mean() + max_std * dists.std()
        cluster_outliers = dists > threshold

        # Map back to global indices
        global_indices = np.where(mask)[0]
        keep[global_indices[cluster_outliers]] = False

    return keep


def discover_clusters(video_path, yolo_model, device, overlay_mask=None,
                      n_sample_frames=80, min_person_area_ratio=0.02):
    """
    Step 1 of calibration: Discover team color clusters.
    
    Samples frames, detects persons, clusters torso colors, shows grid.
    Returns intermediate result — user then picks their team cluster
    in the next notebook cell.
    
    Pipeline:
    1. Sample N frames evenly across the video
    2. Detect persons with YOLO
    3. Extract torso crops (with close-up/staff filtering)
    4. Compute weighted histograms (green-masked, Gaussian-weighted)
    5. K-Means clustering
    6. Outlier rejection — remove misclassified samples
    7. Show clean grid for manual cluster selection
    
    Returns:
        cluster_data dict (pass to finalize_calibration), or None on failure
    """
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_area = frame_w * frame_h

    # Sample frames evenly (skip first/last 5%)
    start = int(total * 0.05)
    end = int(total * 0.95)
    indices = np.linspace(start, end, n_sample_frames, dtype=int)

    all_crops = []
    all_features = []
    reject_reasons = {}

    print(f"Calibration: sampling {n_sample_frames} frames...")
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

            if area / frame_area < min_person_area_ratio:
                continue

            torso, status = extract_torso_crop(frame, bbox, overlay_mask, strict=True)
            if status != "ok":
                reject_reasons[status] = reject_reasons.get(status, 0) + 1
                continue

            features = extract_torso_features(torso)
            if features is None:
                continue

            all_crops.append(cv2.resize(torso, (64, 64)))
            all_features.append(features)

    cap.release()

    n_crops = len(all_crops)
    print(f"Collected {n_crops} valid torso crops.")
    if reject_reasons:
        print(f"  Rejected: {reject_reasons}")

    if n_crops < 10:
        print("ERROR: Too few torso crops. Try lowering confidence.")
        return None

    # ---- K-Means clustering ----
    X = np.array(all_features)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    best_k, best_score, best_km, best_labels = 2, -1, None, None
    for k in [2, 3]:
        if n_crops < k * 3:
            continue
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        print(f"  K={k}: silhouette={score:.3f}")
        if score > best_score:
            best_k, best_score, best_km, best_labels = k, score, km, labels

    print(f"  → Selected K={best_k}")

    # ---- Outlier rejection ----
    keep_mask = _remove_outliers(X_scaled, best_labels, best_k, max_std=1.8)
    n_removed = (~keep_mask).sum()
    if n_removed > 0:
        print(f"  → Removed {n_removed} outlier samples")

        # Re-cluster without outliers for cleaner centroids
        X_clean = X_scaled[keep_mask]
        clean_crops = [c for c, k in zip(all_crops, keep_mask) if k]
        km_clean = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        clean_labels = km_clean.fit_predict(X_clean)

        # Use cleaned data from here on
        all_crops = clean_crops
        all_features = [f for f, k in zip(all_features, keep_mask) if k]
        best_labels = clean_labels
        best_km = km_clean
        X_scaled = X_clean
        n_crops = len(all_crops)

    # ---- Show grid ----
    _show_calibration_grid(all_crops, best_labels, best_k)

    # ---- Print summary ----
    cluster_sizes = {c: int((best_labels == c).sum()) for c in range(best_k)}
    print(f"\n{'='*55}")
    for c in range(best_k):
        # Compute dominant color description for each cluster
        desc = _describe_cluster_color(all_crops, best_labels, c)
        print(f"  Cluster {c}: {cluster_sizes[c]} crops — {desc}")
    print(f"{'='*55}")
    print(f"\n  ✏️  In the NEXT cell, set TARGET_CLUSTER to your")
    print(f"      team's cluster number (0-{best_k - 1}), then run it.\n")

    return {
        "kmeans": best_km,
        "scaler": scaler,
        "labels": best_labels,
        "n_clusters": best_k,
        "cluster_sizes": cluster_sizes,
        "n_crops_total": n_crops,
    }


def _describe_cluster_color(crops, labels, cluster_id):
    """
    Compute a human-readable color description for a cluster.
    Analyzes average HSV values to describe the dominant jersey color.
    """
    indices = np.where(labels == cluster_id)[0]
    if len(indices) == 0:
        return "empty"

    avg_h, avg_s, avg_v = [], [], []
    for i in indices[:20]:  # Sample up to 20 crops
        hsv = cv2.cvtColor(crops[i], cv2.COLOR_BGR2HSV)
        # Mask out green pixels
        green = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
        non_green = (green == 0)
        if non_green.sum() < 10:
            continue
        avg_h.append(hsv[non_green, 0].mean())
        avg_s.append(hsv[non_green, 1].mean())
        avg_v.append(hsv[non_green, 2].mean())

    if not avg_h:
        return "unknown"

    h, s, v = np.mean(avg_h), np.mean(avg_s), np.mean(avg_v)

    # Describe based on HSV
    if s < 40:
        if v > 170:
            return f"LIGHT / WHITE (V={v:.0f}, S={s:.0f})"
        elif v < 80:
            return f"DARK / BLACK (V={v:.0f}, S={s:.0f})"
        else:
            return f"GRAY (V={v:.0f}, S={s:.0f})"
    else:
        # Chromatic — describe by hue
        if h < 10 or h > 170:
            color = "RED"
        elif h < 25:
            color = "ORANGE"
        elif h < 35:
            color = "YELLOW"
        elif h < 85:
            color = "GREEN"
        elif h < 130:
            color = "BLUE"
        elif h < 155:
            color = "PURPLE"
        else:
            color = "PINK"
        return f"{color} (H={h:.0f}, S={s:.0f}, V={v:.0f})"


# ---------------------------------------------------------------------------
# Calibration finalization
# ---------------------------------------------------------------------------

def finalize_calibration(cluster_data, target_cluster):
    """
    Step 2 of calibration: Finalize with user's team choice.
    
    Args:
        cluster_data: dict returned by discover_clusters()
        target_cluster: int — cluster number the user identified as their team
    
    Returns:
        calibration dict ready for pipeline use
    """
    if cluster_data is None:
        print("ERROR: No cluster data. Run discover_clusters() first.")
        return None

    n_clusters = cluster_data["n_clusters"]
    if not (0 <= target_cluster < n_clusters):
        print(f"ERROR: target_cluster must be 0-{n_clusters - 1}, got {target_cluster}")
        return None

    calibration = {
        "kmeans": cluster_data["kmeans"],
        "scaler": cluster_data["scaler"],
        "target_cluster": target_cluster,
        "n_clusters": n_clusters,
        "cluster_sizes": cluster_data["cluster_sizes"],
        "n_crops_total": cluster_data["n_crops_total"],
    }

    sizes = cluster_data["cluster_sizes"]
    print(f"✅ Calibration finalized!")
    for c in range(n_clusters):
        role = "✅ TARGET" if c == target_cluster else "   opponent/other"
        print(f"   Cluster {c}: {sizes[c]} crops — {role}")

    return calibration


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _show_calibration_grid(crops, labels, n_clusters, samples_per_cluster=8):
    """
    Display sample torso crops grouped by cluster.
    
    Each cluster is a separate row with a clear label.
    Shows 8 samples per cluster for better visual confirmation.
    """
    fig, axes = plt.subplots(
        n_clusters, samples_per_cluster,
        figsize=(2.5 * samples_per_cluster, 4 * n_clusters)
    )
    if n_clusters == 1:
        axes = axes[np.newaxis, :]

    # Color-coded backgrounds per cluster
    row_colors = ['#FFE0E0', '#E0E0FF', '#E0FFE0']  # light red, blue, green

    for c in range(n_clusters):
        cluster_indices = np.where(labels == c)[0]
        n_show = min(samples_per_cluster, len(cluster_indices))

        if n_show > 0:
            show_indices = cluster_indices[
                np.linspace(0, len(cluster_indices) - 1, n_show, dtype=int)
            ]
        else:
            show_indices = []

        count = int((labels == c).sum())
        bg_color = row_colors[c % len(row_colors)]

        for j in range(samples_per_cluster):
            ax = axes[c, j]
            ax.set_facecolor(bg_color)

            if j < len(show_indices):
                crop = crops[show_indices[j]]
                ax.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

            ax.set_xticks([])
            ax.set_yticks([])

            # Bold title on first image
            if j == 0:
                ax.set_title(
                    f"── CLUSTER {c} ({count} crops) ──",
                    fontsize=13, fontweight='bold',
                    color=['red', 'blue', 'green'][c % 3],
                    pad=8
                )

            # Add border
            for spine in ax.spines.values():
                spine.set_edgecolor(bg_color)
                spine.set_linewidth(3)

    plt.suptitle(
        "TEAM IDENTIFICATION — Each row = one cluster\n"
        "Pick the cluster number that matches YOUR team",
        fontsize=16, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Runtime classification
# ---------------------------------------------------------------------------

def classify_person(torso_crop, calibration):
    """
    Classify a person's team using the learned calibration.
    
    Returns: (role, confidence) where role is "target", "opponent", or "ambiguous"
    """
    if calibration is None:
        return "ambiguous", 0.0

    features = extract_torso_features(torso_crop)
    if features is None:
        return "ambiguous", 0.0

    X = calibration["scaler"].transform([features])
    cluster = calibration["kmeans"].predict(X)[0]

    # Distance-based confidence
    center = calibration["kmeans"].cluster_centers_[cluster]
    dist = np.linalg.norm(X[0] - center)
    all_dists = [np.linalg.norm(X[0] - c)
                 for c in calibration["kmeans"].cluster_centers_]
    confidence = 1.0 - (dist / (sum(all_dists) + 1e-6))

    if confidence < 0.35:
        return "ambiguous", round(confidence, 3)

    if cluster == calibration["target_cluster"]:
        return "target", round(confidence, 3)
    else:
        return "opponent", round(confidence, 3)
