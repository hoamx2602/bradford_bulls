"""
Auto-calibration for team color classification.

Uses K-Means clustering on torso color histograms to automatically
discover team colors. Works with ANY jersey color — no hardcoded palettes.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


def extract_torso_features(torso_crop):
    """
    Extract color histogram features from a torso crop.
    Uses HSV 3D histogram (12×5×5 = 300 dims) for robust color representation.
    """
    if torso_crop is None or torso_crop.size < 200:
        return None

    hsv = cv2.cvtColor(torso_crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None,
                        [12, 5, 5], [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten().astype(np.float64)
    return hist


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

    # Torso/Jersey ROI: 10-40% of person height = upper body (jersey area)
    # YOLO person bbox: head~0-10%, shoulders~10-25%, jersey~10-40%, shorts~40-60%
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

    # Sharpness check: ONLY during calibration (strict mode)
    if strict:
        gray = cv2.cvtColor(torso, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if lap_var < 100:
            return None, "too_blurry"

    return torso, "ok"


def discover_clusters(video_path, yolo_model, device, overlay_mask=None,
                      n_sample_frames=80, min_person_area_ratio=0.02):
    """
    Step 1 of calibration: Discover team color clusters.
    
    Samples frames, detects persons, clusters torso colors, shows grid.
    Returns intermediate result — user then picks their team cluster
    in the next notebook cell.
    
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
                continue

            features = extract_torso_features(torso)
            if features is None:
                continue

            all_crops.append(cv2.resize(torso, (64, 64)))
            all_features.append(features)

    cap.release()

    n_crops = len(all_crops)
    print(f"Collected {n_crops} valid torso crops.")

    if n_crops < 10:
        print("ERROR: Too few torso crops. Try lowering confidence.")
        return None

    # K-Means clustering
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

    # Show grid
    _show_calibration_grid(all_crops, best_labels, best_k)

    # Print summary
    cluster_sizes = {c: int((best_labels == c).sum()) for c in range(best_k)}
    print(f"\n{'='*55}")
    for c in range(best_k):
        print(f"  Cluster {c}: {cluster_sizes[c]} player crops")
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


def _show_calibration_grid(crops, labels, n_clusters, samples_per_cluster=6):
    """Display sample torso crops per cluster for user identification."""
    fig, axes = plt.subplots(n_clusters, samples_per_cluster,
                             figsize=(3 * samples_per_cluster, 4 * n_clusters))
    if n_clusters == 1:
        axes = axes[np.newaxis, :]

    for c in range(n_clusters):
        cluster_indices = np.where(labels == c)[0]
        n_show = min(samples_per_cluster, len(cluster_indices))

        if n_show > 0:
            show_indices = cluster_indices[
                np.linspace(0, len(cluster_indices) - 1, n_show, dtype=int)
            ]
        else:
            show_indices = []

        # Add row title for cluster
        count = int((labels == c).sum())
        # Add a text annotation to the far left
        fig.text(0.01, (n_clusters - c - 0.5) / n_clusters, 
                 f"CLUSTER {c}\n({count} crops)", 
                 fontsize=14, fontweight='bold', va='center', rotation=90)

        for j in range(samples_per_cluster):
            ax = axes[c, j]
            if j < len(show_indices):
                crop = crops[show_indices[j]]
                ax.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                if j == 0:
                    ax.set_title(f"CLUSTER {c}", color='red', fontsize=15, fontweight='bold')
            ax.axis("off")

    plt.suptitle("MANUAL SELECTION: Identify your team's cluster", 
                 fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.95])
    plt.show()


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
