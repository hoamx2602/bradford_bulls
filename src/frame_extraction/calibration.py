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
    hist = cv2.normalize(hist, hist).flatten()
    return hist


def extract_torso_crop(frame, bbox, overlay_mask=None):
    """
    Extract torso region from a person bounding box.
    Includes quality checks to reject bad crops.
    
    Returns: (torso_crop, status) where status is "ok" or error reason
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    bw, bh = x2 - x1, y2 - y1

    # Skip tiny detections
    if bw < 30 or bh < 50:
        return None, "too_small"

    # Torso ROI: 20-55% of person height, 15-85% of width
    ty1 = y1 + int(bh * 0.20)
    ty2 = y1 + int(bh * 0.55)
    tx1 = x1 + int(bw * 0.15)
    tx2 = x2 - int(bw * 0.15)

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

    return torso, "ok"


def auto_calibrate(video_path, yolo_model, device, overlay_mask=None,
                   n_sample_frames=60, min_person_area_ratio=0.01):
    """
    Auto-calibrate team colors by clustering torso color histograms.
    
    Pipeline:
    1. Sample N frames evenly across video
    2. YOLO detect persons → crop torso regions
    3. Extract color histograms from each torso
    4. K-Means cluster into 2-3 groups
    5. Show clusters → user picks which is their team
    
    Args:
        video_path: Path to video file
        yolo_model: Loaded YOLO model
        device: "cuda" or "cpu"
        overlay_mask: Binary overlay mask (from overlay detection)
        n_sample_frames: Number of frames to sample
        min_person_area_ratio: Min person area / frame area to consider
    
    Returns:
        calibration dict with: kmeans, scaler, target_cluster, n_clusters
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

    all_crops = []       # Resized torso crops for visualization
    all_features = []    # Feature vectors for clustering
    crop_metadata = []   # (frame_idx, bbox) for reference

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

            torso, status = extract_torso_crop(frame, bbox, overlay_mask)
            if status != "ok":
                continue

            features = extract_torso_features(torso)
            if features is None:
                continue

            all_crops.append(cv2.resize(torso, (64, 64)))
            all_features.append(features)
            crop_metadata.append((idx, bbox.tolist()))

    cap.release()

    n_crops = len(all_crops)
    print(f"Collected {n_crops} valid torso crops for clustering.")

    if n_crops < 10:
        print("ERROR: Too few torso crops. Try lowering confidence or checking video.")
        return None

    # K-Means clustering
    X = np.array(all_features)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Try K=2 and K=3, pick best silhouette score
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

    print(f"  → Selected K={best_k} (silhouette={best_score:.3f})")

    # Print cluster summary as text (visible without scrolling)
    cluster_sizes = {c: int((best_labels == c).sum()) for c in range(best_k)}
    print(f"\n{'='*50}")
    print(f"  CLUSTERS FOUND:")
    for c in range(best_k):
        print(f"    Cluster {c}: {cluster_sizes[c]} player crops")
    print(f"{'='*50}")
    print(f"  ↓↓↓ SEE JERSEY SAMPLES BELOW ↓↓↓")
    print(f"  ↓↓↓ THEN SCROLL DOWN TO ENTER YOUR CHOICE ↓↓↓")
    print(f"{'='*50}\n")

    # Visualize clusters
    _show_calibration_grid(all_crops, best_labels, best_k)

    # Prominent message after grid
    print(f"\n{'🔻'*25}")
    print(f"  👇 ENTER YOUR TEAM'S CLUSTER NUMBER BELOW 👇")
    print(f"{'🔻'*25}\n")

    # User selects target team
    while True:
        try:
            target = int(input(
                f"→ Which cluster is YOUR TEAM? Enter number (0-{best_k - 1}): "
            ))
            if 0 <= target < best_k:
                break
            print(f"  Please enter a number between 0 and {best_k - 1}")
        except ValueError:
            print("  Please enter a valid number")

    # Build calibration result
    cluster_sizes = {c: int((best_labels == c).sum()) for c in range(best_k)}
    calibration = {
        "kmeans": best_km,
        "scaler": scaler,
        "target_cluster": target,
        "n_clusters": best_k,
        "cluster_sizes": cluster_sizes,
        "n_crops_total": n_crops,
    }

    target_count = cluster_sizes[target]
    print(f"\n✅ Calibration complete!")
    print(f"   Target team: Cluster {target} ({target_count} crops)")
    for c in range(best_k):
        role = "TARGET" if c == target else "opponent/other"
        print(f"   Cluster {c}: {cluster_sizes[c]} crops — {role}")

    return calibration


def _show_calibration_grid(crops, labels, n_clusters, samples_per_cluster=6):
    """Display sample torso crops per cluster for user identification."""
    fig, axes = plt.subplots(n_clusters, samples_per_cluster,
                             figsize=(3 * samples_per_cluster, 3.5 * n_clusters))
    if n_clusters == 1:
        axes = axes[np.newaxis, :]

    for c in range(n_clusters):
        cluster_indices = np.where(labels == c)[0]
        n_show = min(samples_per_cluster, len(cluster_indices))

        # Pick evenly spaced samples from cluster
        if n_show > 0:
            show_indices = cluster_indices[
                np.linspace(0, len(cluster_indices) - 1, n_show, dtype=int)
            ]
        else:
            show_indices = []

        for j in range(samples_per_cluster):
            ax = axes[c, j]
            if j < len(show_indices):
                crop = crops[show_indices[j]]
                ax.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            ax.axis("off")
            if j == 0:
                count = int((labels == c).sum())
                ax.set_ylabel(f"Cluster {c}\n({count} players)",
                              fontsize=12, fontweight="bold")

    plt.suptitle("AUTO-CALIBRATION — Which cluster is YOUR TEAM?",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
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
