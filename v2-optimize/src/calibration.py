"""
Bradford Bulls v2 — Semi-supervised team color calibration.

Human-in-the-loop approach:
  1. Sample diverse torso crops from the video
  2. Display a numbered grid for the user to inspect
  3. User labels a few examples of their team (by number)
  4. System learns the team color signature and classifies all future detections

Ported from v1 calibration.py — identical logic, cleaner imports.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# ═══════════════════════════════════════════════════════════════════════
# Feature extraction
# ═══════════════════════════════════════════════════════════════════════

def _build_green_mask(hsv_img):
    """Binary mask: 1 = non-green pixel, 0 = grass/green."""
    green = cv2.inRange(hsv_img, (35, 40, 40), (85, 255, 255))
    return (green == 0).astype(np.float32)


def _build_gaussian_weights(h, w, sigma=0.4):
    """2D Gaussian centered on crop — emphasizes jersey center."""
    y = np.linspace(-1, 1, h)
    x = np.linspace(-1, 1, w)
    yy, xx = np.meshgrid(y, x, indexing='ij')
    return np.exp(-0.5 * (xx**2 + yy**2) / sigma**2).astype(np.float32)


def extract_torso_features(torso_crop, overlay_crop_mask=None):
    """Compute a 300-dim weighted HSV histogram for the torso crop."""
    if torso_crop is None or torso_crop.size < 200:
        return None

    hsv = cv2.cvtColor(torso_crop, cv2.COLOR_BGR2HSV)
    h, w = torso_crop.shape[:2]

    weights = _build_gaussian_weights(h, w, 0.4) * _build_green_mask(hsv)

    if overlay_crop_mask is not None:
        if overlay_crop_mask.shape[:2] != (h, w):
            overlay_crop_mask = cv2.resize(
                overlay_crop_mask.astype(np.float32), (w, h),
                interpolation=cv2.INTER_NEAREST
            )
        weights *= overlay_crop_mask.astype(np.float32)

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


# ═══════════════════════════════════════════════════════════════════════
# Torso crop extraction
# ═══════════════════════════════════════════════════════════════════════

def extract_torso_crop(frame, bbox, overlay_mask=None, strict=False):
    """
    Extract the jersey region from a YOLO person bounding box.

    Returns:
        (torso_crop, status_string, torso_overlay_mask)
    """
    fh, fw = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    bw, bh = x2 - x1, y2 - y1

    if bw < 40 or bh < 60:
        return None, "too_small", None

    if (bw * bh) / (fh * fw) > 0.20:
        return None, "close_up", None

    if bh / max(bw, 1) < 1.0:
        return None, "bad_aspect", None

    # Jersey ROI: 10-40% of person height
    ty1 = max(0, y1 + int(bh * 0.10))
    ty2 = min(fh, y1 + int(bh * 0.40))
    tx1 = max(0, x1 + int(bw * 0.10))
    tx2 = min(fw, x2 - int(bw * 0.10))

    if ty2 - ty1 < 15 or tx2 - tx1 < 15:
        return None, "crop_too_small", None

    torso = frame[ty1:ty2, tx1:tx2]

    torso_overlay_crop = None
    if overlay_mask is not None:
        torso_overlay_crop = overlay_mask[ty1:ty2, tx1:tx2]
        clean_ratio = torso_overlay_crop.mean()
        if clean_ratio < 0.75:
            return None, "overlay", None

    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)

    skin = cv2.inRange(hsv, (0, 30, 60), (25, 180, 255))
    if skin.mean() / 255 > 0.6:
        return None, "mostly_skin", None

    grass = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    if grass.mean() / 255 > 0.5:
        return None, "mostly_grass", None

    if strict:
        gray = cv2.cvtColor(torso, cv2.COLOR_BGR2GRAY)
        if cv2.Laplacian(gray, cv2.CV_64F).var() < 100:
            return None, "blurry", None

    return torso, "ok", torso_overlay_crop


# ═══════════════════════════════════════════════════════════════════════
# Step 1: Collect diverse samples
# ═══════════════════════════════════════════════════════════════════════

def collect_samples(video_path, yolo_model, device, overlay_mask=None,
                    n_sample_frames=80, n_display=24):
    """
    Collect torso crops from the video and select a diverse subset for labeling.

    Returns a dict containing:
      - 'all_crops': list of all valid torso crops (64×64 BGR)
      - 'all_features': list of feature vectors
      - 'display_indices': indices of the diverse subset to show the user
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

            torso, status, torso_ov_mask = extract_torso_crop(
                frame, bbox, overlay_mask, strict=True)
            if status != "ok":
                reject_counts[status] = reject_counts.get(status, 0) + 1
                continue

            feat = extract_torso_features(torso, torso_ov_mask)
            if feat is None:
                continue

            all_crops.append(cv2.resize(torso, (128, 128)))
            all_features.append(feat)

    cap.release()
    n_total = len(all_crops)
    print(f"Collected {n_total} valid torso crops.")
    if reject_counts:
        print(f"  Filtered out: {reject_counts}")

    if n_total < 10:
        print("ERROR: Too few crops. Try lowering YOLO confidence or min_person_area_ratio.")
        return None

    X = np.array(all_features)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_show = min(n_display, n_total)
    if n_total <= n_display:
        display_indices = list(range(n_total))
    else:
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


# ═══════════════════════════════════════════════════════════════════════
# Step 2: Show numbered grid
# ═══════════════════════════════════════════════════════════════════════

def _get_dominant_color(crop_bgr):
    """Get the dominant non-green jersey color as a hex string + RGB tuple."""
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    # Mask out green (grass) and very dark (shadows) pixels
    green = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    dark = hsv[:, :, 2] < 30
    valid = (green == 0) & (~dark)
    if valid.sum() < 20:
        return '#808080', (128, 128, 128)

    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pixels = rgb[valid]
    mean_color = pixels.mean(axis=0).astype(int)
    r, g, b = mean_color
    return f'#{r:02x}{g:02x}{b:02x}', (r, g, b)


def show_samples(sample_data):
    """Display a numbered grid of diverse torso crop samples with color bars."""
    if sample_data is None:
        print("ERROR: No sample data. Run collect_samples() first.")
        return

    indices = sample_data["display_indices"]
    crops = sample_data["all_crops"]
    n = len(indices)

    cols = min(6, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 4.5 * rows))
    if rows == 1:
        axes = axes[np.newaxis, :]
    if cols == 1:
        axes = axes[:, np.newaxis]

    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        if i < n:
            idx = indices[i]
            crop = crops[idx]
            # Show crop at full 128×128 resolution
            ax.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            # Dominant color bar at bottom
            hex_color, _ = _get_dominant_color(crop)
            ax.axhspan(crop.shape[0] - 12, crop.shape[0], color=hex_color, alpha=0.9)
            ax.set_title(f"#{i}", fontsize=16, fontweight='bold',
                        color='white',
                        bbox=dict(boxstyle='round,pad=0.3',
                                  facecolor='black', alpha=0.8))
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle(
        f"PICK YOUR TEAM — Write down the numbers of YOUR team's jerseys\n"
        f"Color bar at bottom shows dominant jersey color\n"
        f"(Total {sample_data['n_total']} crops sampled from video)",
        fontsize=15, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    plt.show()


def confirm_selection(sample_data, my_team_indices):
    """
    Show the user exactly which crops they selected, at large size,
    so they can verify before proceeding.

    Returns:
        my_team_indices (potentially filtered if user says some are wrong)
    """
    display_indices = sample_data["display_indices"]
    crops = sample_data["all_crops"]

    valid = [i for i in my_team_indices if 0 <= i < len(display_indices)]
    n = len(valid)
    if n == 0:
        print("ERROR: No valid selections.")
        return []

    cols = min(n, 6)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4.5 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    print(f"\n  You selected {n} crops:")
    for i, grid_num in enumerate(valid):
        ax = axes[i // cols, i % cols]
        idx = display_indices[grid_num]
        crop = crops[idx]
        ax.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        hex_color, (r, g, b) = _get_dominant_color(crop)
        # Thick color border
        for spine in ax.spines.values():
            spine.set_edgecolor(hex_color)
            spine.set_linewidth(5)
        # Brightness label
        brightness = 0.299 * r + 0.587 * g + 0.114 * b
        color_label = "LIGHT" if brightness > 140 else "DARK"
        ax.set_title(f"#{grid_num} — {color_label} ({hex_color})",
                    fontsize=12, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide empty axes
    for i in range(n, rows * cols):
        axes[i // cols, i % cols].set_visible(False)

    plt.suptitle(
        "⚠️ CONFIRM — Are ALL these crops from THE SAME team?\n"
        "If any crop is from the OPPONENT, remove its number and re-run.",
        fontsize=15, fontweight='bold', color='#e74c3c', y=1.02
    )
    plt.tight_layout()
    plt.show()

    # Print color consistency check
    colors = []
    for grid_num in valid:
        idx = display_indices[grid_num]
        _, (r, g, b) = _get_dominant_color(crops[idx])
        brightness = 0.299 * r + 0.587 * g + 0.114 * b
        colors.append((grid_num, brightness, r, g, b))

    brightnesses = [c[1] for c in colors]
    bright_range = max(brightnesses) - min(brightnesses)

    if bright_range > 80:
        print(f"\n  ⚠️  WARNING: Your selections have VERY different brightness levels!")
        print(f"      Brightness range: {min(brightnesses):.0f} — {max(brightnesses):.0f} "
              f"(diff={bright_range:.0f})")
        light_crops = [c[0] for c in colors if c[1] > 140]
        dark_crops = [c[0] for c in colors if c[1] <= 140]
        if light_crops and dark_crops:
            print(f"      LIGHT crops: {light_crops}")
            print(f"      DARK crops:  {dark_crops}")
            print(f"      → Are these really all the same team?")
            print(f"      → If not, remove the wrong ones and re-run.")
    else:
        print(f"  ✅ Color consistency OK (brightness range: {bright_range:.0f})")

    return valid


# ═══════════════════════════════════════════════════════════════════════
# Step 3: Build calibration from user labels
# ═══════════════════════════════════════════════════════════════════════

def build_calibration(sample_data, my_team_indices):
    """
    Build a calibration model from user-labeled samples.
    Uses k-NN approach for robust intra-class variety handling.
    """
    if sample_data is None:
        print("ERROR: No sample data.")
        return None

    display_indices = sample_data["display_indices"]
    X_scaled = sample_data["X_scaled"]
    scaler = sample_data["scaler"]
    crops = sample_data["all_crops"]
    n_total = sample_data["n_total"]

    target_crop_indices = set()
    for grid_num in my_team_indices:
        if 0 <= grid_num < len(display_indices):
            target_crop_indices.add(display_indices[grid_num])
        else:
            print(f"  ⚠️ Grid number {grid_num} out of range, skipping")

    if len(target_crop_indices) < 2:
        print("ERROR: Need at least 2 labeled samples. Try again.")
        return None

    labeled_target_features = X_scaled[list(target_crop_indices)]

    min_dist_to_target = np.full(n_total, np.inf)
    for tf in labeled_target_features:
        dists = np.linalg.norm(X_scaled - tf, axis=1)
        min_dist_to_target = np.minimum(min_dist_to_target, dists)

    labeled_list = list(target_crop_indices)

    # Use DISPLAY-based ratio (how many grid items the user labeled)
    # NOT total-based (which underestimates when total >> displayed)
    display_label_ratio = len(my_team_indices) / max(len(display_indices), 1)
    total_label_ratio = len(target_crop_indices) / max(n_total, 1)
    print(f"  Label ratio: {len(my_team_indices)}/{len(display_indices)} "
          f"displayed ({display_label_ratio:.0%}), "
          f"{len(target_crop_indices)}/{n_total} total ({total_label_ratio:.0%})")

    if display_label_ratio >= 0.30:
        # User labeled ≥30% of displayed grid → they've seen enough variety.
        # Skip expansion — use ONLY labeled samples + k-NN classification.
        expanded_target = set(target_crop_indices)
        print(f"  Using {len(expanded_target)} labeled targets directly "
              f"(display ratio {display_label_ratio:.0%} ≥ 30%, no expansion)")
    else:
        # Few labels — expand conservatively
        intra_target_dists = []
        for i in range(len(labeled_list)):
            for j in range(i + 1, len(labeled_list)):
                intra_target_dists.append(
                    np.linalg.norm(X_scaled[labeled_list[i]] - X_scaled[labeled_list[j]])
                )
        if intra_target_dists:
            # Use 25th percentile × 0.5 (very conservative)
            # Old: median × 1.0 → way too aggressive for diverse jerseys
            expand_radius = np.percentile(intra_target_dists, 25) * 0.5
        else:
            expand_radius = np.percentile(min_dist_to_target, 15)

        expanded_target = set(target_crop_indices)
        max_expanded = int(n_total * 0.6)  # Never expand beyond 60% of total

        for i in range(n_total):
            if len(expanded_target) >= max_expanded:
                break
            if i not in target_crop_indices and min_dist_to_target[i] <= expand_radius:
                expanded_target.add(i)

        print(f"  Expanded {len(target_crop_indices)} labeled → "
              f"{len(expanded_target)} target samples "
              f"(radius={expand_radius:.2f}, cap={max_expanded})")

    non_target_all = [(i, min_dist_to_target[i]) for i in range(n_total)
                      if i not in expanded_target]
    non_target_all.sort(key=lambda x: x[1], reverse=True)

    if len(non_target_all) >= 3:
        n_opp = max(3, int(len(non_target_all) * 0.7))
        opponent_indices = [i for i, _ in non_target_all[:n_opp]]
    else:
        opponent_indices = [i for i, _ in non_target_all]

    if len(opponent_indices) == 0:
        all_by_dist = sorted(range(n_total),
                             key=lambda i: min_dist_to_target[i], reverse=True)
        n_force = max(3, n_total // 5)
        opponent_indices = [i for i in all_by_dist
                            if i not in target_crop_indices][:n_force]
        print(f"  ⚠️ Few opponent samples — forced {len(opponent_indices)} "
              f"from farthest non-labeled crops")

    target_refs = X_scaled[list(expanded_target)]
    opponent_refs = X_scaled[opponent_indices] if opponent_indices else None

    target_centroid = target_refs.mean(axis=0)
    opponent_centroid = (opponent_refs.mean(axis=0) if opponent_refs is not None
                         else target_centroid)

    labels = np.zeros(n_total, dtype=int)
    if opponent_refs is not None and len(opponent_refs) > 0:
        for i in range(n_total):
            d_t = np.min(np.linalg.norm(target_refs - X_scaled[i], axis=1))
            d_o = np.min(np.linalg.norm(opponent_refs - X_scaled[i], axis=1))
            labels[i] = 0 if d_t <= d_o else 1

    n_target = (labels == 0).sum()
    n_opponent = (labels == 1).sum()

    _show_verification(crops, labels, target_crop_indices)

    print(f"\n✅ Calibration built from {len(target_crop_indices)} labeled → "
          f"{len(expanded_target)} expanded target samples")
    print(f"   Target team:  {n_target} crops ({n_target/n_total*100:.0f}%)")
    print(f"   Opponent:     {n_opponent} crops ({n_opponent/n_total*100:.0f}%)")

    return {
        "target_refs": target_refs,
        "opponent_refs": opponent_refs,
        "target_centroid": target_centroid,
        "opponent_centroid": opponent_centroid,
        "scaler": scaler,
        "target_cluster": 0,
        "n_clusters": 2,
        "cluster_sizes": {0: int(n_target), 1: int(n_opponent)},
        "n_crops_total": n_total,
        "n_labeled": len(target_crop_indices),
        "n_expanded": len(expanded_target),
    }


def _show_verification(crops, labels, labeled_indices):
    """Show a verification grid: target vs opponent rows."""
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


# ═══════════════════════════════════════════════════════════════════════
# Runtime classification (used by pipeline.py)
# ═══════════════════════════════════════════════════════════════════════

def classify_person(torso_crop, calibration, overlay_crop_mask=None):
    """
    Classify a person as target team, opponent, or ambiguous.
    Uses k-NN against reference sets.

    Returns: (role, confidence)
    """
    if calibration is None:
        return "ambiguous", 0.0

    features = extract_torso_features(torso_crop, overlay_crop_mask)
    if features is None:
        return "ambiguous", 0.0

    X = calibration["scaler"].transform([features])[0]

    target_refs = calibration.get("target_refs")
    opponent_refs = calibration.get("opponent_refs")
    if target_refs is not None and opponent_refs is not None and len(opponent_refs) > 0:
        d_target = float(np.min(np.linalg.norm(target_refs - X, axis=1)))
        d_opponent = float(np.min(np.linalg.norm(opponent_refs - X, axis=1)))
    else:
        d_target = np.linalg.norm(X - calibration["target_centroid"])
        d_opponent = np.linalg.norm(X - calibration["opponent_centroid"])

    total = d_target + d_opponent + 1e-9
    if d_target < d_opponent:
        confidence = 1.0 - (d_target / total)
        role = "target"
    else:
        confidence = 1.0 - (d_opponent / total)
        role = "opponent"

    if confidence < 0.53:
        return "ambiguous", round(confidence, 3)

    return role, round(confidence, 3)
