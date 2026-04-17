"""
Bradford Bulls v2 — Temporal Annotation Propagation (Phase 1b).

Given manually annotated frames (YOLO format), this module:
  1. Reads each annotated frame from the video
  2. Extracts ±N neighbor frames
  3. Tracks each annotated bbox via template matching
  4. Saves neighbor frames + adjusted annotations

Result: 400 manual annotations → ~2,000 total annotated frames
with REAL motion blur (not synthetic) — better than any augmentation.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

from . import config
from .helpers import compute_torso_sharpness, assign_sharpness_tier, detect_persons


def parse_yolo_annotations(label_path, img_w, img_h):
    """
    Parse a YOLO format annotation file.

    Returns list of dicts: [{"class_id": int, "cx": float, "cy": float, "bw": float, "bh": float}]
    All values are NORMALIZED (0-1).
    """
    annotations = []
    if not os.path.exists(label_path):
        return annotations

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                annotations.append({
                    "class_id": int(parts[0]),
                    "cx": float(parts[1]),
                    "cy": float(parts[2]),
                    "bw": float(parts[3]),
                    "bh": float(parts[4]),
                })
    return annotations


def write_yolo_annotations(label_path, annotations):
    """Write annotations in YOLO format."""
    with open(label_path, 'w') as f:
        for ann in annotations:
            f.write(f"{ann['class_id']} {ann['cx']:.6f} {ann['cy']:.6f} "
                    f"{ann['bw']:.6f} {ann['bh']:.6f}\n")


def track_bbox_to_neighbor(center_gray, neighbor_gray, bbox_norm,
                           offset, match_threshold=None):
    """
    Track a single bbox from center frame to neighbor frame using template matching.

    Args:
        center_gray: Grayscale center frame
        neighbor_gray: Grayscale neighbor frame
        bbox_norm: Dict with normalized cx, cy, bw, bh
        offset: Frame offset (e.g., -2, -1, +1, +2)
        match_threshold: Minimum template match confidence

    Returns:
        adjusted_bbox_norm or None if tracking failed
    """
    if match_threshold is None:
        match_threshold = config.PROPAGATION_MATCH_THRESHOLD

    h, w = center_gray.shape[:2]

    # Convert normalized → pixel coords
    cx_px = int(bbox_norm["cx"] * w)
    cy_px = int(bbox_norm["cy"] * h)
    bw_px = int(bbox_norm["bw"] * w)
    bh_px = int(bbox_norm["bh"] * h)

    # Ensure minimum template size
    if bw_px < 10 or bh_px < 10:
        return None

    # Template from center frame
    x1 = max(0, cx_px - bw_px // 2)
    y1 = max(0, cy_px - bh_px // 2)
    x2 = min(w, x1 + bw_px)
    y2 = min(h, y1 + bh_px)

    template = center_gray[y1:y2, x1:x2]
    if template.size == 0 or template.shape[0] < 10 or template.shape[1] < 10:
        return None

    # Expand search region based on offset magnitude
    # Players move ~10-20px per frame at 30fps
    expand = abs(offset) * 20
    sx1 = max(0, x1 - expand)
    sy1 = max(0, y1 - expand)
    sx2 = min(w, x2 + expand)
    sy2 = min(h, y2 + expand)

    search = neighbor_gray[sy1:sy2, sx1:sx2]
    if search.shape[0] < template.shape[0] or search.shape[1] < template.shape[1]:
        return None

    # Template matching
    result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val < match_threshold:
        return None

    # New bbox position in original frame coordinates
    new_x1 = sx1 + max_loc[0]
    new_y1 = sy1 + max_loc[1]

    # Convert back to normalized YOLO format
    new_cx = (new_x1 + bw_px / 2) / w
    new_cy = (new_y1 + bh_px / 2) / h

    # Bounds check
    if new_cx < 0 or new_cx > 1 or new_cy < 0 or new_cy > 1:
        return None

    return {
        "class_id": bbox_norm["class_id"],
        "cx": new_cx,
        "cy": new_cy,
        "bw": bbox_norm["bw"],
        "bh": bbox_norm["bh"],
        "match_confidence": max_val,
    }


def propagate_frame_annotations(video_path, frame_num, annotations,
                                 radius=None, match_threshold=None):
    """
    Propagate annotations from one frame to its temporal neighbors.

    Args:
        video_path: Path to video file
        frame_num: Frame number of the annotated center frame
        annotations: List of YOLO annotations (normalized)
        radius: ±N frames to propagate
        match_threshold: Minimum template match confidence

    Returns:
        List of (neighbor_frame_num, neighbor_frame_bgr, adjusted_annotations, avg_confidence)
    """
    if radius is None:
        radius = config.PROPAGATION_RADIUS
    if match_threshold is None:
        match_threshold = config.PROPAGATION_MATCH_THRESHOLD
    min_track_ratio = config.PROPAGATION_MIN_TRACK_RATIO

    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Read center frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, center_frame = cap.read()
    if not ret:
        cap.release()
        return []

    center_gray = cv2.cvtColor(center_frame, cv2.COLOR_BGR2GRAY)
    results = []

    for offset in range(-radius, radius + 1):
        if offset == 0:
            continue

        nb_fn = frame_num + offset
        if nb_fn < 0 or nb_fn >= total:
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, nb_fn)
        ret, nb_frame = cap.read()
        if not ret:
            continue

        nb_gray = cv2.cvtColor(nb_frame, cv2.COLOR_BGR2GRAY)

        # Track each annotation
        adjusted = []
        confidences = []
        for ann in annotations:
            tracked = track_bbox_to_neighbor(
                center_gray, nb_gray, ann, offset, match_threshold
            )
            if tracked is not None:
                adjusted.append(tracked)
                confidences.append(tracked["match_confidence"])

        # Check minimum track ratio
        if len(annotations) > 0 and len(adjusted) / len(annotations) >= min_track_ratio:
            avg_conf = np.mean(confidences) if confidences else 0.0
            results.append((nb_fn, nb_frame, adjusted, avg_conf))

    cap.release()
    return results


def run_propagation(video_path, frames_dir, labels_dir, output_dir,
                    yolo_model=None, device="cpu", radius=None):
    """
    Run full annotation propagation for all annotated frames.

    Expected directory structure:
        frames_dir/   ← annotated frame images
        labels_dir/   ← YOLO format .txt files (same names as images)

    Output:
        output_dir/images/   ← original + propagated frames
        output_dir/labels/   ← original + propagated annotations

    Args:
        video_path: Path to the video file
        frames_dir: Directory containing annotated frame images
        labels_dir: Directory containing YOLO label files
        output_dir: Output directory for propagated dataset
        yolo_model: Optional YOLO model for torso sharpness computation
        device: Device for YOLO inference
        radius: ±N frames to propagate
    """
    if radius is None:
        radius = config.PROPAGATION_RADIUS

    frames_dir = Path(frames_dir)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)
    output_images = output_dir / "images"
    output_labels = output_dir / "labels"
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    # Find all annotated frames
    image_files = sorted(
        [f for f in frames_dir.iterdir()
         if f.suffix.lower() in ('.jpg', '.jpeg', '.png')]
    )

    if not image_files:
        print("ERROR: No image files found in frames_dir")
        return

    # Extract frame numbers from filenames
    # Expected format: {match_id}_{frame_num:06d}_{timestamp}.jpg
    frame_map = {}
    for img_path in image_files:
        label_path = labels_dir / (img_path.stem + ".txt")
        if label_path.exists():
            # Parse frame number from filename
            parts = img_path.stem.split("_")
            try:
                frame_num = int(parts[1])
            except (IndexError, ValueError):
                # Try to find 6-digit number in filename
                for part in parts:
                    if part.isdigit() and len(part) == 6:
                        frame_num = int(part)
                        break
                else:
                    print(f"  WARNING: Cannot parse frame number from {img_path.name}, skipping")
                    continue

            frame_map[img_path] = {
                "frame_num": frame_num,
                "label_path": label_path,
            }

    print(f"\nFound {len(frame_map)} annotated frames with labels")
    if not frame_map:
        print("ERROR: No matching label files found")
        return

    # Get video info for annotation normalization
    cap = cv2.VideoCapture(str(video_path))
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    stats = {
        "original_frames": len(frame_map),
        "propagated_frames": 0,
        "total_frames": 0,
        "propagation_failures": 0,
        "tier_gold": 0,
        "tier_silver": 0,
        "tier_bronze": 0,
    }

    for img_path, info in tqdm(frame_map.items(), desc="Propagating annotations"):
        frame_num = info["frame_num"]
        label_path = info["label_path"]

        # Parse annotations
        annotations = parse_yolo_annotations(str(label_path), img_w, img_h)
        if not annotations:
            continue

        # Copy original frame + labels to output
        original_frame = cv2.imread(str(img_path))
        if original_frame is None:
            continue

        # Save original
        out_img_name = f"orig_{img_path.name}"
        cv2.imwrite(str(output_images / out_img_name), original_frame,
                     [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
        write_yolo_annotations(
            str(output_labels / (Path(out_img_name).stem + ".txt")),
            annotations
        )
        stats["total_frames"] += 1
        stats["tier_gold"] += 1  # Original frames are gold

        # Propagate to neighbors
        neighbors = propagate_frame_annotations(
            video_path, frame_num, annotations, radius=radius
        )

        for nb_fn, nb_frame, nb_annotations, avg_conf in neighbors:
            # Compute torso sharpness for the neighbor
            if yolo_model is not None:
                from .helpers import detect_persons
                dets = detect_persons(yolo_model, nb_frame, device)
                torso_sharp = compute_torso_sharpness(nb_frame, dets)
            else:
                # Fallback: simple Laplacian variance
                gray = cv2.cvtColor(nb_frame, cv2.COLOR_BGR2GRAY)
                torso_sharp = min(cv2.Laplacian(gray, cv2.CV_64F).var() / 300, 1.0)

            tier = assign_sharpness_tier(torso_sharp)
            if tier == "reject":
                stats["propagation_failures"] += 1
                continue

            stats[f"tier_{tier}"] += 1

            # Clean annotations (remove match_confidence field)
            clean_anns = [{
                "class_id": a["class_id"],
                "cx": a["cx"],
                "cy": a["cy"],
                "bw": a["bw"],
                "bh": a["bh"],
            } for a in nb_annotations]

            # Save neighbor
            nb_img_name = f"prop_{img_path.stem}_off{nb_fn - frame_num:+d}.jpg"
            cv2.imwrite(str(output_images / nb_img_name), nb_frame,
                         [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
            write_yolo_annotations(
                str(output_labels / (Path(nb_img_name).stem + ".txt")),
                clean_anns
            )
            stats["propagated_frames"] += 1
            stats["total_frames"] += 1

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"  ANNOTATION PROPAGATION RESULTS")
    print(f"{'=' * 60}")
    print(f"  Original annotated frames: {stats['original_frames']}")
    print(f"  Propagated frames:         {stats['propagated_frames']}")
    print(f"  Propagation failures:      {stats['propagation_failures']}")
    print(f"  Total output frames:       {stats['total_frames']}")
    print(f"  Dataset amplification:     {stats['total_frames']/max(stats['original_frames'],1):.1f}x")
    print(f"\n  Quality tier breakdown:")
    print(f"    🥇 Gold:   {stats['tier_gold']:4d} (original, sharp)")
    print(f"    🥈 Silver: {stats['tier_silver']:4d} (propagated, good)")
    print(f"    🥉 Bronze: {stats['tier_bronze']:4d} (propagated, usable)")
    print(f"{'=' * 60}")
    print(f"\n  Output saved to: {output_dir}")
    print(f"  Ready for training: {output_images}")

    return stats
