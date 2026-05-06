#!/usr/bin/env python3
"""
Auto-calibrate team kit HSV ranges from a video.

Inspired by v2-optimize's calibration approach but simplified:
  v2 = runtime k-NN classifier (300-dim HSV histogram + green mask + scaler)
  v3 = static HSV ranges generated once, written into match.meta.yaml

Why static ranges instead of k-NN at runtime?
  - Static ranges are debuggable (you can read them off the YAML)
  - Static ranges are tunable post-hoc (edit YAML, no retrain)
  - 90% as effective when calibration sample is diverse
  - No model state to ship between calibration and inference

Algorithm
---------
  1. Sample N frames evenly spaced through video
  2. YOLO11l detect persons; extract torso crop (top 10-40% of bbox, center 60%)
  3. Filter torsos: skip skin-heavy (head), grass-heavy (background), too-small
  4. Compute median HSV per torso
  5. K-Means (K=3) on torso median HSVs → 3 clusters
       Typically: {team A, team B, refs/staff/coaches}
  6. Display cluster representatives + dominant color swatch
  7. User picks --cluster-id N for target team
  8. Compute HSV range from that cluster's torso pixel distribution:
       H: median ± h_pad   (with wrap-around for red)
       S: p10 → 255
       V: p10 → 255
  9. Write match.meta.yaml with primary_colors + ignore_regions stubs

Usage
-----
  # Step 1: cluster + display
  python scripts/auto_calibrate_kit.py \\
      --video data/videos/M06_black_1080p.mp4 \\
      --output /tmp/calib

  # → opens a matplotlib window showing 3 cluster representatives
  # → also saves /tmp/calib/cluster_*.jpg + /tmp/calib/clusters.json

  # Step 2: pick a cluster and write meta.yaml
  python scripts/auto_calibrate_kit.py \\
      --video data/videos/M06_black_1080p.mp4 \\
      --output /tmp/calib \\
      --cluster-id 1 \\
      --kit-context special \\
      --write-meta data/videos/M06_black_1080p.meta.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# ---------------------------------------------------------------------------
# Default regions to IGNORE when sampling torsos.
# Format: [x1_norm, y1_norm, x2_norm, y2_norm]  (0–1 relative to frame size)
# These cover the most common broadcast overlay zones.
# Override at runtime with --ignore-regions.
# ---------------------------------------------------------------------------
DEFAULT_IGNORE_REGIONS = [
    [0.00, 0.85, 0.65, 1.00],   # bottom-left scoreboard strip
    [0.00, 0.00, 0.18, 0.12],   # top-left stream logo
    [0.82, 0.00, 1.00, 0.12],   # top-right channel/BullsTV logo
    [0.35, 0.00, 0.65, 0.10],   # top-centre match badge / clock
]

def _bbox_center_in_ignore(bbox, frame_shape, ignore_regions):
    """Return True if the centre of *bbox* falls inside any ignore region."""
    fh, fw = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0 / fw
    cy = (y1 + y2) / 2.0 / fh
    for rx1, ry1, rx2, ry2 in ignore_regions:
        if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
            return True
    return False


# Lazy import - only needed if running calibration
def _yolo_detect_persons(video_path: Path, n_frames: int, conf: float, device: str,
                         ignore_regions=None):
    from ultralytics import YOLO
    weights_path = ROOT / "weights" / "yolo11l.pt"
    if not weights_path.exists():
        sys.exit(f"weights not found: {weights_path}")
    model = YOLO(str(weights_path))

    if ignore_regions is None:
        ignore_regions = DEFAULT_IGNORE_REGIONS

    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        sys.exit(f"Cannot read video: {video_path}")
    idxs = np.linspace(int(total * 0.05), int(total * 0.95), n_frames, dtype=int)

    crops = []
    skipped_overlay = 0
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        results = model.predict(frame, classes=[0], conf=conf, device=device, verbose=False)
        if not results or results[0].boxes is None:
            continue
        for box in results[0].boxes:
            bbox = box.xyxy[0].cpu().numpy()
            # ── Skip bboxes whose centre is inside a scoreboard / overlay zone ──
            if _bbox_center_in_ignore(bbox, frame.shape, ignore_regions):
                skipped_overlay += 1
                continue
            crop = _extract_torso(frame, bbox)
            if crop is not None:
                crops.append(crop)
    cap.release()
    if skipped_overlay:
        print(f"  Skipped {skipped_overlay} detections inside overlay/scoreboard regions")
    return crops


def _extract_torso(frame, bbox):
    """Extract jersey torso (10-40% of person height, center 80% width)."""
    fh, fw = frame.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in bbox)
    bw, bh = x2 - x1, y2 - y1
    if bw < 50 or bh < 80:
        return None
    if bh / max(bw, 1) < 1.0:
        return None
    if (bw * bh) / (fh * fw) > 0.20:
        return None  # close-up, won't fit a torso

    ty1 = max(0, y1 + int(bh * 0.10))
    ty2 = min(fh, y1 + int(bh * 0.40))
    tx1 = max(0, x1 + int(bw * 0.15))
    tx2 = min(fw, x2 - int(bw * 0.15))
    if ty2 - ty1 < 20 or tx2 - tx1 < 20:
        return None

    torso = frame[ty1:ty2, tx1:tx2]
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)

    # Reject torsos that are mostly skin or grass
    skin = cv2.inRange(hsv, (0, 30, 60), (25, 180, 255)).mean() / 255
    grass = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255)).mean() / 255
    if skin > 0.5 or grass > 0.5:
        return None

    return torso


def _torso_to_feat(crop):
    """
    20-dim feature vector per torso crop:
      • 18-bin H histogram (only colorful pixels S>40), normalised → sum=1
        then ×2.0 so hue distribution dominates over brightness
      • median S /255   (overall, not just colorful pixels)
      • median V /255

    Using a histogram instead of a single median-H captures the *shape* of the
    colour distribution, which is essential when jerseys share a similar median
    hue but differ in how uniformly coloured they are (e.g. one team solid black
    vs mixed black+white stripes both have low S but very different histograms).
    """
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    colorful = hsv[..., 1] > 40          # pixels with meaningful saturation
    n_colorful = colorful.sum()

    if n_colorful > 80:
        H_vals = hsv[..., 0][colorful].astype(float)
        hist, _ = np.histogram(H_vals, bins=18, range=(0, 180))
        hist = hist.astype(float) / max(hist.sum(), 1)   # normalise
    else:
        # Nearly achromatic crop (dark/white jersey, or bad crop)
        # Give it a flat histogram so it clusters together as "achromatic"
        hist = np.ones(18, dtype=float) / 18.0

    med_s = float(np.median(hsv[..., 1])) / 255.0
    med_v = float(np.median(hsv[..., 2])) / 255.0

    return np.concatenate([hist * 2.0, [med_s, med_v]])


def _cluster_torsos(crops, k):
    """
    K-Means on 20-dim HSV histogram features.
    Returns:
      labels   – cluster index per crop
      centers  – (k, 3) array of representative (H, S, V) in OpenCV scale,
                 computed as per-cluster medians (for display / swatch)
      feats    – raw feature matrix (n, 20)
    """
    from sklearn.cluster import KMeans

    feats = np.array([_torso_to_feat(c) for c in crops])
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = km.fit_predict(feats)

    # Compute per-cluster median HSV (in OpenCV scale) for display
    centers_hsv = []
    for ci in range(k):
        members = [crops[i] for i in range(len(crops)) if labels[i] == ci]
        if members:
            all_medians = []
            for c in members:
                h = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
                all_medians.append([
                    float(np.median(h[..., 0])),
                    float(np.median(h[..., 1])),
                    float(np.median(h[..., 2])),
                ])
            arr = np.array(all_medians)
            centers_hsv.append(np.median(arr, axis=0).tolist())
        else:
            centers_hsv.append([0.0, 0.0, 0.0])

    return labels, np.array(centers_hsv), feats


def _compute_hsv_range(cluster_crops, h_pad=12):
    """Compute (h_low, h_high, s_low, v_low) range from a list of torso crops."""
    all_pixels = []
    for crop in cluster_crops:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        all_pixels.append(hsv.reshape(-1, 3))
    all_pixels = np.concatenate(all_pixels, axis=0)

    H = all_pixels[:, 0]
    S = all_pixels[:, 1]
    V = all_pixels[:, 2]

    h_med = int(np.median(H))
    s_low = max(0, int(np.percentile(S, 10)) - 20)
    v_low = max(0, int(np.percentile(V, 10)) - 20)

    return h_med, s_low, v_low, h_pad


def _hsv_range_to_yaml_entries(h_med, s_low, v_low, h_pad, name="target"):
    """Format HSV range into match.meta.yaml entries (handle red wrap)."""
    if h_med <= h_pad:
        entries = [
            (name, [0, h_med + h_pad], [s_low, 255], [v_low, 255]),
            (f"{name}_wrap", [180 - h_pad + h_med, 180], [s_low, 255], [v_low, 255]),
        ]
    elif h_med >= 180 - h_pad:
        entries = [
            (name, [h_med - h_pad, 180], [s_low, 255], [v_low, 255]),
            (f"{name}_wrap", [0, h_med - (180 - h_pad)], [s_low, 255], [v_low, 255]),
        ]
    else:
        entries = [(name, [h_med - h_pad, h_med + h_pad], [s_low, 255], [v_low, 255])]
    return entries


def _save_cluster_grid(crops, labels, centers, out_dir):
    """Save one composite image per cluster with up to 9 representatives."""
    out_dir.mkdir(parents=True, exist_ok=True)
    K = int(labels.max()) + 1
    summaries = []
    for k in range(K):
        members = [crops[i] for i in range(len(crops)) if labels[i] == k]
        if not members:
            continue
        # Resize all crops to 96x128 and tile up to 3x3
        thumbs = [cv2.resize(c, (96, 128)) for c in members[:9]]
        rows = []
        for r in range(3):
            chunk = thumbs[r*3:(r+1)*3]
            if not chunk:
                break
            while len(chunk) < 3:
                chunk.append(np.zeros((128, 96, 3), dtype=np.uint8))
            rows.append(np.hstack(chunk))
        if not rows:
            continue
        while len(rows) < 3:
            rows.append(np.zeros((128, 96 * 3, 3), dtype=np.uint8))
        grid = np.vstack(rows)
        # Add color swatch on top with cluster center HSV
        ch, cs, cv = (int(round(v)) for v in centers[k])
        swatch_hsv = np.full((30, grid.shape[1], 3), (ch, cs, cv), dtype=np.uint8)
        swatch_bgr = cv2.cvtColor(swatch_hsv, cv2.COLOR_HSV2BGR)
        composite = np.vstack([swatch_bgr, grid])
        # Label
        cv2.putText(composite, f"Cluster {k}  HSV=({ch},{cs},{cv})  n={len(members)}",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        out = out_dir / f"cluster_{k}.jpg"
        cv2.imwrite(str(out), composite)
        summaries.append({
            "cluster_id": k,
            "size": len(members),
            "center_hsv": [ch, cs, cv],
            "image": str(out),
        })

    (out_dir / "clusters.json").write_text(json.dumps(summaries, indent=2))
    return summaries


def _cluster_hue_spread(crops, labels, cluster_id):
    """Return std-dev of median-H across crops in a cluster (0-180 scale)."""
    members = [crops[i] for i in range(len(crops)) if labels[i] == cluster_id]
    if len(members) < 2:
        return 0.0
    meds = []
    for c in members:
        hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
        meds.append(float(np.median(hsv[..., 0])))
    return float(np.std(meds))


def cmd_calibrate(args):
    if not args.video.exists():
        sys.exit(f"Video not found: {args.video}")

    ignore_regions = getattr(args, "ignore_regions", None) or DEFAULT_IGNORE_REGIONS

    print(f"Sampling {args.n_frames} frames from {args.video.name}...")
    print(f"  Ignoring {len(ignore_regions)} overlay region(s): {ignore_regions}")
    crops = _yolo_detect_persons(args.video, args.n_frames, args.conf, args.device,
                                  ignore_regions=ignore_regions)
    print(f"  Got {len(crops)} valid torso crops")
    if len(crops) < args.k * 3:
        sys.exit(f"Too few crops ({len(crops)}) for K={args.k} clustering. "
                 f"Lower --conf or increase --n-frames.")

    print(f"Clustering torsos into K={args.k} clusters...")
    labels, centers, feats = _cluster_torsos(crops, args.k)

    sizes = [(labels == k).sum() for k in range(args.k)]
    print(f"  Cluster sizes: {sizes}")

    # ── Warn about potentially mixed clusters ──────────────────────────────
    print()
    for ci in range(args.k):
        spread = _cluster_hue_spread(crops, labels, ci)
        if spread > 28:
            print(f"  ⚠  Cluster {ci}: high hue spread (σ={spread:.1f}°) — "
                  f"likely mixed or noisy. Consider it a 'reject' cluster.")
        else:
            print(f"  ✓  Cluster {ci}: hue spread σ={spread:.1f}°  (looks uniform)")

    args.output.mkdir(parents=True, exist_ok=True)
    summaries = _save_cluster_grid(crops, labels, centers, args.output)
    print()
    print("Cluster summary (sorted by size):")
    for s in sorted(summaries, key=lambda x: -x["size"]):
        print(f"  cluster {s['cluster_id']}  n={s['size']:3d}  "
              f"center HSV={s['center_hsv']}  → {s['image']}")
    print()
    print(f"Inspect images in {args.output}/cluster_*.jpg, then re-run with --cluster-id N --write-meta PATH")
    return crops, labels, centers


def cmd_write_meta(args, crops=None, labels=None, centers=None):
    """Write match.meta.yaml from a chosen cluster_id."""
    if crops is None:
        # Re-run calibration to get cluster assignments
        crops, labels, centers = cmd_calibrate(args)

    if not (0 <= args.cluster_id < args.k):
        sys.exit(f"--cluster-id must be in [0, {args.k - 1}]")

    cluster_crops = [crops[i] for i in range(len(crops)) if labels[i] == args.cluster_id]
    if not cluster_crops:
        sys.exit(f"Cluster {args.cluster_id} is empty")
    print(f"Computing HSV range from {len(cluster_crops)} crops in cluster {args.cluster_id}...")

    h_med, s_low, v_low, h_pad = _compute_hsv_range(cluster_crops, h_pad=args.h_pad)
    entries = _hsv_range_to_yaml_entries(h_med, s_low, v_low, h_pad, name=args.color_name)
    print(f"  Median HSV: ({h_med}, ?, ?)  S>={s_low}  V>={v_low}  h_pad=±{h_pad}")
    print(f"  Generated {len(entries)} HSV range entries")

    # Build YAML
    lines = [
        f"# Auto-calibrated by scripts/auto_calibrate_kit.py",
        f"# Source: {args.video.name}",
        f"# Cluster {args.cluster_id} of K={args.k}, {len(cluster_crops)} torso samples",
        f"#",
        f"kit_context: {args.kit_context}",
    ]
    if args.opponent:
        lines.append(f'opponent: "{args.opponent}"')
    lines += [
        f"",
        f"target_team:",
        f"  primary_colors:",
    ]
    for name, h_range, s_range, v_range in entries:
        lines.append(
            f"    - {{name: {name}, h: [{h_range[0]}, {h_range[1]}], "
            f"s: [{s_range[0]}, {s_range[1]}], v: [{v_range[0]}, {v_range[1]}]}}"
        )
    lines.append(f"  min_team_score: {args.min_team_score}")
    lines += [
        f"",
        f"ignore_regions:",
        f"  - [0.00, 0.85, 0.50, 1.00]    # bottom-left scoreboard (verify with calibrate_meta.py region)",
        f"  - [0.85, 0.00, 1.00, 0.15]    # top-right BullsTV logo (verify with calibrate_meta.py region)",
    ]
    yaml_text = "\n".join(lines) + "\n"

    args.write_meta.parent.mkdir(parents=True, exist_ok=True)
    args.write_meta.write_text(yaml_text)
    print(f"\nWrote {args.write_meta}")
    print()
    print("=" * 60)
    print(yaml_text)
    print("=" * 60)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("/tmp/calib"))
    p.add_argument("--n-frames", type=int, default=40, help="Sample this many frames (default 40)")
    p.add_argument("--k", type=int, default=3, help="Number of color clusters (default 3)")
    p.add_argument("--conf", type=float, default=0.5, help="YOLO person confidence (default 0.5)")
    p.add_argument("--device", default="auto", help="auto | cuda | cpu")
    p.add_argument(
        "--ignore-regions", type=str, default=None,
        help=(
            "JSON list of [x1,y1,x2,y2] normalised regions to exclude from torso sampling. "
            "Overrides the built-in defaults (scoreboard + stream logo zones). "
            "Example: --ignore-regions '[[0,0.85,0.6,1],[0.8,0,1,0.12]]'"
        ),
    )

    p.add_argument("--cluster-id", type=int, default=None,
                   help="Pick this cluster as target team and write meta.yaml")
    p.add_argument("--write-meta", type=Path, default=None,
                   help="Output path for match.meta.yaml (required with --cluster-id)")
    p.add_argument("--kit-context", default="home",
                   choices=["home", "away", "special", "any"])
    p.add_argument("--opponent", default=None)
    p.add_argument("--color-name", default="target")
    p.add_argument("--h-pad", type=int, default=12, help="Hue half-width (default 12)")
    p.add_argument("--min-team-score", type=float, default=0.10)

    args = p.parse_args()

    # Parse --ignore-regions JSON
    if args.ignore_regions is not None:
        try:
            args.ignore_regions = json.loads(args.ignore_regions)
        except json.JSONDecodeError as e:
            sys.exit(f"--ignore-regions: invalid JSON — {e}")
    else:
        args.ignore_regions = DEFAULT_IGNORE_REGIONS

    # Resolve device
    if args.device == "auto":
        try:
            import torch
            args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            args.device = "cpu"

    if args.cluster_id is not None and args.write_meta is None:
        sys.exit("--write-meta is required with --cluster-id")

    if args.cluster_id is None:
        cmd_calibrate(args)
    else:
        cmd_write_meta(args)


if __name__ == "__main__":
    main()
