"""
Prototype v2: Player-level multi-frame fusion.

Instead of aligning the entire frame, this version:
1. Detects players in the center frame (YOLO)
2. Tracks each player across ±N neighboring frames (template matching)
3. Aligns and fuses only the player crop region
4. Pastes enhanced player crops back onto the sharpest base frame

This avoids ghosting from independent player motion.

Usage:
    python prototype_fusion_v2.py --video videos/M06_black_1080p.mp4 \
                                  --frame 462 \
                                  --window 3 \
                                  --output output/fusion_v2/
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO


def extract_window(video_path: str, center_frame: int, window: int = 3):
    """Extract center_frame ± window frames from video."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start = max(0, center_frame - window)
    end = min(total - 1, center_frame + window)

    frames = {}
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    for fn in range(start, end + 1):
        ret, frame = cap.read()
        if ret:
            frames[fn] = frame
    cap.release()
    return frames, fps


def detect_players(model, frame, conf=0.45):
    """Detect persons and return bounding boxes."""
    results = model.predict(frame, classes=[0], conf=conf, verbose=False)
    detections = []
    if results[0].boxes is not None:
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            area = (x2 - x1) * (y2 - y1)
            detections.append({
                "bbox": (x1, y1, x2, y2),
                "area": area,
                "conf": float(box.conf[0]),
            })
    return detections


def expand_bbox(bbox, frame_shape, pad_ratio=0.15):
    """Expand bbox by pad_ratio for better context during tracking."""
    x1, y1, x2, y2 = bbox
    h, w = frame_shape[:2]
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * pad_ratio), int(bh * pad_ratio)
    return (
        max(0, x1 - px),
        max(0, y1 - py),
        min(w, x2 + px),
        min(h, y2 + py),
    )


def extract_torso_region(frame, bbox):
    """Extract upper body / torso region from player bbox (where logos are)."""
    x1, y1, x2, y2 = bbox
    bh = y2 - y1
    # Torso = top 30% to 65% of the player bbox
    torso_y1 = y1 + int(bh * 0.15)
    torso_y2 = y1 + int(bh * 0.65)
    return (x1, torso_y1, x2, torso_y2)


def track_player_across_frames(ref_crop_gray, frames_dict, ref_fn,
                               search_bbox, frame_shape):
    """Track a player crop across neighboring frames using template matching.

    Returns dict of {frame_num: (matched_x, matched_y)} offsets.
    """
    h_crop, w_crop = ref_crop_gray.shape[:2]
    sx1, sy1, sx2, sy2 = search_bbox
    locations = {ref_fn: (sx1, sy1)}  # reference is at search origin

    for fn, frame in sorted(frames_dict.items()):
        if fn == ref_fn:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Expand search area for neighboring frames (player may have moved)
        dist = abs(fn - ref_fn)
        expand = int(dist * 15)  # ~15px per frame movement budget
        search_x1 = max(0, sx1 - expand)
        search_y1 = max(0, sy1 - expand)
        search_x2 = min(frame_shape[1], sx2 + expand)
        search_y2 = min(frame_shape[0], sy2 + expand)

        search_region = gray[search_y1:search_y2, search_x1:search_x2]

        if (search_region.shape[0] < h_crop or
                search_region.shape[1] < w_crop):
            continue

        result = cv2.matchTemplate(search_region, ref_crop_gray,
                                   cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val > 0.4:  # reasonable match threshold
            match_x = search_x1 + max_loc[0]
            match_y = search_y1 + max_loc[1]
            locations[fn] = (match_x, match_y)

    return locations


def fuse_player_crops(frames_dict, locations, crop_size, method="median"):
    """Extract aligned crops from all frames and fuse them."""
    h, w = crop_size
    crops = []

    for fn in sorted(locations.keys()):
        mx, my = locations[fn]
        frame = frames_dict[fn]
        fh, fw = frame.shape[:2]

        # Bounds check
        if my + h > fh or mx + w > fw:
            continue

        crop = frame[my:my + h, mx:mx + w]
        if crop.shape[:2] == (h, w):
            crops.append(crop)

    if len(crops) < 2:
        return crops[0] if crops else None, len(crops)

    stack = np.stack(crops, axis=0)

    if method == "median":
        fused = np.median(stack, axis=0).astype(np.uint8)
    elif method == "mean":
        fused = np.mean(stack, axis=0).astype(np.uint8)
    else:
        # Sharpness-weighted
        sharpness = []
        for c in crops:
            gray = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
            sharpness.append(cv2.Laplacian(gray, cv2.CV_64F).var())
        weights = np.array(sharpness)
        weights = weights / weights.sum()
        fused = np.zeros_like(crops[0], dtype=np.float64)
        for c, wt in zip(crops, weights):
            fused += c.astype(np.float64) * wt
        fused = fused.astype(np.uint8)

    return fused, len(crops)


def gentle_sharpen(img, sigma=0.8, strength=0.5):
    """Very gentle sharpening to enhance detail without artifacts."""
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 1.0 + strength, blurred, -strength, 0)


def compute_sharpness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def run_fusion_v2(video_path, center_frame, window=3,
                  output_dir="output/fusion_v2"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Extract frames
    frames, fps = extract_window(video_path, center_frame, window)
    n_frames = len(frames)
    print(f"Extracted {n_frames} frames around frame {center_frame}")

    if center_frame not in frames:
        print(f"ERROR: Center frame {center_frame} not found")
        return

    ref_frame = frames[center_frame]
    frame_shape = ref_frame.shape

    # 2. Detect players in center frame
    print("Loading YOLO model...")
    model = YOLO("yolo11l.pt")
    detections = detect_players(model, ref_frame)
    print(f"Detected {len(detections)} players")

    if not detections:
        print("No players detected, skipping")
        return

    # Sort by area (largest first — closer to camera, better for logos)
    detections.sort(key=lambda d: d["area"], reverse=True)

    # 3. For each player: extract torso, track, fuse
    # Start with the sharpest single frame as base
    sharpness_scores = {fn: compute_sharpness(f) for fn, f in frames.items()}
    best_fn = max(sharpness_scores, key=sharpness_scores.get)
    base_frame = frames[best_fn].copy()
    result_frame = base_frame.copy()

    ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)

    player_results = []

    for i, det in enumerate(detections):
        bbox = det["bbox"]
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1

        # Get torso region (where logos are)
        torso_bbox = extract_torso_region(ref_frame, bbox)
        tx1, ty1, tx2, ty2 = torso_bbox
        torso_h, torso_w = ty2 - ty1, tx2 - tx1

        if torso_h < 20 or torso_w < 20:
            continue

        # Also prepare full-player crop for fusion
        exp_bbox = expand_bbox(bbox, frame_shape, pad_ratio=0.05)
        ex1, ey1, ex2, ey2 = exp_bbox
        crop_h, crop_w = ey2 - ey1, ex2 - ex1

        ref_crop_gray = ref_gray[ey1:ey2, ex1:ex2]

        # Track across frames
        locations = track_player_across_frames(
            ref_crop_gray, frames, center_frame,
            exp_bbox, frame_shape
        )

        # Fuse
        fused_crop, n_used = fuse_player_crops(
            frames, locations, (crop_h, crop_w), method="median"
        )

        if fused_crop is None:
            continue

        # Gentle sharpen on the fused crop
        fused_crop = gentle_sharpen(fused_crop, sigma=0.8, strength=0.5)

        # Compute improvement
        orig_crop = ref_frame[ey1:ey2, ex1:ex2]
        orig_sharp = compute_sharpness(orig_crop)
        fused_sharp = compute_sharpness(fused_crop)

        player_results.append({
            "idx": i,
            "bbox": bbox,
            "n_frames_used": n_used,
            "orig_sharpness": orig_sharp,
            "fused_sharpness": fused_sharp,
            "improvement": fused_sharp / max(orig_sharp, 1) - 1,
        })

        # Find where to paste on the base frame (track player in best frame too)
        if best_fn in locations:
            paste_x, paste_y = locations[best_fn]
        elif best_fn == center_frame:
            paste_x, paste_y = ex1, ey1
        else:
            # Re-track in best frame
            best_gray = cv2.cvtColor(frames[best_fn], cv2.COLOR_BGR2GRAY)
            search_region = best_gray[
                max(0, ey1 - 30):min(frame_shape[0], ey2 + 30),
                max(0, ex1 - 30):min(frame_shape[1], ex2 + 30)
            ]
            if (search_region.shape[0] >= crop_h and
                    search_region.shape[1] >= crop_w):
                res = cv2.matchTemplate(search_region, ref_crop_gray,
                                        cv2.TM_CCOEFF_NORMED)
                _, _, _, max_loc = cv2.minMaxLoc(res)
                paste_x = max(0, ex1 - 30) + max_loc[0]
                paste_y = max(0, ey1 - 30) + max_loc[1]
            else:
                paste_x, paste_y = ex1, ey1

        # Alpha blend at edges to avoid hard seams
        ph, pw = min(crop_h, result_frame.shape[0] - paste_y), \
                 min(crop_w, result_frame.shape[1] - paste_x)
        if ph > 0 and pw > 0:
            # Create feathered mask for smooth blending
            mask = np.ones((ph, pw), dtype=np.float32)
            feather = min(10, ph // 4, pw // 4)
            if feather > 1:
                for f in range(feather):
                    alpha = f / feather
                    mask[f, :] *= alpha
                    mask[ph - 1 - f, :] *= alpha
                    mask[:, f] *= alpha
                    mask[:, pw - 1 - f] *= alpha

            mask_3ch = mask[:, :, None]
            roi = result_frame[paste_y:paste_y + ph,
                               paste_x:paste_x + pw].astype(np.float64)
            fused_roi = fused_crop[:ph, :pw].astype(np.float64)
            blended = roi * (1 - mask_3ch) + fused_roi * mask_3ch
            result_frame[paste_y:paste_y + ph,
                         paste_x:paste_x + pw] = blended.astype(np.uint8)

        print(f"  Player {i}: bbox={bw}x{bh}, "
              f"fused {n_used} frames, "
              f"sharpness {orig_sharp:.0f} → {fused_sharp:.0f} "
              f"({player_results[-1]['improvement']:+.0%})")

        # Save individual player comparison
        # Use padded region on ref_frame for comparison
        orig_torso = ref_frame[ty1:ty2, tx1:tx2]
        fused_full = fused_crop  # full player
        # Extract torso from fused crop (relative coordinates)
        fused_torso_y1 = ty1 - ey1
        fused_torso_y2 = ty2 - ey1
        fused_torso_x1 = tx1 - ex1
        fused_torso_x2 = tx2 - ex1
        if (0 <= fused_torso_y1 < fused_crop.shape[0] and
                fused_torso_y2 <= fused_crop.shape[0] and
                0 <= fused_torso_x1 < fused_crop.shape[1] and
                fused_torso_x2 <= fused_crop.shape[1]):
            fused_torso = fused_crop[fused_torso_y1:fused_torso_y2,
                                     fused_torso_x1:fused_torso_x2]

            # Scale up for visibility
            scale = max(1, 200 // max(torso_h, 1))
            if scale > 1:
                orig_torso_big = cv2.resize(orig_torso, None,
                                            fx=scale, fy=scale,
                                            interpolation=cv2.INTER_NEAREST)
                fused_torso_big = cv2.resize(fused_torso, None,
                                             fx=scale, fy=scale,
                                             interpolation=cv2.INTER_NEAREST)
            else:
                orig_torso_big = orig_torso
                fused_torso_big = fused_torso

            # Label
            cv2.putText(orig_torso_big, "ORIG", (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(fused_torso_big, "FUSED", (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if orig_torso_big.shape[0] == fused_torso_big.shape[0]:
                torso_comp = np.hstack([orig_torso_big, fused_torso_big])
                p = output_dir / f"f{center_frame:06d}_player{i}_torso.jpg"
                cv2.imwrite(str(p), torso_comp,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])

    # 4. Save outputs
    # Original center frame
    p_orig = output_dir / f"f{center_frame:06d}_original.jpg"
    cv2.imwrite(str(p_orig), ref_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # Fused result
    p_fused = output_dir / f"f{center_frame:06d}_fused_v2.jpg"
    cv2.imwrite(str(p_fused), result_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # Base (sharpest single)
    p_base = output_dir / f"f{center_frame:06d}_sharpest_single.jpg"
    cv2.imwrite(str(p_base), base_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # Full comparison
    comp_orig = ref_frame.copy()
    comp_fused = result_frame.copy()
    cv2.putText(comp_orig, "ORIGINAL", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    cv2.putText(comp_fused, "PLAYER-LEVEL FUSION", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    full_comp = np.hstack([comp_orig, comp_fused])
    p_comp = output_dir / f"f{center_frame:06d}_COMPARISON.jpg"
    cv2.imwrite(str(p_comp), full_comp, [cv2.IMWRITE_JPEG_QUALITY, 95])

    print(f"\nSaved {len(player_results)} player fusions")
    print(f"Output: {output_dir}")

    overall_orig = compute_sharpness(ref_frame)
    overall_fused = compute_sharpness(result_frame)
    print(f"Overall sharpness: {overall_orig:.0f} → {overall_fused:.0f}")

    return player_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Player-level fusion v2")
    parser.add_argument("--video", required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--output", default="output/fusion_v2")
    args = parser.parse_args()

    run_fusion_v2(args.video, args.frame, args.window, args.output)
