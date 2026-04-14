"""
Prototype: Multi-frame fusion for deblurring sports video frames.

Takes a frame number from a video, extracts ±N neighboring frames,
aligns them using optical flow, and fuses them to produce a sharper result.

Usage:
    python prototype_fusion.py --video videos/M06_black_1080p.mp4 \
                               --frame 462 \
                               --window 5 \
                               --output output/fusion_test/
"""

import argparse
import cv2
import numpy as np
from pathlib import Path


def extract_window(video_path: str, center_frame: int, window: int = 5):
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

    print(f"Extracted {len(frames)} frames: [{start}..{end}], FPS={fps:.0f}")
    return frames, fps


def compute_laplacian_sharpness(frame):
    """Laplacian variance as sharpness metric."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def align_frame_ecc(ref_gray, target, target_gray):
    """Align target to reference using ECC (Enhanced Correlation Coefficient).
    More robust than simple optical flow for global motion."""
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4)
    try:
        _, warp_matrix = cv2.findTransformECC(
            ref_gray, target_gray, warp_matrix, cv2.MOTION_EUCLIDEAN, criteria
        )
        h, w = ref_gray.shape
        aligned = cv2.warpAffine(
            target, warp_matrix, (w, h),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        )
        return aligned, True
    except cv2.error:
        return target, False


def align_frame_optflow(ref_gray, target, target_gray):
    """Align target to reference using dense optical flow (Farneback).
    Handles local motion better than ECC."""
    flow = cv2.calcOpticalFlowFarneback(
        ref_gray, target_gray,
        None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    h, w = ref_gray.shape
    map_x = np.arange(w, dtype=np.float32)[None, :].repeat(h, axis=0) + flow[..., 0]
    map_y = np.arange(h, dtype=np.float32)[:, None].repeat(w, axis=1) + flow[..., 1]
    aligned = cv2.remap(target, map_x, map_y, cv2.INTER_LINEAR)
    return aligned, True


def fuse_median(aligned_frames):
    """Median fusion — robust to outliers and ghosting."""
    stack = np.stack(aligned_frames, axis=0)
    return np.median(stack, axis=0).astype(np.uint8)


def fuse_weighted_sharpness(aligned_frames, sharpness_scores):
    """Weighted average fusion — sharper frames contribute more."""
    weights = np.array(sharpness_scores, dtype=np.float64)
    weights = weights / weights.sum()
    result = np.zeros_like(aligned_frames[0], dtype=np.float64)
    for frame, w in zip(aligned_frames, weights):
        result += frame.astype(np.float64) * w
    return result.astype(np.uint8)


def fuse_temporal_median_filter(aligned_frames, kernel=3):
    """Temporal median filter with sliding window for smoother result."""
    stack = np.stack(aligned_frames, axis=0)
    n = stack.shape[0]
    half_k = kernel // 2
    result_frames = []
    for i in range(n):
        start = max(0, i - half_k)
        end = min(n, i + half_k + 1)
        result_frames.append(np.median(stack[start:end], axis=0).astype(np.uint8))
    # Return the center frame's result
    return result_frames[n // 2]


def apply_unsharp_mask(frame, sigma=1.0, strength=1.5):
    """Gentle unsharp mask to enhance detail after fusion."""
    blurred = cv2.GaussianBlur(frame, (0, 0), sigma)
    sharpened = cv2.addWeighted(frame, 1.0 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def run_fusion(video_path, center_frame, window=5, output_dir="output/fusion_test"):
    """Run full fusion pipeline and save comparison images."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Extract frames
    frames, fps = extract_window(video_path, center_frame, window)
    if center_frame not in frames:
        print(f"ERROR: Center frame {center_frame} not found")
        return

    ref_frame = frames[center_frame]
    ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)

    # 2. Compute sharpness for all frames
    sharpness = {}
    for fn, f in frames.items():
        sharpness[fn] = compute_laplacian_sharpness(f)
        marker = " ← CENTER" if fn == center_frame else ""
        print(f"  Frame {fn:06d}: sharpness={sharpness[fn]:.1f}{marker}")

    # Find the sharpest frame in the window
    best_fn = max(sharpness, key=sharpness.get)
    print(f"  Sharpest in window: frame {best_fn} ({sharpness[best_fn]:.1f})")

    # 3. Align all frames to reference (center frame)
    aligned_ecc = []
    aligned_optflow = []
    sharpness_list = []
    aligned_count_ecc = 0

    for fn in sorted(frames.keys()):
        f = frames[fn]
        f_gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)

        if fn == center_frame:
            aligned_ecc.append(f)
            aligned_optflow.append(f)
        else:
            a_ecc, ok_ecc = align_frame_ecc(ref_gray, f, f_gray)
            aligned_ecc.append(a_ecc)
            if ok_ecc:
                aligned_count_ecc += 1

            a_of, _ = align_frame_optflow(ref_gray, f, f_gray)
            aligned_optflow.append(a_of)

        sharpness_list.append(sharpness[fn])

    print(f"  ECC alignment success: {aligned_count_ecc}/{len(frames)-1}")

    # 4. Apply different fusion strategies
    results = {}

    # Original (no fusion)
    results["0_original"] = ref_frame

    # Median fusion (ECC aligned)
    results["1_median_ecc"] = fuse_median(aligned_ecc)

    # Median fusion (Optical flow aligned)
    results["2_median_optflow"] = fuse_median(aligned_optflow)

    # Weighted fusion by sharpness (Optical flow)
    results["3_weighted_optflow"] = fuse_weighted_sharpness(
        aligned_optflow, sharpness_list
    )

    # Median + unsharp mask
    median_of = results["2_median_optflow"]
    results["4_median_optflow_sharp"] = apply_unsharp_mask(median_of, sigma=1.0, strength=1.5)

    # Sharpest single frame in window (baseline comparison)
    results["5_sharpest_single"] = frames[best_fn]

    # 5. Compute sharpness of each result
    print("\n--- Results ---")
    for name, img in results.items():
        s = compute_laplacian_sharpness(img)
        path = output_dir / f"f{center_frame:06d}_{name}.jpg"
        cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"  {name:30s}  sharpness={s:8.1f}  → {path.name}")

    # 6. Create side-by-side comparison (original vs best fusion)
    best_fusion = results["4_median_optflow_sharp"]
    h, w = ref_frame.shape[:2]

    # Crop center region (where players are) for detail comparison
    cy, cx = h // 2, w // 2
    crop_h, crop_w = h // 3, w // 3
    y1, y2 = cy - crop_h // 2, cy + crop_h // 2
    x1, x2 = cx - crop_w // 2, cx + crop_w // 2

    crop_orig = ref_frame[y1:y2, x1:x2]
    crop_fused = best_fusion[y1:y2, x1:x2]

    # Add labels
    label_orig = crop_orig.copy()
    label_fused = crop_fused.copy()
    cv2.putText(label_orig, "ORIGINAL", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(label_fused, "FUSED", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    comparison = np.hstack([label_orig, label_fused])
    comp_path = output_dir / f"f{center_frame:06d}_COMPARISON.jpg"
    cv2.imwrite(str(comp_path), comparison, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"\n  Comparison saved: {comp_path.name}")

    # Also save full-frame comparison
    full_orig = ref_frame.copy()
    full_fused = best_fusion.copy()
    cv2.putText(full_orig, "ORIGINAL", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    cv2.putText(full_fused, "FUSED (median+optflow+sharpen)", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    full_comp = np.hstack([full_orig, full_fused])
    full_path = output_dir / f"f{center_frame:06d}_FULL_COMPARISON.jpg"
    cv2.imwrite(str(full_path), full_comp, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"  Full comparison saved: {full_path.name}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-frame fusion prototype")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--frame", type=int, required=True,
                        help="Center frame number")
    parser.add_argument("--window", type=int, default=5,
                        help="Number of frames before/after center (default: 5)")
    parser.add_argument("--output", default="output/fusion_test",
                        help="Output directory")
    args = parser.parse_args()

    run_fusion(args.video, args.frame, args.window, args.output)
