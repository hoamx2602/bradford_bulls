"""
Prototype v3: Temporal Focus Stacking

Mimics how the human brain perceives sharpness from video:
- The brain doesn't "average" frames — it accumulates the SHARPEST
  version of each local region across multiple frames.
- For each small patch in the image, we pick the version from the frame
  where THAT SPECIFIC patch is sharpest.

Technical approach:
1. Align neighboring frames to the reference (optical flow)
2. For each pixel, compute LOCAL sharpness across all aligned frames
3. Build per-frame weight maps based on local sharpness
4. Blend using Laplacian pyramid for seamless multi-scale fusion

This is "focus stacking in the time domain" — the same technique
photographers use for depth-of-field, applied to motion blur.

Usage:
    python prototype_fusion_v3.py --video videos/M06_black_1080p.mp4 \
                                  --frame 462 \
                                  --window 3 \
                                  --output output/fusion_v3/
"""

import argparse
import cv2
import numpy as np
from pathlib import Path


# ─── Frame extraction ────────────────────────────────────────────────

def extract_window(video_path: str, center_frame: int, window: int = 3):
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
    print(f"  Extracted {len(frames)} frames [{start}..{end}], FPS={fps:.0f}")
    return frames, fps


# ─── Alignment ───────────────────────────────────────────────────────

def align_to_reference(ref_frame, target_frame):
    """Align target to reference using dense optical flow.
    Returns the warped target frame."""
    ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
    tgt_gray = cv2.cvtColor(target_frame, cv2.COLOR_BGR2GRAY)

    # Dense optical flow (Farneback)
    flow = cv2.calcOpticalFlowFarneback(
        tgt_gray, ref_gray,  # flow FROM target TO ref
        None, 0.5, 5, 15, 3, 7, 1.5, 0
    )

    h, w = ref_gray.shape
    # Build remap coordinates
    coords_x = np.arange(w, dtype=np.float32)[None, :].repeat(h, axis=0)
    coords_y = np.arange(h, dtype=np.float32)[:, None].repeat(w, axis=1)
    map_x = coords_x + flow[..., 0]
    map_y = coords_y + flow[..., 1]

    aligned = cv2.remap(target_frame, map_x, map_y, cv2.INTER_LINEAR)
    return aligned


# ─── Local sharpness map ────────────────────────────────────────────

def compute_local_sharpness(frame, kernel_size=15):
    """Compute per-pixel local sharpness using Laplacian magnitude.

    For each pixel, sharpness = magnitude of Laplacian in a local window.
    High value = sharp edge or detail at that location.
    Low value = flat region or motion blur.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)

    # Laplacian captures edges/detail
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_abs = np.abs(lap)

    # Local average of Laplacian magnitude = local sharpness
    # Using box filter for speed (Gaussian would be smoother)
    local_sharp = cv2.blur(lap_abs, (kernel_size, kernel_size))

    return local_sharp


# ─── Laplacian pyramid blending ─────────────────────────────────────

def build_laplacian_pyramid(img, levels=5):
    """Build a Laplacian pyramid for multi-scale blending."""
    gaussian = [img.astype(np.float64)]
    for _ in range(levels):
        down = cv2.pyrDown(gaussian[-1])
        gaussian.append(down)

    laplacian = []
    for i in range(levels):
        up = cv2.pyrUp(gaussian[i + 1],
                       dstsize=(gaussian[i].shape[1], gaussian[i].shape[0]))
        lap = gaussian[i] - up
        laplacian.append(lap)
    laplacian.append(gaussian[-1])  # top of pyramid = lowest frequency

    return laplacian


def build_gaussian_pyramid(mask, levels=5):
    """Build a Gaussian pyramid for the weight mask."""
    pyramid = [mask.astype(np.float64)]
    for _ in range(levels):
        down = cv2.pyrDown(pyramid[-1])
        pyramid.append(down)
    return pyramid


def reconstruct_from_pyramid(pyramid):
    """Reconstruct image from Laplacian pyramid."""
    img = pyramid[-1]
    for i in range(len(pyramid) - 2, -1, -1):
        up = cv2.pyrUp(img, dstsize=(pyramid[i].shape[1], pyramid[i].shape[0]))
        img = up + pyramid[i]
    return np.clip(img, 0, 255).astype(np.uint8)


def pyramid_blend(frames, weight_maps, levels=5):
    """Blend multiple frames using Laplacian pyramid with weight maps.

    This gives seamless multi-scale blending — no visible seams between
    regions from different frames.
    """
    n = len(frames)

    # Build Laplacian pyramids for each frame
    frame_pyramids = [build_laplacian_pyramid(f, levels) for f in frames]
    # Build Gaussian pyramids for each weight map
    weight_pyramids = [build_gaussian_pyramid(w, levels) for w in weight_maps]

    # Blend at each pyramid level
    blended_pyramid = []
    for level in range(levels + 1):
        blended = np.zeros_like(frame_pyramids[0][level], dtype=np.float64)
        weight_sum = np.zeros_like(blended)

        for i in range(n):
            w = weight_pyramids[i][level]
            # Expand weight to 3 channels if blending color images
            if blended.ndim == 3 and w.ndim == 2:
                w = w[:, :, None]
            blended += frame_pyramids[i][level] * w
            weight_sum += w

        # Avoid division by zero
        weight_sum = np.maximum(weight_sum, 1e-10)
        blended = blended / weight_sum
        blended_pyramid.append(blended)

    return reconstruct_from_pyramid(blended_pyramid)


# ─── Simple weighted blend (fallback) ───────────────────────────────

def simple_sharpness_blend(frames, weight_maps):
    """Direct pixel-wise weighted blend using sharpness maps."""
    result = np.zeros_like(frames[0], dtype=np.float64)
    weight_sum = np.zeros(frames[0].shape[:2], dtype=np.float64)

    for frame, wmap in zip(frames, weight_maps):
        result += frame.astype(np.float64) * wmap[:, :, None]
        weight_sum += wmap

    weight_sum = np.maximum(weight_sum, 1e-10)
    result = result / weight_sum[:, :, None]
    return np.clip(result, 0, 255).astype(np.uint8)


# ─── Main pipeline ──────────────────────────────────────────────────

def temporal_focus_stack(video_path, center_frame, window=3,
                         output_dir="output/fusion_v3",
                         sharpness_kernel=15, pyramid_levels=5):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Temporal Focus Stacking — frame {center_frame}, window ±{window}")
    print(f"{'='*60}")

    # 1. Extract frames
    frames_dict, fps = extract_window(video_path, center_frame, window)
    if center_frame not in frames_dict:
        print("ERROR: center frame not found")
        return

    ref_frame = frames_dict[center_frame]
    sorted_fns = sorted(frames_dict.keys())

    # 2. Align all frames to the center frame
    print("  Aligning frames...")
    aligned = []
    for fn in sorted_fns:
        if fn == center_frame:
            aligned.append(frames_dict[fn])
        else:
            a = align_to_reference(ref_frame, frames_dict[fn])
            aligned.append(a)

    # 3. Compute local sharpness maps
    print("  Computing local sharpness maps...")
    sharpness_maps = []
    for i, frame in enumerate(aligned):
        smap = compute_local_sharpness(frame, kernel_size=sharpness_kernel)
        sharpness_maps.append(smap)
        fn = sorted_fns[i]
        marker = " ← CENTER" if fn == center_frame else ""
        print(f"    Frame {fn}: mean_sharpness={smap.mean():.1f}, "
              f"max={smap.max():.1f}{marker}")

    # 4. Build per-frame weight maps from local sharpness
    # For each pixel: weight = how sharp this frame is at this location
    # relative to all other frames
    print("  Building weight maps...")
    sharpness_stack = np.stack(sharpness_maps, axis=0)  # (N, H, W)

    # Softmax-style weighting: emphasize the sharpest frame for each pixel
    # Temperature controls selectivity: lower = more selective (picks sharpest)
    temperature = 0.3
    # Normalize sharpness to [0, 1] range for numerical stability
    s_min = sharpness_stack.min()
    s_max = sharpness_stack.max()
    if s_max > s_min:
        s_norm = (sharpness_stack - s_min) / (s_max - s_min)
    else:
        s_norm = np.ones_like(sharpness_stack) / len(aligned)

    # Power-based sharpness emphasis (simpler than softmax, very effective)
    # Higher power = more selective (strongly prefers sharpest frame per pixel)
    power = 6.0
    weights = np.power(s_norm + 1e-10, power)

    # Normalize so weights sum to 1 at each pixel
    weight_sum = weights.sum(axis=0, keepdims=True)
    weight_sum = np.maximum(weight_sum, 1e-10)
    weights = weights / weight_sum

    weight_maps = [weights[i] for i in range(len(aligned))]

    # Show which frames dominate
    dominant = np.argmax(weights, axis=0)
    for i, fn in enumerate(sorted_fns):
        pct = (dominant == i).mean() * 100
        marker = " ← CENTER" if fn == center_frame else ""
        print(f"    Frame {fn}: dominates {pct:.1f}% of pixels{marker}")

    # 5. Fuse using multiple methods for comparison
    print("  Fusing...")
    results = {}

    # Original center frame
    results["0_original"] = ref_frame

    # Method A: Simple pixel-wise weighted blend
    results["1_simple_blend"] = simple_sharpness_blend(aligned, weight_maps)

    # Method B: Laplacian pyramid blend (multi-scale, seamless)
    results["2_pyramid_blend"] = pyramid_blend(
        aligned, weight_maps, levels=pyramid_levels
    )

    # Method C: Hard selection (for each pixel, take the sharpest frame)
    # This is the most "brain-like" approach
    hard_result = np.zeros_like(ref_frame, dtype=np.float64)
    for i in range(len(aligned)):
        mask = (dominant == i)
        hard_result[mask] = aligned[i][mask]
    results["3_hard_select"] = hard_result.astype(np.uint8)

    # Method D: Pyramid blend + very gentle sharpen
    pyramid_result = results["2_pyramid_blend"]
    blurred = cv2.GaussianBlur(pyramid_result, (0, 0), 0.8)
    sharpened = cv2.addWeighted(pyramid_result, 1.3, blurred, -0.3, 0)
    results["4_pyramid_sharpen"] = np.clip(sharpened, 0, 255).astype(np.uint8)

    # 6. Compute global sharpness and save
    print("\n  Results:")
    for name, img in results.items():
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        path = output_dir / f"f{center_frame:06d}_{name}.jpg"
        cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"    {name:30s}  sharpness={sharpness:8.1f}")

    # 7. Save comparison crops (center region for detail)
    h, w = ref_frame.shape[:2]
    # Take several crop regions for comparison
    crops = {
        "center": (h // 3, h * 2 // 3, w // 3, w * 2 // 3),
        "left": (h // 4, h * 3 // 4, 0, w // 3),
        "right": (h // 4, h * 3 // 4, w * 2 // 3, w),
    }

    best_method = "2_pyramid_blend"
    for crop_name, (cy1, cy2, cx1, cx2) in crops.items():
        orig_crop = ref_frame[cy1:cy2, cx1:cx2]
        fused_crop = results[best_method][cy1:cy2, cx1:cx2]

        # Label
        oc = orig_crop.copy()
        fc = fused_crop.copy()
        cv2.putText(oc, "ORIGINAL", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(fc, "FOCUS STACKED", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        comp = np.hstack([oc, fc])
        p = output_dir / f"f{center_frame:06d}_crop_{crop_name}.jpg"
        cv2.imwrite(str(p), comp, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # 8. Save weight map visualization
    # Show which frame each pixel comes from as a color map
    viz = np.zeros((h, w, 3), dtype=np.uint8)
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (0, 255, 255), (255, 0, 255),
        (128, 255, 0), (0, 128, 255), (255, 128, 0),
        (128, 0, 255), (255, 0, 128),
    ]
    for i in range(len(aligned)):
        color = colors[i % len(colors)]
        mask = (dominant == i)
        for c in range(3):
            viz[mask, c] = color[c]
    # Overlay on original
    overlay = cv2.addWeighted(ref_frame, 0.5, viz, 0.5, 0)
    center_idx = sorted_fns.index(center_frame)
    cv2.putText(overlay,
                f"Color = which frame is sharpest. "
                f"Green = center frame {center_frame}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    p_viz = output_dir / f"f{center_frame:06d}_weight_viz.jpg"
    cv2.imwrite(str(p_viz), overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"\n  Weight visualization saved: {p_viz.name}")

    print(f"  All outputs in: {output_dir}/")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Temporal Focus Stacking for sports video deblurring")
    parser.add_argument("--video", required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--window", type=int, default=3,
                        help="±N frames around center (default: 3)")
    parser.add_argument("--output", default="output/fusion_v3")
    parser.add_argument("--kernel", type=int, default=15,
                        help="Local sharpness kernel size (default: 15)")
    args = parser.parse_args()

    temporal_focus_stack(args.video, args.frame, args.window, args.output,
                         sharpness_kernel=args.kernel)
