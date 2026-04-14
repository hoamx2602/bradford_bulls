"""
Bradford Bulls — Frame Enhancement Pipeline v2

Why v2? RVRT (v1) produced poor results because it was trained on GoPro
camera-shake blur, but broadcast sports video degrades differently:
H.264 compression + low bitrate + mild motion blur. Different disease,
different medicine.

v2 Pipeline (matched to broadcast video degradation):
  1. Temporal Best-Frame — pick sharpest from ±N neighbors (zero hallucination)
  2. Classical Enhancement — denoise + multi-scale sharpen + CLAHE (zero hallucination)
  3. Real-ESRGAN x2 — super-resolution for real-world degradation (compression+blur+noise)

Strategies:
  classical  — steps 1-2, zero hallucination, original resolution
  realesrgan — steps 1-3, best quality, 2x resolution (720p→1440p)
  compare    — runs both, saves side-by-side comparison images

Usage:
    # Test (10 frames):
    python run_reconstruction.py --video video.mp4 --csv meta.csv --test

    # Classical only (zero hallucination):
    python run_reconstruction.py --video video.mp4 --csv meta.csv --strategy classical

    # Full pipeline (default, best quality):
    python run_reconstruction.py --video video.mp4 --csv meta.csv --strategy realesrgan

    # Compare strategies side-by-side:
    python run_reconstruction.py --video video.mp4 --csv meta.csv --strategy compare --test

    # Colab typical:
    python run_reconstruction.py \\
        --video /content/drive/MyDrive/Bradford_Bulls/videos/M07_white_720p.mp4 \\
        --csv /content/drive/MyDrive/Bradford_Bulls/metadata/M07_white_720p_v3_index.csv \\
        --weights-dir /content/weights \\
        --strategy realesrgan --test
"""

import argparse
import math
import time
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════════════
# QUALITY METRICS
# ═══════════════════════════════════════════════════════════════════════

def compute_sharpness(frame):
    """Laplacian variance — higher = sharper."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


# ═══════════════════════════════════════════════════════════════════════
# STEP 1: TEMPORAL BEST-FRAME SELECTION
# ═══════════════════════════════════════════════════════════════════════

def select_sharpest_neighbor(video_path, center_fn, radius=3, total_frames=None):
    """
    From frames [center-radius, center+radius], pick the sharpest one.
    At 30fps, ±3 frames = ±100ms — content barely changes but sharpness
    can vary significantly due to motion phase.
    """
    cap = cv2.VideoCapture(str(video_path))
    if total_frames is None:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start = max(0, center_fn - radius)
    end = min(total_frames - 1, center_fn + radius)

    best_frame = None
    best_fn = center_fn
    best_sharp = -1

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    for fn in range(start, end + 1):
        ret, frame = cap.read()
        if not ret:
            break
        sharp = compute_sharpness(frame)
        if sharp > best_sharp:
            best_sharp = sharp
            best_frame = frame.copy()
            best_fn = fn

    cap.release()
    return best_frame, best_fn, best_sharp


# ═══════════════════════════════════════════════════════════════════════
# STEP 2: CLASSICAL ENHANCEMENT (zero hallucination)
# ═══════════════════════════════════════════════════════════════════════

def apply_clahe(img, clip_limit=2.0):
    """Adaptive contrast — makes logos more readable in shadows/highlights."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def enhance_classical(img, sharpen_strength=1.5, clahe_clip=2.0):
    """
    Full classical pipeline: denoise → multi-scale sharpen → CLAHE.
    Every operation is a mathematical transform on real pixels — zero hallucination.
    """
    # 1. Mild bilateral denoise — preserve edges, reduce noise before sharpening
    result = cv2.bilateralFilter(img, d=5, sigmaColor=30, sigmaSpace=30)

    # 2. Multi-scale unsharp mask
    #    Fine (sigma=1): jersey numbers, small text on logos
    #    Medium (sigma=3): logo shapes, player outlines
    result_f = result.astype(np.float64)
    base = result.astype(np.float64)

    blur_fine = cv2.GaussianBlur(result, (0, 0), 1.0).astype(np.float64)
    result_f += sharpen_strength * 0.5 * (base - blur_fine)

    blur_med = cv2.GaussianBlur(result, (0, 0), 3.0).astype(np.float64)
    result_f += sharpen_strength * 0.3 * (base - blur_med)

    result = np.clip(result_f, 0, 255).astype(np.uint8)

    # 3. CLAHE — adaptive contrast for logo readability
    result = apply_clahe(result, clip_limit=clahe_clip)

    return result


# ═══════════════════════════════════════════════════════════════════════
# STEP 3: REAL-ESRGAN x2 (standalone — no basicsr dependency)
#
# Why Real-ESRGAN over RVRT:
#   - Trained on second-order degradation pipeline: blur + noise + JPEG
#     compression + resize — exactly what broadcast video looks like
#   - Single image (no temporal window complexity)
#   - Half-precision works (unlike RVRT's deformable attention)
#   - 64MB weights (vs RVRT's 155MB)
#   - 720p → 1440p: more pixels = more logo detail for detection
# ═══════════════════════════════════════════════════════════════════════

class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, num_feat, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """
    RRDB-Net backbone for Real-ESRGAN.
    x2 model: pixel_unshuffle at input → 23 RRDB blocks → 2x upsample layers.
    Architecture matches RealESRGAN_x2plus.pth checkpoint exactly.
    """
    def __init__(self, num_in_ch=3, num_out_ch=3, scale=2,
                 num_feat=64, num_block=23, num_grow_ch=32):
        super().__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(
            *[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        if self.scale == 2:
            x = F.pixel_unshuffle(x, 2)
        elif self.scale == 1:
            x = F.pixel_unshuffle(x, 4)
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.lrelu(self.conv_up1(
            F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(
            F.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


def load_realesrgan(weights_path, device, scale=2, half=True):
    """Load Real-ESRGAN x2plus model (standalone, no basicsr)."""
    model = RRDBNet(num_in_ch=3, num_out_ch=3, scale=scale,
                    num_feat=64, num_block=23, num_grow_ch=32)
    ckpt = torch.load(str(weights_path), map_location="cpu", weights_only=False)
    if "params_ema" in ckpt:
        state_dict = ckpt["params_ema"]
    elif "params" in ckpt:
        state_dict = ckpt["params"]
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()
    if half and device.type == "cuda":
        model = model.half()
    return model


def enhance_realesrgan(model, img_bgr, device, scale=2,
                       tile_size=512, tile_pad=10):
    """
    Real-ESRGAN inference with tiled processing for GPU memory efficiency.
    Input: BGR uint8 (H, W, 3)
    Output: BGR uint8 (H*scale, W*scale, 3)
    """
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    h, w = img.shape[:2]

    # Pad to even dimensions (pixel_unshuffle requires it)
    pad_h = h % 2
    pad_w = w % 2
    if pad_h or pad_w:
        img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')

    ph, pw = img.shape[:2]
    is_half = next(model.parameters()).dtype == torch.float16

    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
    if is_half:
        tensor = tensor.half()

    out_h, out_w = ph * scale, pw * scale

    # If fits in one tile, process directly
    if ph <= tile_size and pw <= tile_size:
        with torch.no_grad():
            output = model(tensor)
        output = output.float().squeeze(0).permute(1, 2, 0).cpu().numpy()
        output = np.clip(output * 255, 0, 255).astype(np.uint8)
        output = output[:h * scale, :w * scale]
        return cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

    # Tiled processing
    output = torch.zeros(1, 3, out_h, out_w)
    tiles_y = math.ceil(ph / tile_size)
    tiles_x = math.ceil(pw / tile_size)

    for ty in range(tiles_y):
        for tx in range(tiles_x):
            # Input tile coordinates
            in_y = ty * tile_size
            in_x = tx * tile_size
            in_y_end = min(in_y + tile_size, ph)
            in_x_end = min(in_x + tile_size, pw)

            # With padding for border continuity
            in_y_pad = max(in_y - tile_pad, 0)
            in_x_pad = max(in_x - tile_pad, 0)
            in_y_end_pad = min(in_y_end + tile_pad, ph)
            in_x_end_pad = min(in_x_end + tile_pad, pw)

            # Ensure even dimensions for pixel unshuffle
            tile_h = in_y_end_pad - in_y_pad
            tile_w = in_x_end_pad - in_x_pad
            if tile_h % 2 != 0:
                if in_y_end_pad < ph:
                    in_y_end_pad += 1
                else:
                    in_y_pad -= 1
            if tile_w % 2 != 0:
                if in_x_end_pad < pw:
                    in_x_end_pad += 1
                else:
                    in_x_pad -= 1

            tile = tensor[:, :, in_y_pad:in_y_end_pad, in_x_pad:in_x_end_pad]
            with torch.no_grad():
                out_tile = model(tile).float().cpu()

            # Coordinates in output space
            out_y = in_y * scale
            out_x = in_x * scale
            out_y_end = in_y_end * scale
            out_x_end = in_x_end * scale

            # Trim padding from output tile
            trim_top = (in_y - in_y_pad) * scale
            trim_left = (in_x - in_x_pad) * scale
            trim_h = (in_y_end - in_y) * scale
            trim_w = (in_x_end - in_x) * scale

            output[:, :, out_y:out_y_end, out_x:out_x_end] = \
                out_tile[:, :, trim_top:trim_top + trim_h,
                         trim_left:trim_left + trim_w]

            if device.type == "cuda":
                torch.cuda.empty_cache()

    output = output.squeeze(0).permute(1, 2, 0).numpy()
    output = np.clip(output * 255, 0, 255).astype(np.uint8)
    output = output[:h * scale, :w * scale]
    return cv2.cvtColor(output, cv2.COLOR_RGB2BGR)


# ═══════════════════════════════════════════════════════════════════════
# COMPARISON OUTPUT
# ═══════════════════════════════════════════════════════════════════════

def make_comparison(original, classical, realesrgan_img, target_h=720):
    """
    Create a 3-panel comparison image.
    All panels resized to same height for fair visual comparison.
    """
    oh, ow = original.shape[:2]
    aspect = ow / oh
    target_w = int(target_h * aspect)

    panels = []
    labels = ["ORIGINAL", "CLASSICAL", "REAL-ESRGAN x2"]
    sources = [original, classical, realesrgan_img]

    for img, label in zip(sources, labels):
        resized = cv2.resize(img, (target_w, target_h),
                             interpolation=cv2.INTER_LANCZOS4)
        cv2.putText(resized, label, (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        panels.append(resized)

    return np.hstack(panels)


def make_crop_comparison(original, enhanced, label="ENHANCED",
                         crop_ratio=0.4, target_h=600):
    """
    Side-by-side center crop comparison — shows detail difference clearly.
    """
    oh, ow = original.shape[:2]
    ch = int(oh * crop_ratio)
    cw = int(ow * crop_ratio)
    cy = (oh - ch) // 2
    cx = (ow - cw) // 2

    orig_crop = original[cy:cy + ch, cx:cx + cw].copy()

    # For enhanced (may be 2x resolution), crop same region
    eh, ew = enhanced.shape[:2]
    s = eh / oh
    ecy = int(cy * s)
    ecx = int(cx * s)
    ech = int(ch * s)
    ecw = int(cw * s)
    enh_crop = enhanced[ecy:ecy + ech, ecx:ecx + ecw].copy()

    # Resize both to same dimensions
    aspect = cw / ch
    tw = int(target_h * aspect)
    orig_resized = cv2.resize(orig_crop, (tw, target_h),
                              interpolation=cv2.INTER_LANCZOS4)
    enh_resized = cv2.resize(enh_crop, (tw, target_h),
                             interpolation=cv2.INTER_LANCZOS4)

    cv2.putText(orig_resized, "ORIGINAL", (8, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.putText(enh_resized, label, (8, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return np.hstack([orig_resized, enh_resized])


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Bradford Bulls — Frame Enhancement v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--csv", required=True, help="Metadata CSV from extraction")
    parser.add_argument("--output", default=None, help="Output directory")
    parser.add_argument("--strategy",
                        choices=["classical", "realesrgan", "compare"],
                        default="realesrgan",
                        help="Enhancement strategy (default: realesrgan)")
    parser.add_argument("--test", action="store_true",
                        help="Process first 10 frames only")
    parser.add_argument("--n-frames", type=int, default=None,
                        help="Process first N frames")
    parser.add_argument("--weights-dir", default="/content/weights",
                        help="Directory with model weights")
    parser.add_argument("--temporal-radius", type=int, default=3,
                        help="±N frames for best-frame selection (default: 3)")
    parser.add_argument("--no-temporal", action="store_true",
                        help="Skip temporal best-frame selection")
    parser.add_argument("--sharpen-strength", type=float, default=1.5,
                        help="Classical sharpening strength (default: 1.5)")
    parser.add_argument("--clahe-clip", type=float, default=2.0,
                        help="CLAHE clip limit (default: 2.0)")
    parser.add_argument("--tile-size", type=int, default=512,
                        help="Tile size for Real-ESRGAN (default: 512)")
    parser.add_argument("--quality", type=int, default=95,
                        help="JPEG save quality (default: 95)")
    args = parser.parse_args()

    video_path = Path(args.video)
    csv_path = Path(args.csv)
    match_id = video_path.stem

    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = csv_path.parent.parent / "frames_enhanced" / match_id
    output_dir.mkdir(parents=True, exist_ok=True)

    assert video_path.exists(), f"Video not found: {video_path}"
    assert csv_path.exists(), f"CSV not found: {csv_path}"

    # Load metadata
    df = pd.read_csv(csv_path)
    target_fns = df["frame_num"].tolist()

    if args.n_frames:
        target_fns = target_fns[:args.n_frames]
    elif args.test:
        target_fns = target_fns[:10]

    # Video info
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("Bradford Bulls — Frame Enhancement v2")
    print("=" * 60)
    print(f"  Video:     {video_path.name} ({W}x{H} @ {fps:.0f}fps)")
    print(f"  Metadata:  {len(df)} frames total")
    print(f"  Process:   {len(target_fns)} frames "
          f"{'(TEST)' if args.test else '(PRODUCTION)'}")
    print(f"  Strategy:  {args.strategy}")
    if not args.no_temporal:
        print(f"  Temporal:  ±{args.temporal_radius} frames")
    print(f"  Output:    {output_dir}")
    if device.type == "cuda":
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"  Device:    {torch.cuda.get_device_name(0)} ({gpu_mem:.1f} GB)")
    else:
        print("  Device:    CPU (will be slow)")
    if args.strategy in ("realesrgan", "compare"):
        out_res = f"{W * 2}x{H * 2}"
        print(f"  Output res: {out_res} (2x upscale)")
    print()

    # ── Load model ──
    esrgan_model = None
    if args.strategy in ("realesrgan", "compare"):
        esrgan_path = Path(args.weights_dir) / "RealESRGAN_x2plus.pth"
        assert esrgan_path.exists(), \
            f"Weights not found: {esrgan_path}\n" \
            f"Download: wget -O {esrgan_path} " \
            f"'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth'"
        print("Loading Real-ESRGAN x2plus...")
        t0 = time.time()
        esrgan_model = load_realesrgan(str(esrgan_path), device, scale=2,
                                       half=(device.type == "cuda"))
        print(f"  Loaded in {time.time() - t0:.1f}s "
              f"({'fp16' if device.type == 'cuda' else 'fp32'})")
        print()

    # ── Process frames ──
    results = []
    t_start = time.time()

    for fn in tqdm(target_fns, desc="Enhancing"):
        row = df[df["frame_num"] == fn].iloc[0]
        fname = row["filename"]

        # Step 1: Temporal best-frame selection
        if not args.no_temporal:
            frame, actual_fn, orig_sharp = select_sharpest_neighbor(
                video_path, fn, args.temporal_radius, total_frames)
        else:
            cap = cv2.VideoCapture(str(video_path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                print(f"  SKIP frame {fn}: could not read")
                continue
            actual_fn = fn
            orig_sharp = compute_sharpness(frame)

        if frame is None:
            print(f"  SKIP frame {fn}: could not read")
            continue

        try:
            # Step 2+3: Enhance based on strategy
            if args.strategy == "classical":
                enhanced = enhance_classical(
                    frame, args.sharpen_strength, args.clahe_clip)
                method = "classical"
                output_res = f"{W}x{H}"

            elif args.strategy == "realesrgan":
                enhanced = enhance_realesrgan(
                    esrgan_model, frame, device,
                    scale=2, tile_size=args.tile_size)
                # Light CLAHE on upscaled result for contrast
                enhanced = apply_clahe(enhanced, args.clahe_clip)
                method = "realesrgan_x2"
                output_res = f"{W * 2}x{H * 2}"

            elif args.strategy == "compare":
                classical = enhance_classical(
                    frame, args.sharpen_strength, args.clahe_clip)
                realesrgan_out = enhance_realesrgan(
                    esrgan_model, frame, device,
                    scale=2, tile_size=args.tile_size)
                realesrgan_out = apply_clahe(realesrgan_out, args.clahe_clip)

                # Save the best (realesrgan) as the main output
                enhanced = realesrgan_out
                method = "realesrgan_x2"
                output_res = f"{W * 2}x{H * 2}"

                # Save comparison images
                stem = Path(fname).stem

                # Full 3-panel comparison
                comp = make_comparison(frame, classical, realesrgan_out)
                cv2.imwrite(
                    str(output_dir / f"{stem}_comparison.jpg"), comp,
                    [cv2.IMWRITE_JPEG_QUALITY, args.quality])

                # Center crop comparison (shows detail better)
                crop_comp = make_crop_comparison(
                    frame, realesrgan_out, "REAL-ESRGAN x2")
                cv2.imwrite(
                    str(output_dir / f"{stem}_crop_compare.jpg"), crop_comp,
                    [cv2.IMWRITE_JPEG_QUALITY, args.quality])

                # Also save classical version separately
                cv2.imwrite(
                    str(output_dir / f"{stem}_classical.jpg"), classical,
                    [cv2.IMWRITE_JPEG_QUALITY, args.quality])

            # Compute enhanced sharpness (at original resolution for comparison)
            if enhanced.shape[0] != H or enhanced.shape[1] != W:
                enh_for_metric = cv2.resize(enhanced, (W, H),
                                            interpolation=cv2.INTER_LANCZOS4)
            else:
                enh_for_metric = enhanced
            enh_sharp = compute_sharpness(enh_for_metric)

            # Save enhanced frame
            cv2.imwrite(str(output_dir / fname), enhanced,
                        [cv2.IMWRITE_JPEG_QUALITY, args.quality])

            results.append({
                "frame_num": fn,
                "filename": fname,
                "actual_frame": actual_fn,
                "temporal_shift": actual_fn - fn,
                "method": method,
                "output_resolution": output_res,
                "orig_sharpness": round(orig_sharp, 1),
                "enhanced_sharpness": round(enh_sharp, 1),
                "improvement": round(enh_sharp / max(orig_sharp, 1), 2),
            })

            if device.type == "cuda":
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"  ERROR frame {fn}: {e}")
            import traceback
            traceback.print_exc()
            cv2.imwrite(str(output_dir / fname), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, args.quality])
            results.append({
                "frame_num": fn, "filename": fname,
                "actual_frame": fn, "temporal_shift": 0,
                "method": "original (error fallback)",
                "output_resolution": f"{W}x{H}",
                "orig_sharpness": round(orig_sharp, 1),
                "enhanced_sharpness": round(orig_sharp, 1),
                "improvement": 1.0,
            })

    elapsed = time.time() - t_start

    # ── Save results ──
    results_df = pd.DataFrame(results)
    index_csv = output_dir / f"{match_id}_enhancement_index.csv"
    results_df.to_csv(index_csv, index=False)

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    if len(results_df) > 0:
        shifted = results_df[results_df["temporal_shift"] != 0]
        print(f"  Temporal selection picked different frame: "
              f"{len(shifted)}/{len(results_df)}")
        if len(shifted) > 0:
            print(f"    Avg shift: {shifted['temporal_shift'].abs().mean():.1f} frames")

        print()
        for method in results_df["method"].unique():
            subset = results_df[results_df["method"] == method]
            print(f"  [{method}]")
            print(f"    Frames:      {len(subset)}")
            print(f"    Sharpness:   {subset['orig_sharpness'].mean():.0f} → "
                  f"{subset['enhanced_sharpness'].mean():.0f}")
            print(f"    Improvement: {subset['improvement'].mean():.2f}x "
                  f"(min {subset['improvement'].min():.2f}x, "
                  f"max {subset['improvement'].max():.2f}x)")

    print(f"\n  Time:    {elapsed:.1f}s "
          f"({elapsed / max(len(results), 1):.1f}s/frame)")
    print(f"  Frames:  {output_dir}/")
    print(f"  Index:   {index_csv}")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
