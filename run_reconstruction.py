"""
Bradford Bulls — Frame Reconstruction Pipeline

Reconstructs extracted frames into sharp, clear frames using RVRT
(video deblurring) with NAFNet as single-frame fallback.

Setup (run once on Colab):
    !pip install torch torchvision timm einops opencv-python-headless gdown
    !git clone https://github.com/JingyunLiang/RVRT.git /content/RVRT

    # Download weights:
    !mkdir -p /content/weights
    !wget -q -O /content/weights/005_RVRT_videodeblurring_GoPro_16frames.pth \
        'https://github.com/JingyunLiang/RVRT/releases/download/v0.0/005_RVRT_videodeblurring_GoPro_16frames.pth'
    !gdown 1S0PVRbyTakYY9a82kujgZLbMihfNBLfC -O /content/weights/NAFNet-GoPro-width64.pth

Usage:
    # Test mode (first 10 frames):
    python run_reconstruction.py --video /path/to/video.mp4 --csv /path/to/metadata.csv --test

    # Production (all frames):
    python run_reconstruction.py --video /path/to/video.mp4 --csv /path/to/metadata.csv

    # Custom output dir:
    python run_reconstruction.py --video video.mp4 --csv meta.csv --output /path/to/output

    # Google Colab typical usage:
    python run_reconstruction.py \
        --video /content/drive/MyDrive/Bradford_Bulls/videos/M06_black_1080p.mp4 \
        --csv /content/drive/MyDrive/Bradford_Bulls/metadata/M06_black_1080p_v3_index.csv \
        --output /content/drive/MyDrive/Bradford_Bulls/frames_reconstructed/M06_black_1080p \
        --rvrt-dir /content/RVRT \
        --weights-dir /content/weights \
        --test
"""

import argparse
import sys
import os
import time
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm


# ─── Scene Change Detection ─────────────────────────────────────────

def compute_histogram(frame, bins=64):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [bins, bins], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def detect_scene_boundaries(video_path, frame_nums, window, threshold=0.65):
    """Detect scene changes around each target frame.
    Returns dict: frame_num → (safe_start, safe_end)
    """
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    boundaries = {}

    for center_fn in tqdm(frame_nums, desc="Scene detection"):
        scan_start = max(0, center_fn - window - 2)
        scan_end = min(total - 1, center_fn + window + 2)

        cap.set(cv2.CAP_PROP_POS_FRAMES, scan_start)
        prev_hist = None
        scene_cuts = []

        for fn in range(scan_start, scan_end + 1):
            ret, frame = cap.read()
            if not ret:
                break
            hist = compute_histogram(frame)
            if prev_hist is not None:
                diff = cv2.compareHist(
                    prev_hist.reshape(-1, 1).astype(np.float32),
                    hist.reshape(-1, 1).astype(np.float32),
                    cv2.HISTCMP_BHATTACHARYYA,
                )
                if diff > threshold:
                    scene_cuts.append(fn)
            prev_hist = hist

        safe_start = max(0, center_fn - window)
        safe_end = min(total - 1, center_fn + window)

        for cut_fn in scene_cuts:
            if cut_fn <= center_fn and cut_fn > safe_start:
                safe_start = cut_fn
            elif cut_fn > center_fn and cut_fn < safe_end:
                safe_end = cut_fn - 1

        boundaries[center_fn] = (safe_start, safe_end)

    cap.release()
    return boundaries


# ─── RVRT Model ─────────────────────────────────────────────────────

def load_rvrt(weights_path, rvrt_dir, device):
    sys.path.insert(0, rvrt_dir)
    from models.network_rvrt import RVRT as RVRTNet

    model = RVRTNet(
        upscale=1,
        clip_size=2,
        img_size=[2, 64, 64],
        window_size=[2, 8, 8],
        num_blocks=[1, 2, 1],
        depths=[2, 2, 2],
        embed_dims=[192, 192, 192],
        num_heads=[6, 6, 6],
        inputconv_groups=[1, 3, 3, 3, 3, 3],
        deformable_groups=12,
        attention_heads=12,
        attention_window=[3, 3],
        cpu_cache_length=100,
    )

    checkpoint = torch.load(weights_path, map_location="cpu")
    state_dict = checkpoint.get("params", checkpoint.get("params_ema", checkpoint))
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()
    return model


# ─── NAFNet Model (fallback) ────────────────────────────────────────

class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + self.eps).sqrt()
        C = self.weight.shape[0]
        return self.weight.reshape(1, C, 1, 1) * y + self.bias.reshape(1, C, 1, 1)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, dw_expand=2, ffn_expand=2):
        super().__init__()
        dw_channel = c * dw_expand
        self.conv1 = nn.Conv2d(c, dw_channel, 1)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, 1, 1, groups=dw_channel)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, 1),
        )
        self.sg = SimpleGate()
        ffn_channel = ffn_expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1)
        self.sg2 = SimpleGate()
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)))
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)))

    def forward(self, inp):
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        y = inp + x * self.beta
        x = self.conv4(self.norm2(y))
        x = self.sg2(x)
        x = self.conv5(x)
        return y + x * self.gamma


class NAFNet(nn.Module):
    def __init__(self, img_channel=3, width=64, middle_blk_num=1,
                 enc_blk_nums=(1, 1, 1, 28), dec_blk_nums=(1, 1, 1, 1)):
        super().__init__()
        self.intro = nn.Conv2d(img_channel, width, 3, 1, 1)
        self.ending = nn.Conv2d(width, img_channel, 3, 1, 1)
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan *= 2
        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])
        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(
                nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)))
            chan //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
        self.padder_size = 2 ** len(enc_blk_nums)

    def forward(self, inp):
        B, C, H, W = inp.shape
        mod_h = (self.padder_size - H % self.padder_size) % self.padder_size
        mod_w = (self.padder_size - W % self.padder_size) % self.padder_size
        inp = F.pad(inp, (0, mod_w, 0, mod_h))
        x = self.intro(inp)
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)
        x = self.middle_blks(x)
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)
        x = self.ending(x)
        x = x + inp
        return x[:, :, :H, :W]


def load_nafnet(weights_path, device):
    model = NAFNet()
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("params", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()
    return model


# ─── Reconstruction ─────────────────────────────────────────────────

def extract_frame_window(video_path, center_fn, start_fn, end_fn):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_fn)
    frames = []
    center_idx = None
    for fn in range(start_fn, end_fn + 1):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        if fn == center_fn:
            center_idx = len(frames) - 1
    cap.release()
    return frames, center_idx


def pad_or_trim_sequence(frames, target_length=16, center_idx=None):
    n = len(frames)
    if n == target_length:
        return frames, center_idx
    if n > target_length:
        if center_idx is None:
            center_idx = n // 2
        half = target_length // 2
        start = max(0, center_idx - half)
        start = min(start, n - target_length)
        return frames[start:start + target_length], center_idx - start
    # Pad
    padded = list(frames)
    new_center = center_idx if center_idx is not None else n // 2
    while len(padded) < target_length:
        padded.append(padded[-1])
        if len(padded) < target_length:
            padded.insert(0, padded[0])
            new_center += 1
    return padded, new_center


def reconstruct_rvrt(model, frames, center_idx, device,
                     tile_h=256, tile_w=256, overlap_h=20, overlap_w=20):
    imgs = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            for f in frames]
    tensor = np.stack(imgs, axis=0)
    tensor = torch.from_numpy(tensor).permute(0, 3, 1, 2).unsqueeze(0).to(device)

    _, T, C, H, W = tensor.shape
    pad_h = (8 - H % 8) % 8
    pad_w = (8 - W % 8) % 8
    if pad_h > 0 or pad_w > 0:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")

    _, _, _, pH, pW = tensor.shape

    if pH <= tile_h * 1.5 and pW <= tile_w * 1.5:
        with torch.no_grad():
            output = model(tensor)
        result = output[0, center_idx].cpu().permute(1, 2, 0).numpy()
        result = np.clip(result * 255, 0, 255).astype(np.uint8)[:H, :W]
        return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

    output_all = torch.zeros(1, T, C, pH, pW, device="cpu")
    weight_map = torch.zeros(1, 1, 1, pH, pW, device="cpu")

    step_h = tile_h - overlap_h
    step_w = tile_w - overlap_w

    y_pos = list(range(0, max(pH - tile_h + 1, 1), step_h))
    if y_pos[-1] + tile_h < pH:
        y_pos.append(pH - tile_h)
    x_pos = list(range(0, max(pW - tile_w + 1, 1), step_w))
    if x_pos[-1] + tile_w < pW:
        x_pos.append(pW - tile_w)

    for y in y_pos:
        for x in x_pos:
            tile = tensor[:, :, :, y:y + tile_h, x:x + tile_w]
            with torch.no_grad():
                out_tile = model(tile).cpu()

            w = torch.ones(1, 1, 1, tile_h, tile_w)
            feather = min(overlap_h // 2, 10)
            if feather > 1:
                for fi in range(feather):
                    a = fi / feather
                    w[:, :, :, fi, :] *= a
                    w[:, :, :, tile_h - 1 - fi, :] *= a
                    w[:, :, :, :, fi] *= a
                    w[:, :, :, :, tile_w - 1 - fi] *= a

            output_all[:, :, :, y:y + tile_h, x:x + tile_w] += out_tile * w
            weight_map[:, :, :, y:y + tile_h, x:x + tile_w] += w
            torch.cuda.empty_cache()

    weight_map = torch.clamp(weight_map, min=1e-10)
    output_all = output_all / weight_map

    result = output_all[0, center_idx].permute(1, 2, 0).numpy()
    result = np.clip(result * 255, 0, 255).astype(np.uint8)[:H, :W]
    return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)


def reconstruct_nafnet(model, frame, device, tile_size=256, overlap=32):
    h, w = frame.shape[:2]
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    result = np.zeros_like(img, dtype=np.float64)
    wmap = np.zeros((h, w, 1), dtype=np.float64)

    step = tile_size - overlap
    y_pos = list(range(0, max(h - tile_size + 1, 1), step))
    if y_pos[-1] + tile_size < h:
        y_pos.append(h - tile_size)
    x_pos = list(range(0, max(w - tile_size + 1, 1), step))
    if x_pos[-1] + tile_size < w:
        x_pos.append(w - tile_size)

    for y in y_pos:
        for x in x_pos:
            tile = img[y:y + tile_size, x:x + tile_size]
            th, tw = tile.shape[:2]
            t = torch.from_numpy(tile).permute(2, 0, 1).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(t)
            out = out.squeeze(0).permute(1, 2, 0).cpu().numpy()[:th, :tw]

            mask = np.ones((th, tw, 1), dtype=np.float64)
            feather = min(overlap // 2, 16)
            if feather > 1:
                for fi in range(feather):
                    a = fi / feather
                    mask[fi, :] *= a
                    mask[th - 1 - fi, :] *= a
                    mask[:, fi] *= a
                    mask[:, tw - 1 - fi] *= a

            result[y:y + th, x:x + tw] += out.astype(np.float64) * mask
            wmap[y:y + th, x:x + tw] += mask

    wmap = np.maximum(wmap, 1e-10)
    result = np.clip(result / wmap * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)


def compute_sharpness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


# ─── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Bradford Bulls — Frame Reconstruction (RVRT + NAFNet)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test mode (10 frames):
  python run_reconstruction.py --video video.mp4 --csv metadata.csv --test

  # Production (all frames):
  python run_reconstruction.py --video video.mp4 --csv metadata.csv

  # Colab typical:
  python run_reconstruction.py \\
      --video /content/drive/MyDrive/Bradford_Bulls/videos/M06_black_1080p.mp4 \\
      --csv /content/drive/MyDrive/Bradford_Bulls/metadata/M06_black_1080p_v3_index.csv \\
      --output /content/drive/MyDrive/Bradford_Bulls/frames_reconstructed/M06_black_1080p
        """,
    )
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--csv", required=True,
                        help="Path to metadata CSV from 02_team_aware_extraction")
    parser.add_argument("--output", default=None,
                        help="Output directory (default: frames_reconstructed/<match_id>/ next to CSV)")
    parser.add_argument("--test", action="store_true",
                        help="Test mode: process only first 10 frames")
    parser.add_argument("--n-frames", type=int, default=None,
                        help="Process first N frames (overrides --test)")
    parser.add_argument("--rvrt-dir", default="/content/RVRT",
                        help="Path to cloned RVRT repo (default: /content/RVRT)")
    parser.add_argument("--weights-dir", default="/content/weights",
                        help="Directory containing model weights (default: /content/weights)")
    parser.add_argument("--window", type=int, default=8,
                        help="±N frames around center for RVRT (default: 8)")
    parser.add_argument("--min-window", type=int, default=7,
                        help="Minimum frames for RVRT, else NAFNet fallback (default: 7)")
    parser.add_argument("--scene-threshold", type=float, default=0.65,
                        help="Scene change detection threshold (default: 0.65)")
    parser.add_argument("--quality", type=int, default=95,
                        help="JPEG quality for saved frames (default: 95)")
    args = parser.parse_args()

    video_path = Path(args.video)
    csv_path = Path(args.csv)
    match_id = video_path.stem

    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = csv_path.parent.parent / "frames_reconstructed" / match_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Validate inputs ──
    assert video_path.exists(), f"Video not found: {video_path}"
    assert csv_path.exists(), f"CSV not found: {csv_path}"

    rvrt_weights = Path(args.weights_dir) / "005_RVRT_videodeblurring_GoPro_16frames.pth"
    nafnet_weights = Path(args.weights_dir) / "NAFNet-GoPro-width64.pth"
    assert rvrt_weights.exists(), f"RVRT weights not found: {rvrt_weights}"
    assert nafnet_weights.exists(), f"NAFNet weights not found: {nafnet_weights}"
    assert Path(args.rvrt_dir).exists(), f"RVRT repo not found: {args.rvrt_dir}"

    # ── Load metadata ──
    df = pd.read_csv(csv_path)
    target_fns = df["frame_num"].tolist()

    if args.n_frames:
        target_fns = target_fns[:args.n_frames]
    elif args.test:
        target_fns = target_fns[:10]

    # ── Video info ──
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    print("=" * 60)
    print("Bradford Bulls — Frame Reconstruction")
    print("=" * 60)
    print(f"  Video:    {video_path.name} ({W}x{H} @ {fps:.0f}fps)")
    print(f"  Metadata: {len(df)} frames total")
    print(f"  Process:  {len(target_fns)} frames {'(TEST)' if args.test else '(PRODUCTION)'}")
    print(f"  Output:   {output_dir}")
    print(f"  Window:   ±{args.window} frames, min {args.min_window} for RVRT")
    print(f"  Device:   ", end="")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"{torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
    else:
        print("CPU (WARNING: will be very slow)")
    print()

    # ── Phase 1: Scene change detection ──
    print("[Phase 1] Scene change detection...")
    t0 = time.time()
    boundaries = detect_scene_boundaries(
        video_path, target_fns, args.window, args.scene_threshold
    )
    trimmed = sum(1 for fn, (s, e) in boundaries.items()
                  if (fn - s) < args.window or (e - fn) < args.window)
    short = sum(1 for fn, (s, e) in boundaries.items()
                if (e - s + 1) < args.min_window)
    print(f"  Done in {time.time() - t0:.1f}s")
    print(f"  Trimmed by scene cut: {trimmed}")
    print(f"  NAFNet fallback (short window): {short}")
    print()

    # ── Phase 2: Load models ──
    print("[Phase 2] Loading models...")
    t0 = time.time()
    print("  Loading RVRT (video deblurring)...")
    rvrt_model = load_rvrt(str(rvrt_weights), args.rvrt_dir, device)
    print("  Loading NAFNet (single-frame fallback)...")
    nafnet_model = load_nafnet(str(nafnet_weights), device)
    print(f"  Done in {time.time() - t0:.1f}s")
    print()

    # ── Phase 3: Reconstruct ──
    print(f"[Phase 3] Reconstructing {len(target_fns)} frames...")
    RVRT_INPUT_LENGTH = 16
    results = []
    rvrt_count = 0
    nafnet_count = 0
    error_count = 0

    for fn in tqdm(target_fns, desc="Reconstructing"):
        safe_start, safe_end = boundaries[fn]
        window_size = safe_end - safe_start + 1

        frames_window, center_idx = extract_frame_window(
            video_path, fn, safe_start, safe_end
        )

        if not frames_window or center_idx is None:
            print(f"  SKIP frame {fn}: could not read")
            continue

        original = frames_window[center_idx].copy()
        orig_sharp = compute_sharpness(original)

        row = df[df["frame_num"] == fn].iloc[0]
        fname = row["filename"]

        try:
            if window_size >= args.min_window:
                padded, new_center = pad_or_trim_sequence(
                    frames_window, RVRT_INPUT_LENGTH, center_idx
                )
                reconstructed = reconstruct_rvrt(
                    rvrt_model, padded, new_center, device
                )
                method = "rvrt"
                rvrt_count += 1
            else:
                reconstructed = reconstruct_nafnet(nafnet_model, original, device)
                method = "nafnet"
                nafnet_count += 1

            recon_sharp = compute_sharpness(reconstructed)

            cv2.imwrite(
                str(output_dir / fname), reconstructed,
                [cv2.IMWRITE_JPEG_QUALITY, args.quality],
            )

            results.append({
                "frame_num": fn,
                "filename": fname,
                "method": method,
                "window_size": window_size,
                "orig_sharpness": round(orig_sharp, 1),
                "recon_sharpness": round(recon_sharp, 1),
                "improvement": round(recon_sharp / max(orig_sharp, 1), 2),
            })

            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  ERROR frame {fn}: {e}")
            cv2.imwrite(
                str(output_dir / fname), original,
                [cv2.IMWRITE_JPEG_QUALITY, args.quality],
            )
            results.append({
                "frame_num": fn,
                "filename": fname,
                "method": "original",
                "window_size": window_size,
                "orig_sharpness": round(orig_sharp, 1),
                "recon_sharpness": round(orig_sharp, 1),
                "improvement": 1.0,
            })
            error_count += 1

    print()

    # ── Phase 4: Save results ──
    results_df = pd.DataFrame(results)
    recon_csv = output_dir / f"{match_id}_reconstruction_index.csv"
    results_df.to_csv(recon_csv, index=False)

    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  RVRT reconstructed: {rvrt_count}")
    print(f"  NAFNet fallback:    {nafnet_count}")
    print(f"  Errors (saved original): {error_count}")
    print(f"  Total:              {len(results)}")
    print()

    if len(results_df) > 0:
        for method in results_df["method"].unique():
            subset = results_df[results_df["method"] == method]
            print(f"  [{method}] {len(subset)} frames | "
                  f"sharpness {subset['orig_sharpness'].mean():.0f} → "
                  f"{subset['recon_sharpness'].mean():.0f} | "
                  f"avg improvement {subset['improvement'].mean():.2f}x")
        print()

    print(f"  Frames saved:   {output_dir}/")
    print(f"  Metadata saved: {recon_csv}")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
