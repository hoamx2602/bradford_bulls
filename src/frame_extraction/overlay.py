"""
Static overlay detection using temporal variance analysis.

Detects fixed broadcast elements (scoreboard, watermark, channel logo)
by finding pixels that remain constant across many frames.
"""

import cv2
import numpy as np


def detect_static_overlays(video_path, n_samples=30):
    """
    Detect static overlay regions (scoreboard, watermark) via temporal variance.
    
    Pixels that barely change across N sampled frames = static overlay.
    
    Args:
        video_path: Path to video file
        n_samples: Number of frames to sample for variance computation
    
    Returns:
        overlay_mask: Binary mask (H, W), 1=clean pixel, 0=overlay pixel
        overlay_ratio: Fraction of frame that is overlay
    """
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # Sample frames evenly across video (skip first/last 5%)
    start = int(total * 0.05)
    end = int(total * 0.95)
    indices = np.linspace(start, end, n_samples, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Downscale for speed (variance computation doesn't need full res)
            small = cv2.resize(frame, (w // 2, h // 2))
            frames.append(small.astype(np.float32))
    cap.release()

    if len(frames) < 5:
        print("WARNING: Too few frames for overlay detection. Skipping.")
        return np.ones((h, w), dtype=np.uint8), 0.0

    # Compute per-pixel variance across all sampled frames
    stacked = np.stack(frames)               # (N, H/2, W/2, 3)
    variance = stacked.var(axis=0).mean(axis=2)  # (H/2, W/2)

    # Low variance = static content (overlay, watermark)
    # Use adaptive threshold: bottom 3% of variance values
    thresh = max(np.percentile(variance, 3), 8.0)
    static_mask_small = (variance < thresh).astype(np.uint8)

    # Only consider edge regions (overlays are always at edges/corners)
    sh, sw = static_mask_small.shape
    edge_mask = np.zeros_like(static_mask_small)
    edge_mask[:int(sh * 0.18), :] = 1          # Top 18%
    edge_mask[int(sh * 0.80):, :] = 1          # Bottom 20%
    edge_mask[:, :int(sw * 0.22)] = 1          # Left 22%
    edge_mask[:, int(sw * 0.78):] = 1          # Right 22%
    static_mask_small = static_mask_small * edge_mask

    # Morphological cleanup: remove noise, connect nearby regions
    kernel = np.ones((11, 11), np.uint8)
    static_mask_small = cv2.morphologyEx(static_mask_small, cv2.MORPH_CLOSE, kernel)
    static_mask_small = cv2.morphologyEx(static_mask_small, cv2.MORPH_OPEN, kernel)

    # Upscale back to full resolution
    static_mask_full = cv2.resize(static_mask_small, (w, h), interpolation=cv2.INTER_NEAREST)

    # Invert: 1=clean, 0=overlay
    overlay_mask = 1 - static_mask_full
    overlay_ratio = 1.0 - overlay_mask.mean()

    return overlay_mask, overlay_ratio


def visualize_overlay(frame, overlay_mask):
    """Show detected overlay regions highlighted in red."""
    import matplotlib.pyplot as plt

    vis = frame.copy()
    vis[overlay_mask == 0] = [0, 0, 255]  # Red for overlay regions

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    axes[0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Frame")
    axes[0].axis("off")

    blend = cv2.addWeighted(frame, 0.6, vis, 0.4, 0)
    axes[1].imshow(cv2.cvtColor(blend, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"Overlay regions (red) — {(1-overlay_mask.mean())*100:.1f}% masked")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()
