#!/usr/bin/env python3
"""
Calibrate match.meta.yaml fields from a sample frame.

Two modes:

1) HSV color calibration — measure jersey color from a torso bbox
   python scripts/calibrate_meta.py hsv \\
       --frame sample.jpg \\
       --bbox 800 350 1100 700      # x1 y1 x2 y2 in pixel coords (torso of a Bradford player)

2) Ignore region normalization — convert overlay pixel bbox to [0..1] coords
   python scripts/calibrate_meta.py region \\
       --frame sample.jpg \\
       --bbox 0 920 960 1080        # x1 y1 x2 y2 in pixel coords (scoreboard)

Helper to extract a sample frame from a video at timestamp T (seconds):
   ffmpeg -ss 60 -i video.mp4 -frames:v 1 sample.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def cmd_hsv(args):
    img = cv2.imread(str(args.frame))
    if img is None:
        sys.exit(f'Cannot read frame: {args.frame}')
    h, w = img.shape[:2]
    x1, y1, x2, y2 = args.bbox
    if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
        sys.exit(f'bbox out of image bounds (image {w}x{h})')

    crop = img[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    H = hsv[..., 0].flatten()
    S = hsv[..., 1].flatten()
    V = hsv[..., 2].flatten()

    print(f'Frame    : {args.frame.name} ({w}x{h})')
    print(f'Crop bbox: ({x1}, {y1}) → ({x2}, {y2})  size={x2-x1}x{y2-y1}  pixels={crop.size//3}')
    print()
    print('HSV stats (over all pixels in bbox):')
    for ch, name in [(H, 'H'), (S, 'S'), (V, 'V')]:
        print(
            f'  {name}: median={int(np.median(ch)):3d}  '
            f'mean={int(ch.mean()):3d}  '
            f'p10={int(np.percentile(ch, 10)):3d}  '
            f'p90={int(np.percentile(ch, 90)):3d}  '
            f'min={int(ch.min()):3d}  max={int(ch.max()):3d}'
        )

    # Suggest range: H ± `pad`, S/V from p10
    h_med = int(np.median(H))
    s_p10 = max(0, int(np.percentile(S, 10)) - 20)
    v_p10 = max(0, int(np.percentile(V, 10)) - 20)
    pad = args.h_pad

    # Detect red wrap-around (median near 0 or 180)
    if h_med <= pad:
        h_low_main, h_high_main = 0, h_med + pad
        wrap = (180 - pad + h_med, 180)
    elif h_med >= 180 - pad:
        h_low_main, h_high_main = h_med - pad, 180
        wrap = (0, h_med - (180 - pad))
    else:
        h_low_main, h_high_main = h_med - pad, h_med + pad
        wrap = None

    name = args.name or 'color_1'
    print()
    print('Suggested entry for match.meta.yaml:')
    print('  primary_colors:')
    print(f'    - {{name: {name}, h: [{h_low_main}, {h_high_main}], '
          f's: [{s_p10}, 255], v: [{v_p10}, 255]}}')
    if wrap:
        print(f'    - {{name: {name}_wrap, h: [{wrap[0]}, {wrap[1]}], '
              f's: [{s_p10}, 255], v: [{v_p10}, 255]}}')

    if args.save_crop:
        out = args.save_crop
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), crop)
        print(f'\nSaved crop → {out}')


def cmd_region(args):
    img = cv2.imread(str(args.frame))
    if img is None:
        sys.exit(f'Cannot read frame: {args.frame}')
    h, w = img.shape[:2]
    x1, y1, x2, y2 = args.bbox
    if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
        sys.exit(f'bbox out of image bounds (image {w}x{h})')

    nx1, ny1 = x1 / w, y1 / h
    nx2, ny2 = x2 / w, y2 / h

    print(f'Frame      : {args.frame.name} ({w}x{h})')
    print(f'Pixel bbox : ({x1}, {y1}) → ({x2}, {y2})')
    print(f'Normalized : [{nx1:.4f}, {ny1:.4f}, {nx2:.4f}, {ny2:.4f}]')
    print()
    print('Add to match.meta.yaml under `ignore_regions:`')
    print(f'  - [{nx1:.2f}, {ny1:.2f}, {nx2:.2f}, {ny2:.2f}]')
    if args.label:
        print(f'  # ↑ {args.label}')

    if args.save_overlay:
        out = args.save_overlay
        annotated = img.copy()
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 4)
        cv2.putText(
            annotated, args.label or 'ignore region',
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), annotated)
        print(f'\nSaved overlay → {out}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)

    hsv = sub.add_parser('hsv', help='Compute HSV stats from a torso bbox')
    hsv.add_argument('--frame', type=Path, required=True)
    hsv.add_argument('--bbox', type=int, nargs=4, required=True, metavar=('X1', 'Y1', 'X2', 'Y2'))
    hsv.add_argument('--name', type=str, default=None, help='Color name in suggestion (e.g., red)')
    hsv.add_argument('--h-pad', type=int, default=10, help='Hue half-width around median (default 10)')
    hsv.add_argument('--save-crop', type=Path, default=None, help='Save the torso crop for visual check')
    hsv.set_defaults(func=cmd_hsv)

    reg = sub.add_parser('region', help='Convert pixel bbox to normalized ignore_region')
    reg.add_argument('--frame', type=Path, required=True)
    reg.add_argument('--bbox', type=int, nargs=4, required=True, metavar=('X1', 'Y1', 'X2', 'Y2'))
    reg.add_argument('--label', type=str, default=None, help='Free-text label for the region')
    reg.add_argument('--save-overlay', type=Path, default=None, help='Save frame with region drawn for visual check')
    reg.set_defaults(func=cmd_region)

    args = p.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
