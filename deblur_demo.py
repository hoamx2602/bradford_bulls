#!/usr/bin/env python3
"""
Logo clarity demo — Crop torso + Super-Resolution bằng Real-ESRGAN.

Vấn đề gốc: logo trên áo cầu thủ chỉ 30-60px wide trong frame 1080p.
Deblur trên full frame không hiệu quả với object nhỏ như vậy.
Giải pháp đúng: Crop vùng torso → Upscale 4x → Annotator thấy rõ.

SETUP (chạy 1 lần):
    pip install realesrgan basicsr facexlib gfpgan

CHẠY:
    # Tự detect torso (dùng bbox thủ công nếu biết vị trí)
    python deblur_demo.py frame1.jpg frame2.jpg

    # Chỉ định bbox torso thủ công (x1 y1 x2 y2) nếu không dùng YOLO
    python deblur_demo.py frame1.jpg --bbox 420 180 620 420

    # Dùng YOLO để auto-detect person + crop torso
    python deblur_demo.py frame1.jpg --yolo weights/yolo11l.pt

OUTPUT:
    sr_output/
    ├── frame1_original.jpg          — ảnh gốc full frame
    ├── frame1_torso_orig.jpg        — crop torso gốc (nhỏ, mờ)
    ├── frame1_torso_sr.jpg          — crop torso sau SR 4× (to, nét)
    └── frame1_compare.jpg           — so sánh side-by-side
"""

from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import cv2
import numpy as np

OUT_QUALITY = 95


# ─── Super-Resolution: Real-ESRGAN ───────────────────────────────────────────

def load_realesrgan(scale: int = 4, tile: int = 256):
    """Load Real-ESRGAN model. Cài: pip install realesrgan"""
    try:
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet
    except ImportError as e:
        return None, (
            f"Import lỗi: {e}\n"
            f"Thử: pip install realesrgan basicsr facexlib gfpgan\n"
            f"Debug: python -c \"from realesrgan import RealESRGANer\""
        )

    import torch

    # Model x4plus — tốt nhất cho general images (blur + noise + artifact)
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=23, num_grow_ch=32, scale=scale)

    weights_candidates = [
        Path("weights/RealESRGAN_x4plus.pth"),
        Path("RealESRGAN_x4plus.pth"),
    ]
    weights_path = next((p for p in weights_candidates if p.exists()), None)

    if weights_path is None:
        # Tự download nếu chưa có
        print("  Downloading RealESRGAN_x4plus.pth (~67MB)...")
        url = ("https://github.com/xinntao/Real-ESRGAN/releases/download/"
               "v0.1.0/RealESRGAN_x4plus.pth")
        Path("weights").mkdir(exist_ok=True)
        weights_path = Path("weights/RealESRGAN_x4plus.pth")
        try:
            import urllib.request
            urllib.request.urlretrieve(url, str(weights_path))
        except Exception as e:
            return None, f"Download thất bại: {e}. Tải thủ công từ {url}"

    upsampler = RealESRGANer(
        scale=scale,
        model_path=str(weights_path),
        model=model,
        tile=tile,           # tile để tránh OOM; 0 = no tiling (cần VRAM lớn)
        tile_pad=10,
        pre_pad=0,
        half=False,          # True nếu có GPU và muốn nhanh hơn
    )
    print(f"  Real-ESRGAN x{scale} loaded  (tile={tile})")
    return upsampler, None


def apply_sr(upsampler, img_bgr: np.ndarray, scale: int = 4) -> np.ndarray:
    """Chạy SR, trả về ảnh upscaled."""
    out, _ = upsampler.enhance(img_bgr, outscale=scale)
    return out


# ─── Crop torso từ full frame ─────────────────────────────────────────────────

def crop_torso_from_bbox(frame: np.ndarray,
                         bbox_xyxy: tuple[int,int,int,int],
                         torso_top: float = 0.10,
                         torso_bot: float = 0.55,
                         pad: float = 0.10) -> tuple[np.ndarray, tuple]:
    """
    Từ person bbox (xyxy), cắt vùng torso (nơi có logo).

    torso_top/bot: tỷ lệ chiều cao bbox từ trên xuống
    pad: padding thêm ra mỗi phía

    Trả về (torso_crop, actual_xyxy_in_frame)
    """
    fh, fw = frame.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    bw, bh = x2 - x1, y2 - y1

    # Vùng torso
    ty1 = int(y1 + bh * torso_top)
    ty2 = int(y1 + bh * torso_bot)
    tx1 = int(x1 + bw * pad)
    tx2 = int(x2 - bw * pad)

    # Clamp
    ty1 = max(0, ty1); ty2 = min(fh, ty2)
    tx1 = max(0, tx1); tx2 = min(fw, tx2)

    return frame[ty1:ty2, tx1:tx2].copy(), (tx1, ty1, tx2, ty2)


def detect_persons_yolo(frame: np.ndarray,
                         weights: Path,
                         conf: float = 0.5) -> list[tuple[int,int,int,int]]:
    """YOLO person detection. Trả về list bbox xyxy."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("  ultralytics chưa cài: pip install ultralytics")
        return []

    model = YOLO(str(weights))
    results = model.predict(frame, classes=[0], conf=conf, verbose=False)
    if not results or results[0].boxes is None:
        return []
    bboxes = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].cpu().numpy())
        bw, bh = x2 - x1, y2 - y1
        if bw >= 40 and bh >= 80 and bh / max(bw, 1) >= 1.0:
            bboxes.append((x1, y1, x2, y2))
    return bboxes


# ─── Visualization ────────────────────────────────────────────────────────────

def add_label(img: np.ndarray, text: str,
              color=(255,255,255), bg=(0,0,0)) -> np.ndarray:
    out = img.copy()
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
    cv2.rectangle(out, (0, 0), (tw + 16, th + 18), bg, -1)
    cv2.putText(out, text, (8, th + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)
    return out


def make_comparison(orig_crop: np.ndarray,
                    sr_crop: np.ndarray,
                    orig_label: str = "ORIGINAL (crop)",
                    sr_label: str = "Real-ESRGAN 4×") -> np.ndarray:
    """
    Side-by-side: crop gốc (nhỏ, upscaled với NEAREST để thấy pixel)
    vs SR output (to, nét).
    """
    target_h = sr_crop.shape[0]
    # Scale orig lên cùng kích thước NEAREST NEIGHBOR để thấy độ mờ thực
    scale = target_h / orig_crop.shape[0]
    orig_display = cv2.resize(
        orig_crop,
        (int(orig_crop.shape[1] * scale), target_h),
        interpolation=cv2.INTER_NEAREST  # intentionally blocky để so sánh
    )
    left  = add_label(orig_display, orig_label, color=(80, 80, 255))
    right = add_label(sr_crop,      sr_label,   color=(80, 255, 80))

    # Separator
    sep = np.full((target_h, 4, 3), 200, dtype=np.uint8)
    return np.hstack([left, sep, right])


def sharpness(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# ─── Main ─────────────────────────────────────────────────────────────────────

def process_one(img_path: Path, args, upsampler, scale: int) -> None:
    print(f"\n{'─'*60}")
    print(f"  {img_path.name}")

    frame = cv2.imread(str(img_path))
    if frame is None:
        print("  ✗ Không đọc được ảnh"); return

    fh, fw = frame.shape[:2]
    stem = img_path.stem

    # ── Xác định bboxes cần xử lý ──
    if args.bbox:
        bboxes = [tuple(args.bbox)]
        print(f"  Dùng bbox thủ công: {bboxes[0]}")
    elif args.yolo:
        bboxes = detect_persons_yolo(frame, args.yolo, conf=args.conf)
        print(f"  YOLO detect: {len(bboxes)} person(s)")
        if not bboxes:
            print("  Không detect được cầu thủ nào."); return
    else:
        # Dùng 1/3 giữa frame làm vùng mặc định nếu không có bbox
        bboxes = [(fw//4, fh//6, 3*fw//4, 5*fh//6)]
        print(f"  Không có bbox/YOLO — dùng center crop: {bboxes[0]}")

    # Lưu full frame gốc
    cv2.imwrite(str(args.output / f"{stem}_original.jpg"),
                frame, [cv2.IMWRITE_JPEG_QUALITY, OUT_QUALITY])

    # ── Xử lý từng bbox ──
    for i, bbox in enumerate(bboxes):
        sfx = f"_p{i}" if len(bboxes) > 1 else ""

        # Crop torso
        torso, (tx1, ty1, tx2, ty2) = crop_torso_from_bbox(frame, bbox)
        if torso.size == 0:
            print(f"  Person {i}: torso crop rỗng, bỏ qua"); continue

        th_orig, tw_orig = torso.shape[:2]
        s_orig = sharpness(torso)
        print(f"\n  Person {i}: torso {tw_orig}×{th_orig}px  sharpness={s_orig:.1f}")

        # Lưu torso gốc (kéo to bằng NEAREST để thấy pixel)
        torso_display = cv2.resize(torso, (tw_orig * scale, th_orig * scale),
                                   interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(args.output / f"{stem}{sfx}_torso_orig.jpg"),
                    torso_display, [cv2.IMWRITE_JPEG_QUALITY, OUT_QUALITY])

        # ── Super-Resolution ──
        t0 = time.time()
        torso_sr = apply_sr(upsampler, torso, scale=scale)
        elapsed = time.time() - t0

        s_sr = sharpness(torso_sr)
        gain = s_sr / max(s_orig, 1e-3)
        print(f"  Real-ESRGAN {scale}×: {tw_orig*scale}×{th_orig*scale}px  "
              f"sharpness={s_sr:.1f} ({gain:.1f}×)  [{elapsed:.1f}s]")

        cv2.imwrite(str(args.output / f"{stem}{sfx}_torso_sr.jpg"),
                    torso_sr, [cv2.IMWRITE_JPEG_QUALITY, OUT_QUALITY])

        # ── So sánh side-by-side ──
        cmp = make_comparison(torso, torso_sr,
                               orig_label=f"Orig {tw_orig}×{th_orig}px",
                               sr_label=f"SR {tw_orig*scale}×{th_orig*scale}px")
        cv2.imwrite(str(args.output / f"{stem}{sfx}_compare.jpg"),
                    cmp, [cv2.IMWRITE_JPEG_QUALITY, OUT_QUALITY])

        # ── Annotate full frame: bbox + vùng torso ──
        annotated = frame.copy()
        x1, y1, x2, y2 = bbox
        cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,200,0), 2)      # person
        cv2.rectangle(annotated, (tx1,ty1), (tx2,ty2), (0,100,255), 3) # torso
        cv2.putText(annotated, "torso (logo zone)",
                    (tx1, ty1-8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0,100,255), 2, cv2.LINE_AA)
        cv2.imwrite(str(args.output / f"{stem}{sfx}_annotated.jpg"),
                    annotated, [cv2.IMWRITE_JPEG_QUALITY, OUT_QUALITY])

        print(f"  → {args.output}/{stem}{sfx}_*.jpg")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
             formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="*", help="Image file(s)")
    ap.add_argument("--folder",  type=Path, help="Folder ảnh đầu vào")
    ap.add_argument("--output",  type=Path, default=Path("sr_output"))
    ap.add_argument("--scale",   type=int, default=4, choices=[2, 4],
                    help="Upscale factor (default 4)")
    ap.add_argument("--tile",    type=int, default=256,
                    help="Tile size cho SR (nhỏ hơn nếu OOM, default 256)")
    ap.add_argument("--bbox",    type=int, nargs=4, metavar=("X1","Y1","X2","Y2"),
                    help="Person bbox thủ công trong ảnh (xyxy pixels)")
    ap.add_argument("--yolo",    type=Path, default=None,
                    help="Path đến YOLO weights để auto-detect person")
    ap.add_argument("--conf",    type=float, default=0.5,
                    help="YOLO confidence threshold")
    args = ap.parse_args()

    # Collect images
    paths: list[Path] = [Path(p) for p in args.images]
    if args.folder:
        for ext in ("*.jpg","*.jpeg","*.png","*.JPG","*.JPEG","*.PNG"):
            paths.extend(sorted(args.folder.glob(ext)))
    if not paths:
        ap.error("Cần ảnh đầu vào: python deblur_demo.py frame1.jpg ...")

    args.output.mkdir(parents=True, exist_ok=True)

    # Load model một lần
    print("Loading Real-ESRGAN...")
    upsampler, err = load_realesrgan(scale=args.scale, tile=args.tile)
    if upsampler is None:
        print(f"✗ {err}"); sys.exit(1)

    for p in paths:
        process_one(p, args, upsampler, args.scale)

    print(f"\n{'='*60}")
    print(f"Xong. Output → {args.output}/")
    print("File quan trọng nhất: *_compare.jpg")


if __name__ == "__main__":
    main()
