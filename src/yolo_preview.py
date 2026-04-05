"""
Preview YOLO-format labels (normalized cx, cy, w, h) on images.
Works with autodistill / Roboflow exports that include data.yaml.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np


def parse_data_yaml_names(data_yaml: Path) -> list[str]:
    """Read `names:` list from a YOLO data.yaml without requiring PyYAML."""
    names: list[str] = []
    in_block = False
    for line in data_yaml.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("names:"):
            in_block = True
            continue
        if not in_block:
            continue
        if stripped.startswith("- "):
            names.append(stripped[2:].strip().strip("'\""))
            continue
        if names and stripped and ":" in stripped:
            if not line.startswith(" ") and not line.startswith("\t"):
                break
    return names


def yolo_lines_to_boxes(
    lines: list[str],
    img_w: int,
    img_h: int,
) -> list[tuple[int, float, float, float, float]]:
    """Parse YOLO label lines -> list of (cls_id, x1, y1, x2, y2) in pixel coords."""
    out: list[tuple[int, float, float, float, float]] = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(float(parts[0]))
        cx, cy, bw, bh = map(float, parts[1:5])
        x1 = (cx - bw / 2) * img_w
        y1 = (cy - bh / 2) * img_h
        x2 = (cx + bw / 2) * img_w
        y2 = (cy + bh / 2) * img_h
        out.append((cls_id, x1, y1, x2, y2))
    return out


def draw_yolo_on_image(
    image_path: Path,
    label_path: Path | None,
    class_names: list[str] | None = None,
) -> np.ndarray:
    """
    Load image and draw bounding boxes. Returns BGR image (OpenCV convention).
    If label_path is missing or empty, returns the original image.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    h, w = img.shape[:2]
    if not label_path or not label_path.is_file():
        return img
    lines = label_path.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        return img
    boxes = yolo_lines_to_boxes(lines, w, h)
    n_cls = len(class_names) if class_names else 0
    for cls_id, x1, y1, x2, y2 in boxes:
        color = plt.cm.tab20(cls_id % 20)[:3]
        bgr = (int(color[2] * 255), int(color[1] * 255), int(color[0] * 255))
        x1i, y1i = int(max(0, x1)), int(max(0, y1))
        x2i, y2i = int(min(w - 1, x2)), int(min(h - 1, y2))
        cv2.rectangle(img, (x1i, y1i), (x2i, y2i), bgr, 2)
        if class_names and 0 <= cls_id < len(class_names):
            label = class_names[cls_id]
        else:
            label = str(cls_id)
        cv2.putText(
            img,
            label,
            (x1i, max(0, y1i - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            bgr,
            1,
            cv2.LINE_AA,
        )
    return img


def show_yolo_preview(
    image_path: Path,
    label_path: Path | None = None,
    class_names: list[str] | None = None,
    figsize: tuple[float, float] = (14, 8),
) -> None:
    """Matplotlib display (RGB) for notebooks."""
    if label_path is None:
        label_path = image_path.parent.parent / "labels" / (image_path.stem + ".txt")
    bgr = draw_yolo_on_image(image_path, label_path, class_names)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(rgb)
    ax.set_title(image_path.name)
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def iter_split_image_paths(
    dataset_root: Path,
    split: str = "train",
    limit: int | None = None,
    pattern: str = "*.*",
) -> Iterable[Path]:
    """Yield image paths under dataset_root/{split}/images/."""
    img_dir = dataset_root / split / "images"
    if not img_dir.is_dir():
        return
    paths = sorted(img_dir.glob(pattern))
    if limit is not None:
        paths = paths[:limit]
    yield from paths


def preview_dataset_samples(
    dataset_root: Path,
    num_frames: int = 6,
    split: str = "train",
    seed: int | None = 42,
    figsize_per: tuple[float, float] = (5, 4),
) -> None:
    """
    Show a grid of random (or seeded) frames with YOLO boxes from data.yaml names.

    dataset_root: folder containing data.yaml and train/images, train/labels, etc.
    """
    dataset_root = Path(dataset_root)
    yaml_path = dataset_root / "data.yaml"
    class_names = parse_data_yaml_names(yaml_path) if yaml_path.is_file() else None

    img_dir = dataset_root / split / "images"
    if not img_dir.is_dir():
        raise FileNotFoundError(f"Missing images dir: {img_dir}")

    all_imgs = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    if not all_imgs:
        raise FileNotFoundError(f"No jpg/png under {img_dir}")

    rng = np.random.default_rng(seed)
    k = min(num_frames, len(all_imgs))
    if len(all_imgs) <= num_frames:
        picks = all_imgs
    else:
        idx = rng.choice(len(all_imgs), size=k, replace=False)
        picks = [all_imgs[int(i)] for i in idx]

    label_root = dataset_root / split / "labels"
    n = len(picks)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(figsize_per[0] * cols, figsize_per[1] * rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for ax, ip in zip(axes, picks):
        lp = label_root / (ip.stem + ".txt")
        bgr = draw_yolo_on_image(ip, lp if lp.is_file() else None, class_names)
        ax.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        n_boxes = len(lp.read_text(encoding="utf-8").strip().splitlines()) if lp.is_file() else 0
        ax.set_title(f"{ip.name}\n({n_boxes} boxes)", fontsize=9)
        ax.axis("off")

    for j in range(len(picks), len(axes)):
        axes[j].axis("off")

    plt.suptitle(f"{dataset_root.name} / {split}", fontsize=12)
    plt.tight_layout()
    plt.show()
