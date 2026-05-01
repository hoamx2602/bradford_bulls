"""
Export annotation package -> YOLO training format.

Reads:
  - <package>/manifest.json
  - <package>/tracks/track_*/meta.json
  - <package>/annotations.jsonl     (filled by reviewer; one line per track-brand)

Writes YOLO directory tree:
  out/
  ├── data.yaml                     # ultralytics format
  ├── images/
  │   ├── train/
  │   └── val/
  ├── labels/
  │   ├── train/                    # one .txt per image: class cx cy w h (normalized)
  │   └── val/
  └── classes.txt

Annotation contract (annotations.jsonl)
---------------------------------------
Each line is a JSON object with shape:
    {
      "track_id": 42,
      "brand_id": "aon_red",        # one of brand_ids in manifest, or "unknown"
      "position": "chest_front",    # optional
      "visibility_quality": "clear",# optional
      "is_target_team": true,       # optional, default true
      "skip": false                 # if true, this track is excluded from export
    }

Tracks not present in annotations.jsonl are SKIPPED (treated as not yet labeled).
"""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

import yaml

from track_annotation.utils.geometry import bbox_area
from track_annotation.utils.logging import get_logger
from track_annotation.utils.video_io import VideoReader

log = get_logger(__name__)


def export_to_yolo(
    package_dir: str | Path,
    output_dir: str | Path,
    val_ratio: float = 0.15,
    seed: int = 42,
    single_class: bool = False,
    class_name: str = "logo",
) -> Path:
    """
    Export an annotation package to YOLO training format.

    Parameters
    ----------
    package_dir : str | Path
        Annotation package directory (output of build_package).
    output_dir : str | Path
        Where to write the YOLO dataset.
    val_ratio : float
        Fraction of images to put into the val split.
    seed : int
        RNG seed for split.
    single_class : bool
        If True, exports a single-class "logo" dataset for Stage A training.
        If False, exports per-brand classes (for Stage A+B combined or testing).
    class_name : str
        Class name when single_class=True.

    Returns
    -------
    Path
        Path to the YOLO dataset directory.
    """
    package_dir = Path(package_dir).resolve()
    output_dir = Path(output_dir).resolve()
    out_images_train = output_dir / "images" / "train"
    out_images_val = output_dir / "images" / "val"
    out_labels_train = output_dir / "labels" / "train"
    out_labels_val = output_dir / "labels" / "val"
    for d in (out_images_train, out_images_val, out_labels_train, out_labels_val):
        d.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((package_dir / "manifest.json").read_text())
    annotations = _load_annotations_jsonl(package_dir / "annotations.jsonl")

    # Build class list
    if single_class:
        class_list = [class_name]
        class_to_id = {class_name: 0}
    else:
        used_brands = sorted({
            a["brand_id"] for a in annotations.values()
            if not a.get("skip") and a.get("brand_id") not in (None, "unknown")
        })
        class_list = used_brands
        class_to_id = {b: i for i, b in enumerate(class_list)}

    if not class_list:
        raise RuntimeError("No usable annotations found in package; run reviewer first.")

    # Per-frame label aggregation: a frame may host multiple tracks
    # frame_idx -> list of (class_id, bbox)
    frame_labels: dict[int, list[tuple[int, list[float]]]] = {}
    track_meta: dict[int, dict] = {}

    for track_dir in sorted((package_dir / "tracks").iterdir()):
        if not track_dir.is_dir():
            continue
        meta = json.loads((track_dir / "meta.json").read_text())
        tid = meta["track_id"]
        ann = annotations.get(tid)
        if ann is None or ann.get("skip"):
            continue

        cls_name = class_name if single_class else ann["brand_id"]
        if cls_name not in class_to_id:
            log.warning(f"track {tid}: brand '{cls_name}' not in class list, skipping")
            continue
        cls_id = class_to_id[cls_name]

        # Use ALL detections in the track (label propagation: same brand for all frames)
        for det in meta["detections"]:
            frame_idx = det["frame_idx"]
            frame_labels.setdefault(frame_idx, []).append((cls_id, det["bbox"]))

        track_meta[tid] = meta

    if not frame_labels:
        raise RuntimeError("No labeled detections to export.")

    # Train/val split at FRAME level (not track) to avoid leak
    rng = random.Random(seed)
    all_frames = sorted(frame_labels.keys())
    rng.shuffle(all_frames)
    n_val = max(1, int(round(len(all_frames) * val_ratio)))
    val_frames = set(all_frames[:n_val])

    # Write images + labels
    video_path = Path(manifest["video"]["path"])
    if not video_path.exists():
        raise FileNotFoundError(f"Source video not accessible: {video_path}")

    video_w = manifest["video"]["width"]
    video_h = manifest["video"]["height"]
    stem = video_path.stem

    log.info(f"Exporting {len(all_frames)} frames ({len(val_frames)} val, classes={len(class_list)})")
    with VideoReader(video_path) as reader:
        for frame_idx in all_frames:
            split = "val" if frame_idx in val_frames else "train"
            img_dir = out_images_val if split == "val" else out_images_train
            lbl_dir = out_labels_val if split == "val" else out_labels_train

            frame = reader.read_at(frame_idx)
            if frame is None:
                continue
            img_name = f"{stem}_{frame_idx:08d}.jpg"
            import cv2
            cv2.imwrite(str(img_dir / img_name), frame)

            label_lines = []
            for cls_id, bbox in frame_labels[frame_idx]:
                if bbox_area(tuple(bbox)) <= 0:
                    continue
                x1, y1, x2, y2 = bbox
                cx = ((x1 + x2) / 2.0) / video_w
                cy = ((y1 + y2) / 2.0) / video_h
                w = (x2 - x1) / video_w
                h = (y2 - y1) / video_h
                label_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            (lbl_dir / f"{stem}_{frame_idx:08d}.txt").write_text("\n".join(label_lines))

    # data.yaml
    data_yaml = {
        "path": str(output_dir),
        "train": "images/train",
        "val": "images/val",
        "names": {i: n for i, n in enumerate(class_list)},
    }
    (output_dir / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))
    (output_dir / "classes.txt").write_text("\n".join(class_list))

    log.info(f"YOLO export complete: {output_dir}")
    return output_dir


def _load_annotations_jsonl(path: Path) -> dict[int, dict]:
    """Load annotations.jsonl into a dict keyed by track_id (last write wins)."""
    if not path.exists():
        return {}
    out: dict[int, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                out[int(obj["track_id"])] = obj
            except (json.JSONDecodeError, KeyError) as e:
                log.warning(f"Skipping malformed annotation line: {e}")
    return out
