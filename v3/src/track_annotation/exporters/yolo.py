"""
Export annotation package -> YOLO training format.

Class modes
-----------
- "single"  : 1 class ("logo"). Use for Stage A class-agnostic detector.
- "brand"   : N classes = unique master brands present in annotations.
              RECOMMENDED for most training runs because it consolidates
              variants of the same brand (aon_red + aon_white -> "aon").
- "variant" : N classes = unique variants present in annotations.
              Use only when a fine-grained variant-aware model is needed.

Reads
-----
- <package>/manifest.json
- <package>/tracks/track_*/meta.json
- <package>/annotations.jsonl

Annotation contract (annotations.jsonl)
---------------------------------------
Each line is a JSON object:
    {
      "track_id": 42,
      "brand_id": "aon",
      "variant_id": "aon_red",
      ...
      "skip": false
    }

Tracks not present in annotations.jsonl are SKIPPED.

Writes
------
    out/
    ├── data.yaml                     # ultralytics format
    ├── images/{train,val}/
    ├── labels/{train,val}/           # one .txt per image: class cx cy w h (normalized)
    └── classes.txt
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Literal

import yaml

from track_annotation.utils.geometry import bbox_area
from track_annotation.utils.logging import get_logger
from track_annotation.utils.video_io import VideoReader

log = get_logger(__name__)

ClassMode = Literal["single", "brand", "variant"]


def export_to_yolo(
    package_dir: str | Path,
    output_dir: str | Path,
    val_ratio: float = 0.15,
    seed: int = 42,
    class_mode: ClassMode = "brand",
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
        Fraction of frames into the val split.
    seed : int
        RNG seed for split.
    class_mode : "single" | "brand" | "variant"
        See module docstring.

    Returns
    -------
    Path to YOLO dataset directory.
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

    # ---- Build class list ----
    if class_mode == "single":
        class_list = ["logo"]
        class_to_id = {"logo": 0}

        def _label_class_for(_ann: dict) -> str:
            return "logo"

    elif class_mode == "brand":
        used = sorted({
            a["brand_id"] for a in annotations.values()
            if not a.get("skip") and a.get("brand_id") not in (None, "unknown")
        })
        class_list = used
        class_to_id = {b: i for i, b in enumerate(class_list)}

        def _label_class_for(ann: dict) -> str:
            return ann["brand_id"]

    elif class_mode == "variant":
        used = sorted({
            a["variant_id"] for a in annotations.values()
            if not a.get("skip") and a.get("variant_id") not in (None, "unknown")
        })
        class_list = used
        class_to_id = {v: i for i, v in enumerate(class_list)}

        def _label_class_for(ann: dict) -> str:
            return ann["variant_id"]

    else:
        raise ValueError(f"Unknown class_mode: {class_mode}")

    if not class_list:
        raise RuntimeError(
            "No usable annotations found in package. "
            "Run reviewer first; if class_mode != 'single', ensure brand/variant fields are set."
        )

    # ---- Aggregate per-frame labels ----
    # frame_idx -> list of (class_id, bbox_xyxy)
    frame_labels: dict[int, list[tuple[int, list[float]]]] = {}
    n_tracks_used = 0

    for track_dir in sorted((package_dir / "tracks").iterdir()):
        if not track_dir.is_dir():
            continue
        meta = json.loads((track_dir / "meta.json").read_text())
        tid = meta["track_id"]
        ann = annotations.get(tid)
        if ann is None or ann.get("skip"):
            continue

        try:
            cls_name = _label_class_for(ann)
        except KeyError:
            log.warning(f"track {tid}: missing class field for class_mode={class_mode}")
            continue

        if cls_name not in class_to_id:
            log.warning(f"track {tid}: class '{cls_name}' not in class list, skipping")
            continue
        cls_id = class_to_id[cls_name]

        for det in meta["detections"]:
            frame_labels.setdefault(det["frame_idx"], []).append((cls_id, det["bbox"]))
        n_tracks_used += 1

    if not frame_labels:
        raise RuntimeError("No labeled detections to export.")

    # ---- Train/val split (frame-level to avoid leak) ----
    rng = random.Random(seed)
    all_frames = sorted(frame_labels.keys())
    rng.shuffle(all_frames)
    n_val = max(1, int(round(len(all_frames) * val_ratio)))
    val_frames = set(all_frames[:n_val])

    # ---- Write images + labels ----
    video_path = Path(manifest["video"]["path"])
    if not video_path.exists():
        raise FileNotFoundError(f"Source video not accessible: {video_path}")

    video_w = manifest["video"]["width"]
    video_h = manifest["video"]["height"]
    stem = video_path.stem

    log.info(
        f"Exporting class_mode={class_mode}, "
        f"{len(class_list)} classes, "
        f"{n_tracks_used} tracks, "
        f"{len(all_frames)} frames "
        f"({len(val_frames)} val)"
    )

    import cv2  # local import; only needed for export
    with VideoReader(video_path) as reader:
        for frame_idx in all_frames:
            split = "val" if frame_idx in val_frames else "train"
            img_dir = out_images_val if split == "val" else out_images_train
            lbl_dir = out_labels_val if split == "val" else out_labels_train

            frame = reader.read_at(frame_idx)
            if frame is None:
                continue
            img_name = f"{stem}_{frame_idx:08d}.jpg"
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

    # ---- data.yaml + classes.txt ----
    data_yaml = {
        "path": str(output_dir),
        "train": "images/train",
        "val": "images/val",
        "names": {i: n for i, n in enumerate(class_list)},
    }
    (output_dir / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))
    (output_dir / "classes.txt").write_text("\n".join(class_list))

    # Capture export metadata
    (output_dir / "export_meta.json").write_text(json.dumps({
        "source_package": str(package_dir),
        "class_mode": class_mode,
        "num_classes": len(class_list),
        "num_tracks_used": n_tracks_used,
        "num_frames": len(all_frames),
        "num_val": len(val_frames),
        "val_ratio": val_ratio,
        "seed": seed,
        "kit_context": manifest.get("match_context", {}).get("kit_context"),
    }, indent=2))

    log.info(f"YOLO export complete: {output_dir}")
    return output_dir


def _load_annotations_jsonl(path: Path) -> dict[int, dict]:
    """Last-write-wins per track_id."""
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
