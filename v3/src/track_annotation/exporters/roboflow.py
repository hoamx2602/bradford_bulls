"""
Upload an annotation package to Roboflow as a labeled dataset.

Requires:
  - ROBOFLOW_API_KEY env var, or pass api_key= to upload_to_roboflow().
  - A Roboflow project already created.

Note: this is a thin convenience wrapper around the roboflow SDK. For more
control (versioning, augmentation), use Roboflow UI directly.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from track_annotation.utils.logging import get_logger

log = get_logger(__name__)


def upload_to_roboflow(
    yolo_dataset_dir: str | Path,
    workspace: str,
    project: str,
    api_key: str | None = None,
    batch_name: str | None = None,
) -> None:
    """
    Upload a YOLO dataset (output of export_to_yolo) to a Roboflow project.

    Parameters
    ----------
    yolo_dataset_dir : str | Path
        Path to YOLO dataset (must contain images/ and labels/).
    workspace : str
        Roboflow workspace slug.
    project : str
        Roboflow project slug.
    api_key : str | None
        Roboflow API key. If None, reads from ROBOFLOW_API_KEY env var.
    batch_name : str | None
        Optional batch name for grouping uploads in Roboflow UI.
    """
    api_key = api_key or os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise ValueError(
            "Roboflow API key required. Set ROBOFLOW_API_KEY env var or pass api_key=..."
        )

    try:
        from roboflow import Roboflow  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError(
            "roboflow SDK not installed. Run: pip install roboflow>=1.1.0"
        ) from e

    yolo_dataset_dir = Path(yolo_dataset_dir).resolve()
    images_dir = yolo_dataset_dir / "images"
    labels_dir = yolo_dataset_dir / "labels"
    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(f"Expected images/ and labels/ in {yolo_dataset_dir}")

    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)

    uploaded = 0
    failed = 0
    for split in ("train", "val"):
        for img_path in sorted((images_dir / split).glob("*.jpg")):
            label_path = labels_dir / split / f"{img_path.stem}.txt"
            try:
                proj.upload(
                    image_path=str(img_path),
                    annotation_path=str(label_path) if label_path.exists() else None,
                    batch_name=batch_name,
                    split=split,
                )
                uploaded += 1
            except Exception as e:  # noqa: BLE001
                log.warning(f"Upload failed for {img_path.name}: {e}")
                failed += 1

    log.info(f"Roboflow upload: {uploaded} succeeded, {failed} failed")
