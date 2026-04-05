"""
Player Visibility Filtering
Filters sampled frames to keep only those where players are clearly visible:
- Min bounding box size (player not too far away)
- Sharpness check (not motion-blurred)
- Exports qualified frames to frames_clear/ for annotation
"""

import csv
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

from .config import (
    DEVICE,
    FRAMES_CLEAR_DIR,
    FRAMES_DIR,
    METADATA_DIR,
    MIN_PLAYER_AREA_RATIO,
    MIN_SHARPNESS,
    PERSON_CONFIDENCE,
)


def _compute_sharpness(image: np.ndarray) -> float:
    """Compute sharpness score using Laplacian variance."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _compute_region_sharpness(image: np.ndarray, bbox: list[float]) -> float:
    """Compute sharpness of a specific region (player bounding box)."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return 0.0

    region = image[y1:y2, x1:x2]
    return _compute_sharpness(region)


class PlayerVisibilityFilter:
    """
    Filters frames to keep only those with clearly visible players.
    """

    def __init__(
        self,
        min_area_ratio: float = MIN_PLAYER_AREA_RATIO,
        min_sharpness: float = MIN_SHARPNESS,
        person_confidence: float = PERSON_CONFIDENCE,
    ):
        self.min_area_ratio = min_area_ratio
        self.min_sharpness = min_sharpness
        self.person_confidence = person_confidence
        self._yolo_model = None

    @property
    def yolo_model(self):
        if self._yolo_model is None:
            print("[Filter] Loading YOLOv8 model...")
            self._yolo_model = YOLO("yolov8n.pt")
        return self._yolo_model

    def filter_frames(
        self,
        frames_dir: Path = FRAMES_DIR,
        output_dir: Path = FRAMES_CLEAR_DIR,
        metadata_dir: Path = METADATA_DIR,
        frames_index_csv: Path = None,
    ) -> pd.DataFrame:
        """
        Filter frames for clear player visibility.

        Returns DataFrame with filtering results.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        if frames_index_csv is None:
            frames_index_csv = metadata_dir / "frames_index.csv"

        if not frames_index_csv.exists():
            raise FileNotFoundError(
                f"frames_index.csv not found at {frames_index_csv}. "
                "Run frame sampling first."
            )

        df = pd.read_csv(frames_index_csv)
        frame_files = sorted(frames_dir.glob("frame_*.jpg"))

        if not frame_files:
            raise FileNotFoundError(f"No frame files found in {frames_dir}")

        print(f"\n{'='*60}")
        print(f"PLAYER VISIBILITY FILTERING")
        print(f"{'='*60}")
        print(f"  Input frames:     {len(frame_files)}")
        print(f"  Min area ratio:   {self.min_area_ratio:.1%}")
        print(f"  Min sharpness:    {self.min_sharpness}")
        print(f"{'='*60}\n")

        results = []
        passed_count = 0

        for frame_file in tqdm(frame_files, desc="Filtering frames"):
            frame = cv2.imread(str(frame_file))
            if frame is None:
                continue

            frame_area = frame.shape[0] * frame.shape[1]

            # Detect persons
            detections = self.yolo_model.predict(
                frame,
                classes=[0],
                conf=self.person_confidence,
                device=DEVICE,
                verbose=False,
            )

            best_player = None
            best_area_ratio = 0.0
            best_sharpness = 0.0
            has_clear_player = False

            if detections and len(detections[0].boxes) > 0:
                boxes = detections[0].boxes
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    bbox = [float(x1), float(y1), float(x2), float(y2)]
                    bbox_area = (x2 - x1) * (y2 - y1)
                    area_ratio = bbox_area / frame_area

                    if area_ratio < self.min_area_ratio:
                        continue

                    sharpness = _compute_region_sharpness(frame, bbox)

                    if sharpness < self.min_sharpness:
                        continue

                    # This player is clearly visible
                    if area_ratio > best_area_ratio:
                        best_area_ratio = float(area_ratio)
                        best_sharpness = sharpness
                        best_player = bbox
                        has_clear_player = True

            result = {
                "filename": frame_file.name,
                "has_clear_player": has_clear_player,
                "best_area_ratio": best_area_ratio,
                "best_sharpness": best_sharpness,
            }
            results.append(result)

            if has_clear_player:
                passed_count += 1
                # Copy frame to frames_clear/
                shutil.copy2(frame_file, output_dir / frame_file.name)

        results_df = pd.DataFrame(results)

        # Save filtering results
        filter_csv = metadata_dir / "frames_filter_results.csv"
        results_df.to_csv(filter_csv, index=False)

        # Update frames_index.csv with visibility info
        if "is_player_visible" not in df.columns:
            visibility_map = {
                r["filename"]: r["has_clear_player"] for r in results
            }
            df["is_player_visible"] = df["filename"].map(visibility_map).fillna(False)
            df.to_csv(frames_index_csv, index=False)

        print(f"\n{'='*60}")
        print(f"FILTERING RESULTS")
        print(f"{'='*60}")
        print(f"  Input frames:      {len(frame_files)}")
        print(f"  Clear players:     {passed_count} "
              f"({passed_count/len(frame_files)*100:.1f}%)")
        print(f"  Rejected:          {len(frame_files) - passed_count}")
        print(f"  Output dir:        {output_dir}")
        print(f"  Filter details:    {filter_csv}")
        print(f"{'='*60}\n")

        return results_df
