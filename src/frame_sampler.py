"""
Smart 3-Layer Frame Sampling
L1: Temporal Sampling (reduce FPS)
L2: Scene Change Detection (pHash + SSIM)
L3: Player Presence Filter (YOLOv8 person detection)

Saves frames with timestamp mapping to CSV.
"""

import csv
from pathlib import Path
from typing import Optional

import cv2
import imagehash
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
from ultralytics import YOLO

from .config import (
    DEVICE,
    FRAMES_DIR,
    METADATA_DIR,
    MIN_PERSONS_IN_FRAME,
    PERSON_CONFIDENCE,
    PHASH_THRESHOLD,
    SSIM_THRESHOLD,
    TARGET_FPS,
)
from .video_pipeline import VideoMetadata


def _format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}h{m:02d}m{s:02d}s"
    return f"{m:02d}m{s:02d}s"


def _compute_phash(frame_bgr: np.ndarray) -> imagehash.ImageHash:
    """Compute perceptual hash of a frame."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    return imagehash.phash(pil_img)


def _compute_ssim(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """Compute SSIM between two frames (grayscale)."""
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    # Resize to same dimensions if needed
    if gray1.shape != gray2.shape:
        gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))
    score = ssim(gray1, gray2)
    return float(score)


class FrameSampler:
    """
    Smart frame sampler with 3-layer filtering.
    Each layer can be enabled/disabled independently.
    """

    def __init__(
        self,
        target_fps: int = TARGET_FPS,
        phash_threshold: int = PHASH_THRESHOLD,
        ssim_threshold: float = SSIM_THRESHOLD,
        person_confidence: float = PERSON_CONFIDENCE,
        min_persons: int = MIN_PERSONS_IN_FRAME,
        enable_l2: bool = True,
        enable_l3: bool = True,
    ):
        self.target_fps = target_fps
        self.phash_threshold = phash_threshold
        self.ssim_threshold = ssim_threshold
        self.person_confidence = person_confidence
        self.min_persons = min_persons
        self.enable_l2 = enable_l2
        self.enable_l3 = enable_l3

        # Lazy-load YOLO model (only when L3 is needed)
        self._yolo_model = None

    @property
    def yolo_model(self):
        if self._yolo_model is None:
            print("[L3] Loading YOLOv8 model for person detection...")
            self._yolo_model = YOLO("yolov8n.pt")  # nano for speed on filtering
        return self._yolo_model

    def _detect_persons(self, frame: np.ndarray) -> list[dict]:
        """Detect persons in frame using YOLOv8. Returns list of detections."""
        results = self.yolo_model.predict(
            frame,
            classes=[0],  # class 0 = person in COCO
            conf=self.person_confidence,
            device=DEVICE,
            verbose=False,
        )

        persons = []
        if results and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                bbox_area = (x2 - x1) * (y2 - y1)
                frame_area = frame.shape[0] * frame.shape[1]
                area_ratio = bbox_area / frame_area

                persons.append({
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": conf,
                    "area_ratio": area_ratio,
                })

        return persons

    def extract_frames(
        self,
        video_meta: VideoMetadata,
        output_dir: Path = FRAMES_DIR,
        metadata_dir: Path = METADATA_DIR,
        max_frames: Optional[int] = None,
    ) -> Path:
        """
        Extract frames from video with smart 3-layer sampling.

        Returns path to the frames_index.csv metadata file.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_meta.file_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_meta.file_path}")

        original_fps = video_meta.fps
        # L1: Calculate frame interval for temporal sampling
        frame_interval = max(1, int(original_fps / self.target_fps))

        total_frames = video_meta.total_frames
        estimated_l1_frames = total_frames // frame_interval

        print(f"\n{'='*60}")
        print(f"SMART FRAME SAMPLING")
        print(f"{'='*60}")
        print(f"  Video:         {video_meta.title}")
        print(f"  Original:      {total_frames:,} frames @ {original_fps} FPS")
        print(f"  L1 Temporal:   1 frame every {frame_interval} frames "
              f"(~{self.target_fps} FPS) → ~{estimated_l1_frames:,} frames")
        print(f"  L2 Scene:      {'ON' if self.enable_l2 else 'OFF'} "
              f"(pHash>{self.phash_threshold}, SSIM<{self.ssim_threshold})")
        print(f"  L3 Person:     {'ON' if self.enable_l3 else 'OFF'} "
              f"(conf>{self.person_confidence}, min={self.min_persons})")
        print(f"{'='*60}\n")

        # CSV metadata file
        csv_path = metadata_dir / "frames_index.csv"
        csv_file = open(csv_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "frame_id", "original_frame_num", "timestamp_sec", "timestamp_hms",
            "source_video", "fps_original", "persons_detected",
            "max_person_area_ratio", "filename",
        ])

        prev_hash = None
        prev_frame = None
        saved_count = 0
        l1_count = 0
        l2_skipped = 0
        l3_skipped = 0

        pbar = tqdm(total=total_frames, desc="Extracting frames", unit="frame")

        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            pbar.update(1)

            # L1: Temporal Sampling - skip frames based on interval
            if frame_num % frame_interval != 0:
                frame_num += 1
                continue

            l1_count += 1

            # L2: Scene Change Detection
            if self.enable_l2 and prev_frame is not None:
                current_hash = _compute_phash(frame)

                hash_diff = current_hash - prev_hash
                if hash_diff < self.phash_threshold:
                    # Hashes are similar, check SSIM for confirmation
                    similarity = _compute_ssim(prev_frame, frame)
                    if similarity > self.ssim_threshold:
                        # Scene hasn't changed enough, skip
                        l2_skipped += 1
                        frame_num += 1
                        continue

                prev_hash = current_hash
            elif self.enable_l2:
                prev_hash = _compute_phash(frame)

            prev_frame = frame.copy()

            # L3: Player Presence Filter
            persons = []
            max_area_ratio = 0.0
            if self.enable_l3:
                persons = self._detect_persons(frame)
                if len(persons) < self.min_persons:
                    l3_skipped += 1
                    frame_num += 1
                    continue
                max_area_ratio = max(p["area_ratio"] for p in persons)

            # Frame passed all filters - save it
            saved_count += 1
            timestamp_sec = frame_num / original_fps
            timestamp_hms = _format_timestamp(timestamp_sec)

            filename = f"frame_{saved_count:06d}_{timestamp_hms}.jpg"
            frame_path = output_dir / filename
            cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

            csv_writer.writerow([
                saved_count, frame_num, f"{timestamp_sec:.2f}", timestamp_hms,
                video_meta.file_path.name, original_fps,
                len(persons), f"{max_area_ratio:.4f}", filename,
            ])

            if max_frames and saved_count >= max_frames:
                print(f"\n[Info] Reached max_frames limit ({max_frames})")
                break

            frame_num += 1

        pbar.close()
        cap.release()
        csv_file.close()

        print(f"\n{'='*60}")
        print(f"SAMPLING RESULTS")
        print(f"{'='*60}")
        print(f"  Total frames:     {total_frames:,}")
        print(f"  After L1 (temp):  {l1_count:,} "
              f"({l1_count/total_frames*100:.1f}%)")
        if self.enable_l2:
            after_l2 = l1_count - l2_skipped
            print(f"  After L2 (scene): {after_l2:,} "
                  f"(removed {l2_skipped:,} similar frames)")
        if self.enable_l3:
            print(f"  After L3 (person): {saved_count:,} "
                  f"(removed {l3_skipped:,} frames without players)")
        print(f"  Final saved:      {saved_count:,} frames")
        print(f"  Reduction:        {(1 - saved_count/total_frames)*100:.1f}%")
        print(f"  Saved to:         {output_dir}")
        print(f"  Index CSV:        {csv_path}")
        print(f"{'='*60}\n")

        return csv_path
