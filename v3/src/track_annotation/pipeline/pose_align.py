"""
Pose-aligned multi-frame fusion (ý tưởng 1).

Optional module: for ambiguous tracks where the keyframes are still too blurred,
this module aligns N frames of the same player using shoulder/hip keypoints
(homography or affine transform), then averages the aligned crops to produce a
higher-SNR representation. This is a *visualization aid* for the annotator;
labels are still applied to original frames.

Why pose-aligned average is NOT "synthesized data":
  We integrate signal of the SAME logo viewed from slightly different angles by
  warping each view to a canonical pose. Pixels are real; only coordinates are
  remapped. Equivalent in principle to astrophotography stacking.

Backend
-------
This implementation uses MediaPipe Pose for fast inference (no extra weights).
For higher accuracy, swap to RTMPose-L from mmpose; see TODO at the bottom.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from track_annotation.pipeline.detect_track import Detection, Track
from track_annotation.utils.logging import get_logger
from track_annotation.utils.video_io import VideoReader

log = get_logger(__name__)


# Lazy-import mediapipe to keep import time fast for users who don't use this.
def _get_pose_detector():
    try:
        import mediapipe as mp
    except ImportError as e:
        raise ImportError(
            "mediapipe is required for pose-aligned fusion. "
            "Install with: pip install mediapipe>=0.10.0"
        ) from e
    return mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
    )


# MediaPipe Pose landmark indices we use (subset of 33)
LM_L_SHOULDER = 11
LM_R_SHOULDER = 12
LM_L_HIP = 23
LM_R_HIP = 24


@dataclass
class AlignedFrame:
    """One aligned crop within a fused stack."""

    frame_idx: int
    aligned_crop: np.ndarray  # canonical-pose torso crop (H, W, 3) uint8


def fuse_track_aligned(
    video_path: str,
    track: Track,
    canonical_size: tuple[int, int] = (256, 256),
    max_frames: int = 20,
) -> tuple[np.ndarray | None, list[AlignedFrame]]:
    """
    Pose-align N frames of one track and return a fused (median-averaged) crop.

    Parameters
    ----------
    video_path : str
        Path to source video.
    track : Track
        Track to fuse. Frames are read from video_path at det.frame_idx.
    canonical_size : (h, w)
        Output size of canonical-pose torso crop.
    max_frames : int
        Cap on number of frames to use (uniformly sampled along track).

    Returns
    -------
    fused : np.ndarray | None
        Median-averaged canonical-pose torso crop, or None if fusion failed.
    aligned : list[AlignedFrame]
        Per-frame aligned crops used in the fusion.
    """
    if not track.detections:
        return None, []

    # Uniformly sample frames from the track
    n = min(max_frames, len(track.detections))
    if n == 0:
        return None, []
    idxs = np.linspace(0, len(track.detections) - 1, n).astype(int)
    sampled = [track.detections[i] for i in idxs]

    pose_detector = _get_pose_detector()

    aligned: list[AlignedFrame] = []
    with VideoReader(video_path) as reader:
        for det in sampled:
            frame = reader.read_at(det.frame_idx)
            if frame is None:
                continue
            crop = _align_torso(frame, det, pose_detector, canonical_size)
            if crop is not None:
                aligned.append(AlignedFrame(frame_idx=det.frame_idx, aligned_crop=crop))

    if not aligned:
        log.warning(f"track {track.track_id}: no frames could be pose-aligned")
        return None, []

    stack = np.stack([a.aligned_crop for a in aligned], axis=0).astype(np.float32)
    fused = np.median(stack, axis=0).astype(np.uint8)
    return fused, aligned


def _align_torso(
    frame: np.ndarray,
    det: Detection,
    pose_detector,
    canonical_size: tuple[int, int],
) -> np.ndarray | None:
    """
    Detect pose within the player bbox, compute affine to canonical pose,
    return warped torso crop.
    """
    x1, y1, x2, y2 = (int(round(v)) for v in det.bbox)
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    res = pose_detector.process(rgb)
    if not res.pose_landmarks:
        return None

    lms = res.pose_landmarks.landmark
    ch, cw = crop.shape[:2]

    def _xy(idx: int) -> tuple[float, float]:
        return (lms[idx].x * cw, lms[idx].y * ch)

    pts_src = np.array(
        [_xy(LM_L_SHOULDER), _xy(LM_R_SHOULDER), _xy(LM_L_HIP), _xy(LM_R_HIP)],
        dtype=np.float32,
    )

    # Canonical pose: shoulders at (1/4, 1/4) and (3/4, 1/4); hips at (1/4, 3/4) and (3/4, 3/4)
    target_h, target_w = canonical_size
    pts_dst = np.array(
        [
            [target_w * 0.25, target_h * 0.25],
            [target_w * 0.75, target_h * 0.25],
            [target_w * 0.25, target_h * 0.75],
            [target_w * 0.75, target_h * 0.75],
        ],
        dtype=np.float32,
    )

    # Use estimateAffinePartial2D (similarity transform: rotation + scale + translation)
    M, _ = cv2.estimateAffinePartial2D(pts_src, pts_dst, method=cv2.LMEDS)
    if M is None:
        return None

    warped = cv2.warpAffine(crop, M, (target_w, target_h), flags=cv2.INTER_LINEAR)
    return warped


# ============================================================
# TODO: RTMPose backend
# ============================================================
# For better accuracy on small / partially occluded players, replace MediaPipe
# with RTMPose-L from mmpose:
#
#     from mmpose.apis import MMPoseInferencer
#     inferencer = MMPoseInferencer(pose2d="rtmpose-l_8xb256-420e_coco-256x192")
#     result = next(inferencer(crop))
#     keypoints = result["predictions"][0][0]["keypoints"]  # (17, 2) COCO format
#
# COCO keypoint indices: 5=L_SHOULDER, 6=R_SHOULDER, 11=L_HIP, 12=R_HIP
