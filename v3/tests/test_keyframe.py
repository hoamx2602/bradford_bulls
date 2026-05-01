"""Unit tests for keyframe selection."""

import pytest

from track_annotation.config import KeyframeConfig
from track_annotation.pipeline.detect_track import Detection, Track
from track_annotation.pipeline.keyframe import select_keyframes


def _det(frame_idx: int, sharpness: float, area: float = 0.01) -> Detection:
    return Detection(
        frame_idx=frame_idx,
        timestamp_s=frame_idx / 30.0,
        bbox=(0.0, 0.0, 100.0, 100.0),
        confidence=0.9,
        class_id=0,
        sharpness=sharpness,
        area_ratio=area,
    )


class TestSelectKeyframes:
    def test_empty_track(self):
        cfg = KeyframeConfig()
        kfs = select_keyframes(Track(track_id=1, detections=[]), cfg)
        assert kfs == []

    def test_picks_sharpest(self):
        track = Track(
            track_id=1,
            detections=[
                _det(0, sharpness=10.0),
                _det(10, sharpness=100.0),
                _det(20, sharpness=50.0),
            ],
        )
        cfg = KeyframeConfig(num_per_track=1, strategies=["sharpest"])
        kfs = select_keyframes(track, cfg)
        assert len(kfs) == 1
        assert kfs[0].detection.frame_idx == 10
        assert kfs[0].strategy == "sharpest"

    def test_picks_largest(self):
        track = Track(
            track_id=1,
            detections=[
                _det(0, sharpness=50, area=0.01),
                _det(10, sharpness=50, area=0.05),
                _det(20, sharpness=50, area=0.02),
            ],
        )
        cfg = KeyframeConfig(num_per_track=1, strategies=["largest"])
        kfs = select_keyframes(track, cfg)
        assert kfs[0].detection.frame_idx == 10

    def test_midpoint(self):
        track = Track(
            track_id=1,
            detections=[_det(i, sharpness=10) for i in range(0, 21)],
        )
        cfg = KeyframeConfig(num_per_track=1, strategies=["midpoint"])
        kfs = select_keyframes(track, cfg)
        # 21 detections; midpoint index = 10
        assert kfs[0].detection.frame_idx == 10

    def test_dedupe_when_strategies_collide(self):
        # If sharpest, largest, midpoint all point to the same frame, only one keyframe
        track = Track(track_id=1, detections=[_det(5, sharpness=100, area=0.5)])
        cfg = KeyframeConfig(
            num_per_track=3, strategies=["sharpest", "largest", "midpoint"]
        )
        kfs = select_keyframes(track, cfg)
        assert len(kfs) == 1
        assert kfs[0].strategy == "sharpest"  # first strategy wins

    def test_unknown_strategy_raises(self):
        track = Track(track_id=1, detections=[_det(0, sharpness=1)])
        cfg = KeyframeConfig(num_per_track=1, strategies=["sharpest"])
        cfg.strategies = ["bogus"]  # bypass pydantic
        with pytest.raises(ValueError, match="Unknown keyframe strategy"):
            select_keyframes(track, cfg)
