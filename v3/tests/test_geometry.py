"""Unit tests for geometry helpers."""

import numpy as np
import pytest

from track_annotation.utils.geometry import (
    bbox_area,
    bbox_area_ratio,
    bbox_center,
    compute_sharpness,
    crop_with_padding,
    iou,
)


class TestIoU:
    def test_identical_boxes(self):
        box = (0.0, 0.0, 10.0, 10.0)
        assert iou(box, box) == pytest.approx(1.0)

    def test_disjoint_boxes(self):
        a = (0.0, 0.0, 10.0, 10.0)
        b = (20.0, 20.0, 30.0, 30.0)
        assert iou(a, b) == 0.0

    def test_partial_overlap(self):
        a = (0.0, 0.0, 10.0, 10.0)
        b = (5.0, 5.0, 15.0, 15.0)
        # Intersection = 5x5 = 25; Union = 100 + 100 - 25 = 175
        assert iou(a, b) == pytest.approx(25 / 175)

    def test_zero_area_box(self):
        a = (0.0, 0.0, 0.0, 0.0)
        b = (0.0, 0.0, 10.0, 10.0)
        assert iou(a, b) == 0.0


class TestBboxArea:
    def test_normal(self):
        assert bbox_area((0, 0, 10, 20)) == 200.0

    def test_zero(self):
        assert bbox_area((5, 5, 5, 5)) == 0.0

    def test_negative_clamps(self):
        # Inverted box should give zero, not negative
        assert bbox_area((10, 10, 5, 5)) == 0.0


class TestBboxAreaRatio:
    def test_full_frame(self):
        assert bbox_area_ratio((0, 0, 100, 100), (100, 100)) == pytest.approx(1.0)

    def test_quarter_frame(self):
        assert bbox_area_ratio((0, 0, 50, 50), (100, 100)) == pytest.approx(0.25)


class TestBboxCenter:
    def test_centered_origin(self):
        assert bbox_center((0, 0, 10, 10)) == (5.0, 5.0)


class TestComputeSharpness:
    def test_uniform_image_low_sharpness(self):
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        assert compute_sharpness(img) < 1.0

    def test_random_image_high_sharpness(self):
        rng = np.random.default_rng(42)
        img = rng.integers(0, 256, (100, 100, 3), dtype=np.uint8)
        assert compute_sharpness(img) > 1000.0

    def test_with_bbox(self):
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        # Sharpness within an empty (uniform) region should be ~0
        assert compute_sharpness(img, bbox=(10, 10, 50, 50)) < 1.0


class TestCropWithPadding:
    def test_basic_crop(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        crop, used = crop_with_padding(img, (40, 40, 60, 60), pad_ratio=0.0, min_size=0)
        assert crop.shape == (20, 20, 3)
        assert used == (40, 40, 60, 60)

    def test_padding_clamps_to_image(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        crop, used = crop_with_padding(img, (95, 95, 100, 100), pad_ratio=0.5, min_size=0)
        # Padding would push x2 to 102.5; should clamp to 100
        assert used[2] == 100
        assert used[3] == 100

    def test_upscale_for_min_size(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        crop, _ = crop_with_padding(img, (50, 50, 60, 60), pad_ratio=0.0, min_size=200)
        assert max(crop.shape[:2]) >= 200
