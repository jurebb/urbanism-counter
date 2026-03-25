import pytest
from feature_detector import FeatureDetector, FEATURE_CATEGORY


def make_det(label, x1, y1, x2, y2, conf=0.8):
    return {
        "label": label,
        "category": FEATURE_CATEGORY.get(label, label),
        "x": (x1 + x2) // 2, "y": (y1 + y2) // 2,
        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        "conf": conf,
    }


class TestNMS:
    def test_no_overlap_keeps_all(self):
        dets = [
            make_det("bench", 0, 0, 50, 50),
            make_det("bench", 200, 200, 250, 250),
        ]
        result = FeatureDetector._nms(dets)
        assert len(result) == 2

    def test_high_iou_keeps_one(self):
        dets = [
            make_det("bench", 0, 0, 100, 100, conf=0.9),
            make_det("bench", 5, 5, 105, 105, conf=0.5),  # heavily overlapping
        ]
        result = FeatureDetector._nms(dets)
        assert len(result) == 1
        assert result[0]["conf"] == 0.9

    def test_different_categories_not_suppressed(self):
        dets = [
            make_det("bench", 0, 0, 100, 100),
            make_det("ornate street lamp", 0, 0, 100, 100),  # same box, different category
        ]
        result = FeatureDetector._nms(dets)
        assert len(result) == 2

    def test_containment_suppresses_inner(self):
        # small box (lamp head) fully contained inside large box (lamp column)
        outer = make_det("ornate street lamp", 0, 0, 200, 400, conf=0.9)
        inner = make_det("ornate street lamp", 80, 10, 120, 80, conf=0.6)  # centre inside outer
        result = FeatureDetector._nms([outer, inner])
        assert len(result) == 1

    def test_empty_returns_empty(self):
        assert FeatureDetector._nms([]) == []


class TestCountSummary:
    def test_counts_by_category(self):
        fd = FeatureDetector.__new__(FeatureDetector)
        fd.detections = [
            make_det("bench", 0, 0, 50, 50),
            make_det("park bench", 100, 0, 150, 50),
            make_det("ornate street lamp", 200, 0, 250, 100),
        ]
        counts = fd.count_summary()
        assert counts["Seating"] == 2
        assert counts["Lamps"] == 1

    def test_empty_detections(self):
        fd = FeatureDetector.__new__(FeatureDetector)
        fd.detections = []
        assert fd.count_summary() == {}
