import numpy as np
import pytest
from heatmap import HeatmapAccumulator, _studio_leonardo_cmap


def make_track(x1, y1, x2, y2, track_id=1, conf=0.9, cls=0):
    return np.array([x1, y1, x2, y2, track_id, conf, cls], dtype=np.float32)


class TestHeatmapAccumulator:
    def test_accumulator_starts_empty(self):
        h = HeatmapAccumulator(100, 100)
        assert h.accumulator.max() == 0

    def test_update_increments_accumulator(self):
        h = HeatmapAccumulator(100, 100)
        tracks = np.array([make_track(10, 10, 30, 50)])
        h.update(tracks)
        assert h.accumulator.max() > 0

    def test_update_class_filter_excludes(self):
        h = HeatmapAccumulator(100, 100)
        car_track = np.array([make_track(10, 10, 30, 50, cls=2)])
        h.update(car_track, class_filter=[0])  # only people
        assert h.accumulator.max() == 0

    def test_update_class_filter_includes(self):
        h = HeatmapAccumulator(100, 100)
        person_track = np.array([make_track(10, 10, 30, 50, cls=0)])
        h.update(person_track, class_filter=[0])
        assert h.accumulator.max() > 0

    def test_render_returns_same_shape(self):
        h = HeatmapAccumulator(100, 80)
        tracks = np.array([make_track(10, 10, 30, 50)])
        h.update(tracks)
        frame = np.zeros((80, 100, 3), dtype=np.uint8)
        out = h.render(frame)
        assert out.shape == frame.shape

    def test_render_empty_returns_frame_unchanged(self):
        h = HeatmapAccumulator(100, 80)
        frame = np.zeros((80, 100, 3), dtype=np.uint8)
        frame[10, 10] = [42, 42, 42]
        out = h.render(frame.copy())
        assert out[10, 10, 0] == 42

    def test_dwell_only_skips_moving_tracks(self):
        h = HeatmapAccumulator(200, 200, dwell_only=True, dwell_threshold=5.0)
        # same track moves 50px per frame — should be skipped
        for i in range(20):
            t = np.array([make_track(i * 50, 10, i * 50 + 20, 40, track_id=1)])
            h.update(t)
        assert h.accumulator.max() == 0


class TestStudioLeonardoCmap:
    def test_shape(self):
        lut = _studio_leonardo_cmap()
        assert lut.shape == (256, 1, 3)

    def test_dtype(self):
        lut = _studio_leonardo_cmap()
        assert lut.dtype == np.uint8

    def test_no_black_at_low_end(self):
        lut = _studio_leonardo_cmap()
        # first entry should not be pure black (we start at dark Blueprint)
        assert lut[0, 0].sum() > 0
