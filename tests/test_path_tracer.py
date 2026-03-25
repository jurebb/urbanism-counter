import numpy as np
import pytest
from path_tracer import PathTracer


def make_track(x1, y1, x2, y2, track_id=1, cls=0):
    return np.array([x1, y1, x2, y2, track_id, 0.9, cls], dtype=np.float32)


class TestSmooth:
    def test_short_trail_returned_unchanged(self):
        trail = [(0, 0), (1, 1)]
        assert PathTracer._smooth(trail, window=9) == trail

    def test_smoothed_length_matches_input(self):
        trail = [(i, i) for i in range(20)]
        result = PathTracer._smooth(trail, window=5)
        assert len(result) == len(trail)

    def test_none_break_preserved(self):
        trail = [(0, 0), (1, 1), (2, 2), None, (50, 50), (51, 51), (52, 52)]
        result = PathTracer._smooth(trail, window=3)
        assert None in result

    def test_none_breaks_dont_cross_segments(self):
        # points before and after None should stay on their respective sides
        trail = [(0, 0), (1, 0), (2, 0), None, (100, 0), (101, 0), (102, 0)]
        result = PathTracer._smooth(trail, window=3)
        none_idx = result.index(None)
        before = [p for p in result[:none_idx] if p is not None]
        after  = [p for p in result[none_idx+1:] if p is not None]
        assert all(p[0] < 50 for p in before)
        assert all(p[0] > 50 for p in after)

    def test_no_zero_pull_at_edges(self):
        # edge values should stay near first/last point, not pulled toward 0
        trail = [(100, 100)] * 15
        result = PathTracer._smooth(trail, window=9)
        assert result[0][0] > 50
        assert result[-1][0] > 50


class TestUpdate:
    def test_large_jump_inserts_none(self):
        pt = PathTracer(500, 500, max_jump=50)
        t1 = np.array([make_track(100, 100, 120, 140, track_id=1)])
        t2 = np.array([make_track(400, 400, 420, 440, track_id=1)])  # far jump
        pt.update(t1)
        pt.update(t2)
        assert None in pt._trails[1]

    def test_small_step_no_break(self):
        pt = PathTracer(500, 500, max_jump=50)
        t1 = np.array([make_track(100, 100, 120, 140, track_id=1)])
        t2 = np.array([make_track(110, 100, 130, 140, track_id=1)])  # small step
        pt.update(t1)
        pt.update(t2)
        assert None not in pt._trails[1]

    def test_trail_length_counts_only_real_points(self):
        pt = PathTracer(500, 500, max_jump=50, trail_length=5)
        # add 10 normal steps
        for i in range(10):
            t = np.array([make_track(i * 3, 100, i * 3 + 20, 140, track_id=1)])
            pt.update(t)
        real = sum(1 for p in pt._trails[1] if p is not None)
        assert real <= 5


class TestRender:
    def test_render_returns_same_shape(self):
        pt = PathTracer(200, 200)
        tracks = np.array([make_track(50, 50, 70, 90, track_id=1)])
        for _ in range(5):
            pt.update(tracks)
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        out = pt.render(frame)
        assert out.shape == (200, 200, 3)
