import math
import cv2
import numpy as np
from collections import deque


def _studio_leonardo_cmap() -> np.ndarray:
    """Multi-stop colormap through Studio Leonardo palette (BGR), no Steel.

    Stops: black → Blueprint → Sky → Daylight → Terracotta → Redline
    """
    # (B, G, R)
    stops = [
        (0,   (162, 86,  26 )),   # Blueprint  #1A56A2 — low density
        (0.40,(237, 191, 148)),   # Sky        #94BFED
        (0.62,(79,  182, 235)),   # Daylight   #EBB64F
        (0.80,(36,  98,  208)),   # Terracotta #D06224
        (1.0, (0,   46,  234)),   # Redline    #EA2E00
    ]
    lut = np.zeros((256, 1, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        # find the two surrounding stops
        for k in range(len(stops) - 1):
            t0, c0 = stops[k]
            t1, c1 = stops[k + 1]
            if t0 <= t <= t1:
                alpha = (t - t0) / (t1 - t0)
                lut[i, 0] = tuple(int(c0[c] + alpha * (c1[c] - c0[c])) for c in range(3))
                break
    return lut


STUDIO_CMAP = _studio_leonardo_cmap()


class HeatmapAccumulator:
    """Accumulates track centre-point positions and renders a live heatmap overlay."""

    def __init__(self, width: int, height: int, blur_radius: int = None, alpha: float = 0.72,
                 dwell_only: bool = False, dwell_threshold: float = 6.0):
        self.width = width
        self.height = height
        self.alpha = alpha
        self.dwell_only = dwell_only
        self.dwell_threshold = dwell_threshold  # max total px displacement (first→last over window) to count as stationary
        self._ROLLING = 15  # window length for displacement check
        if blur_radius is None:
            blur_radius = max(51, (width // 50) | 1)  # default: ~2% of width
        self.blur_radius = blur_radius if blur_radius % 2 == 1 else blur_radius + 1
        self.accumulator = np.zeros((height, width), dtype=np.float32)
        self._position_history: dict = {}  # track_id -> deque of (cx, cy)

    def update(self, tracks, class_filter: list = None):
        """Add current frame's track centres to the accumulator.

        Args:
            tracks: numpy array [x1, y1, x2, y2, track_id, conf, cls, ...]
            class_filter: list of COCO class ids to include, or None for all
        """
        for track in tracks:
            cls_id = int(track[6])
            if class_filter is not None and cls_id not in class_filter:
                continue
            track_id = int(track[4])
            cx = int((track[0] + track[2]) / 2)
            cy = int((track[1] + track[3]) / 2)

            if self.dwell_only:
                hist = self._position_history.setdefault(track_id, deque(maxlen=self._ROLLING + 1))
                hist.append((cx, cy))
                if len(hist) < self._ROLLING:
                    continue  # wait for full window
                # total displacement first→last; catches slow walkers that average below per-frame threshold
                dx = hist[-1][0] - hist[0][0]
                dy = hist[-1][1] - hist[0][1]
                if math.sqrt(dx*dx + dy*dy) > self.dwell_threshold:
                    continue  # moving — skip

            if 0 <= cx < self.width and 0 <= cy < self.height:
                self.accumulator[cy, cx] += 1

    def _to_colormap(self) -> tuple[np.ndarray, np.ndarray]:
        """Shared: blur → log scale → normalize → colormap. Returns (colored, normalized uint8)."""
        blurred = cv2.GaussianBlur(self.accumulator, (self.blur_radius, self.blur_radius), 0)
        log_scaled = np.log1p(blurred)  # compress range so new/sparse contributors stay visible
        normalized = cv2.normalize(log_scaled, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.applyColorMap(normalized, STUDIO_CMAP), normalized

    def render(self, frame: np.ndarray) -> np.ndarray:
        """Return frame with heatmap blended on top."""
        if self.accumulator.max() == 0:
            return frame
        heatmap_color, normalized = self._to_colormap()
        raw_mask = (normalized > 5).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)[:, :, np.newaxis]
        blended = cv2.addWeighted(frame, 1 - self.alpha, heatmap_color, self.alpha, 0)
        return np.where(mask, blended, frame)

    def save(self, path: str):
        """Save a standalone heatmap image (no video frame underneath)."""
        if self.accumulator.max() == 0:
            return
        heatmap_color, _ = self._to_colormap()
        cv2.imwrite(path, heatmap_color)
        print(f"Heatmap saved: {path}")

    def save_overlay(self, frame: np.ndarray, path: str):
        """Save heatmap blended onto a given frame (e.g. last frame of the video)."""
        cv2.imwrite(path, self.render(frame))
        print(f"Heatmap overlay saved: {path}")
