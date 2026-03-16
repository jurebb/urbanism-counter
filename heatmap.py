import cv2
import numpy as np


class HeatmapAccumulator:
    """Accumulates track centre-point positions and renders a live heatmap overlay."""

    def __init__(self, width: int, height: int, blur_radius: int = 0, alpha: float = 0.5):
        self.width = width
        self.height = height
        self.alpha = alpha
        # blur_radius defaults to ~2% of width, always odd
        if blur_radius == 0:
            blur_radius = max(51, (width // 50) | 1)
        self.blur_radius = blur_radius if blur_radius % 2 == 1 else blur_radius + 1
        self.accumulator = np.zeros((height, width), dtype=np.float32)

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
            cx = int((track[0] + track[2]) / 2)
            cy = int((track[1] + track[3]) / 2)
            if 0 <= cx < self.width and 0 <= cy < self.height:
                self.accumulator[cy, cx] += 1

    def _to_colormap(self) -> tuple[np.ndarray, np.ndarray]:
        """Shared: blur → log scale → normalize → colormap. Returns (colored, normalized uint8)."""
        blurred = cv2.GaussianBlur(self.accumulator, (self.blur_radius, self.blur_radius), 0)
        log_scaled = np.log1p(blurred)  # compress range so new/sparse contributors stay visible
        normalized = cv2.normalize(log_scaled, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.applyColorMap(normalized, cv2.COLORMAP_JET), normalized

    def render(self, frame: np.ndarray) -> np.ndarray:
        """Return frame with heatmap blended on top."""
        if self.accumulator.max() == 0:
            return frame
        heatmap_color, normalized = self._to_colormap()
        mask = (normalized > 5).astype(np.uint8)[:, :, np.newaxis]
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
