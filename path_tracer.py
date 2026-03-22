import cv2
import numpy as np
from collections import defaultdict


class PathTracer:
    """Traces movement paths per tracked person.

    Two outputs:
    - Live trail overlay: last N positions per active track, fading opacity
    - Accumulated density image: all path segments binned into a grid,
      thicker/brighter where more people walked (desire lines)

    Usage:
        tracer = PathTracer(w, h)
        # per frame:
        tracer.update(tracks, class_filter=[0])
        frame = tracer.render(frame)
        # at end:
        tracer.save("output/paths.png")
    """

    TRAIL_TAIL   = (230, 248, 255)   # Stucco  #FFF8E6 — trail tail (old)
    TRAIL_HEAD   = (79,  182, 235)   # Daylight #EBB64F — trail head (recent)
    DENSITY_CMAP = cv2.COLORMAP_HOT

    MIN_HOLD_FRAMES = 600  # frames an inactive trail is frozen before decay starts

    def __init__(self, width: int, height: int, trail_length: int = 60,
                 max_jump: int = 120, decay_every: int = 10, alpha: float = 0.6, grid_scale: float = 0.5):
        self.width = width
        self.height = height
        self.trail_length = trail_length  # frames to keep per live trail
        self.max_jump = max_jump          # px threshold — jump larger than this restarts trail
        self.decay_every = decay_every    # frames between decay steps for inactive trails
        self.alpha = alpha                # blend strength for density overlay

        # grid_scale: downsample factor for density accumulator (0.5 = half res)
        self.gw = int(width * grid_scale)
        self.gh = int(height * grid_scale)
        self.grid_scale = grid_scale
        self.density = np.zeros((self.gh, self.gw), dtype=np.float32)

        self._trails: dict = defaultdict(list)  # track_id -> [(x, y), ...]
        self._inactive_since: dict = {}          # tid -> frame when it went inactive
        self._frame_count = 0

    def update(self, tracks: np.ndarray, class_filter: list = None):
        """Record current frame positions and decay inactive trails. Call once per frame."""
        self._frame_count += 1
        active_ids = set()

        for track in tracks:
            cls_id = int(track[6])
            if class_filter is not None and cls_id not in class_filter:
                continue
            track_id = int(track[4])
            x1, y1, x2, y2 = int(track[0]), int(track[1]), int(track[2]), int(track[3])
            cx = (x1 + x2) // 2
            cy = y1 + (y2 - y1) // 8  # near head, stable during occlusion

            trail = self._trails[track_id]
            active_ids.add(track_id)

            # inertia: large jumps slide toward new position instead of cutting
            if trail:
                lx, ly = trail[-1]
                dx, dy = cx - lx, cy - ly
                dist = (dx * dx + dy * dy) ** 0.5
                if dist > self.max_jump:
                    t = self.max_jump / dist  # fraction: bigger jump = smaller step
                    cx = int(lx + dx * t)
                    cy = int(ly + dy * t)

            trail.append((cx, cy))
            if len(trail) > self.trail_length:
                trail.pop(0)

            # accumulate into density grid
            gx = int(cx * self.grid_scale)
            gy = int(cy * self.grid_scale)
            if 0 <= gx < self.gw and 0 <= gy < self.gh:
                self.density[gy, gx] += 1

        # decay inactive trails: hold for MIN_HOLD_FRAMES, then decay
        if self._frame_count % self.decay_every == 0:
            for tid in list(self._trails.keys()):
                if tid not in active_ids:
                    self._inactive_since.setdefault(tid, self._frame_count)
                    if self._frame_count - self._inactive_since[tid] >= self.MIN_HOLD_FRAMES:
                        trail = self._trails[tid]
                        if trail:
                            trail.pop(0)
                        else:
                            del self._trails[tid]
                            del self._inactive_since[tid]
                else:
                    self._inactive_since.pop(tid, None)

    def render(self, frame: np.ndarray) -> np.ndarray:
        """Draw live fading trails. Inactive trails decay naturally via update()."""
        overlay = frame.copy()
        for tid, trail in self._trails.items():
            if len(trail) < 2:
                continue
            for i in range(1, len(trail)):
                t = i / len(trail)  # 0→1, tail→head
                color = tuple(int(self.TRAIL_TAIL[c] + t * (self.TRAIL_HEAD[c] - self.TRAIL_TAIL[c])) for c in range(3))
                thickness = max(2, int(0.3 + t * 7))
                cv2.line(overlay, trail[i - 1], trail[i], color, thickness, cv2.LINE_AA)
        cv2.addWeighted(overlay, self.alpha, frame, 1 - self.alpha, 0, frame)
        return frame

    def _density_to_image(self) -> np.ndarray:
        """Shared: blur → log scale → normalize → colormap → full res."""
        blur_r = max(5, (self.gw // 30) | 1)
        blurred = cv2.GaussianBlur(self.density, (blur_r, blur_r), 0)
        log_scaled = np.log1p(blurred)
        normalized = cv2.normalize(log_scaled, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        colored = cv2.applyColorMap(normalized, self.DENSITY_CMAP)
        return cv2.resize(colored, (self.width, self.height), interpolation=cv2.INTER_LINEAR), \
               cv2.resize(normalized, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

    def render_density(self, frame: np.ndarray) -> np.ndarray:
        """Blend accumulated density overlay onto frame (optional, call instead of render)."""
        if self.density.max() == 0:
            return frame
        colored, normalized = self._density_to_image()
        mask = (normalized > 5).astype(np.uint8)[:, :, np.newaxis]
        blended = cv2.addWeighted(frame, 1 - self.alpha, colored, self.alpha, 0)
        return np.where(mask, blended, frame)

    def save(self, path: str):
        """Save standalone accumulated density image (desire lines map)."""
        if self.density.max() == 0:
            return
        colored, _ = self._density_to_image()
        cv2.imwrite(path, colored)
        print(f"Path density saved: {path}")
