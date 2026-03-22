import cv2
import numpy as np
from ultralytics import YOLOWorld

DEFAULT_CLASSES = [
    "bench", "park bench", "tree", "fountain", "trash can", "lamp post", "bicycle rack",
]

FEATURE_COLORS = {
    # seating — Terracotta #D06224
    "bench":             (36, 98, 208),
    "park bench":        (36, 98, 208),
    "picnic table":      (36, 98, 208),
    # trees — Garden #34763D
    "bare tree":         (61, 118, 52),
    "leafless tree":     (61, 118, 52),
    "tree trunk":        (61, 118, 52),
    # lighting — Daylight #EBB64F
    "street lamp":       (79, 182, 235),
    "street light":      (79, 182, 235),
    "light pole":        (79, 182, 235),
    "ornate lamp post":  (79, 182, 235),
    # waste — Sky #94BFED
    "trash can":         (237, 191, 148),
    "litter bin":        (237, 191, 148),
    # water — Blueprint #1A56A2
    "fountain":          (162, 86, 26),
}


class FeatureDetector:
    """Detects static park features using YOLOWorld open-vocabulary detection.

    Usage:
        detector = FeatureDetector()
        detector.detect(first_frame)          # once — static camera assumption
        # per frame:
        frame = detector.render(frame)
        label, dist = detector.nearest(cx, cy)  # hook for proximity/clustering
    """

    def __init__(self, classes: list = None, conf: float = 0.1,
                 model_name: str = "yolov8x-worldv2.pt"):
        self.classes = classes or DEFAULT_CLASSES
        self.conf = conf
        self.model = YOLOWorld(model_name)
        self.model.set_classes(self.classes)
        self.detections: list = []

    def detect(self, frame: np.ndarray) -> list:
        """Run detection on frame, store and return feature locations.

        Each detection: {label, x, y, x1, y1, x2, y2, conf}
        Call once on first/reference frame for a static camera.
        """
        results = self.model.predict(frame, conf=self.conf, verbose=False)
        self.detections = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            label = self.classes[cls_id] if cls_id < len(self.classes) else "unknown"
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            self.detections.append({
                "label": label,
                "x": (x1 + x2) // 2, "y": (y1 + y2) // 2,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "conf": float(box.conf[0]),
            })
        summary = ", ".join(f"{d['label']} ({d['conf']:.2f})" for d in self.detections)
        print(f"FeatureDetector: {len(self.detections)} features — {summary or 'none'}")
        return self.detections

    def render(self, frame: np.ndarray) -> np.ndarray:
        """Draw feature markers on frame."""
        for det in self.detections:
            color = FEATURE_COLORS.get(det["label"], (200, 200, 200))
            cv2.rectangle(frame, (det["x1"], det["y1"]), (det["x2"], det["y2"]), color, 2)
            cv2.circle(frame, (det["x"], det["y"]), 6, color, -1)
            cv2.putText(frame, det["label"], (det["x1"], det["y1"] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return frame

    def nearest(self, cx: float, cy: float) -> tuple:
        """Return (label, distance_px) of nearest feature to (cx, cy).

        Returns (None, inf) if no features detected.
        Hook for per-track proximity/clustering analysis.
        """
        if not self.detections:
            return None, float("inf")
        best = min(self.detections, key=lambda d: (d["x"] - cx) ** 2 + (d["y"] - cy) ** 2)
        dist = ((best["x"] - cx) ** 2 + (best["y"] - cy) ** 2) ** 0.5
        return best["label"], dist
