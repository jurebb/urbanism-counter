import cv2
import numpy as np
from ultralytics import YOLOWorld

FEATURE_COLORS = {
    # seating — Daylight #EBB64F (prominent, warm yellow)
    "bench":                  (79, 182, 235),
    "park bench":             (79, 182, 235),
    "picnic table":           (79, 182, 235),
    "outdoor chair":          (79, 182, 235),
    # trees — Garden #34763D
    "tree":                   (61, 118, 52),
    "urban tree":             (61, 118, 52),
    "park tree":              (61, 118, 52),
    "street tree":            (61, 118, 52),
    "deciduous tree":         (61, 118, 52),
    # lighting — Terracotta #D06224 (demoted)
    "ornate street lamp":     (36, 98, 208),
    # waste — Sky #94BFED
    "street bin":             (237, 191, 148),
    "outdoor waste bin":      (237, 191, 148),
    # water — Blueprint #1A56A2
    "fountain":               (162, 86, 26),
}

# Maps each label to its canonical category for deduplication and counting
FEATURE_CATEGORY = {
    "bench":                "Seating",
    "park bench":           "Seating",
    "picnic table":         "Seating",
    "outdoor chair":        "Seating",
    "tree":                 "Trees",
    "urban tree":           "Trees",
    "park tree":            "Trees",
    "street tree":          "Trees",
    "deciduous tree":       "Trees",
    "ornate street lamp":   "Lamps",
    "street bin":           "Bins",
    "outdoor waste bin":    "Bins",
    "fountain":             "Fountains",
}

NMS_IOU = 0.4  # suppress overlapping boxes within the same category


class FeatureDetector:
    """Detects static park features using YOLOWorld open-vocabulary detection.

    Usage:
        detector = FeatureDetector()
        detector.detect(first_frame)          # once — static camera assumption
        # per frame:
        frame = detector.render(frame)
        label, dist = detector.nearest(cx, cy)  # hook for proximity/clustering
    """

    def __init__(self, classes: list = None, conf: float = 0.1, imgsz: int = 1280,
                 model_name: str = "yolov8x-worldv2.pt"):
        self.classes = classes or list(FEATURE_COLORS.keys())
        self.conf = conf
        self.imgsz = imgsz
        self.model = YOLOWorld(model_name)
        self.model.set_classes(self.classes)
        self.detections: list = []

    def detect(self, frame: np.ndarray) -> list:
        """Run detection on frame, store and return deduplicated feature locations.

        Each detection: {label, category, x, y, x1, y1, x2, y2, conf}
        Call once on first/reference frame for a static camera.
        """
        results = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz, iou=0.3, verbose=False)
        raw = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            label = self.classes[cls_id] if cls_id < len(self.classes) else "unknown"
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            raw.append({
                "label": label,
                "category": FEATURE_CATEGORY.get(label, label),
                "x": (x1 + x2) // 2, "y": (y1 + y2) // 2,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "conf": float(box.conf[0]),
            })

        self.detections = self._nms(raw)
        summary = ", ".join(f"{d['label']} ({d['conf']:.2f})" for d in self.detections)
        print(f"FeatureDetector: {len(self.detections)} features (after NMS) — {summary or 'none'}")
        return self.detections

    @staticmethod
    def _nms(detections: list) -> list:
        """Apply NMS + containment dedup within each semantic category."""
        if not detections:
            return []

        by_category: dict = {}
        for d in detections:
            by_category.setdefault(d["category"], []).append(d)

        kept = []
        for cat_dets in by_category.values():
            # IoU-based NMS
            boxes = [[d["x1"], d["y1"], d["x2"] - d["x1"], d["y2"] - d["y1"]] for d in cat_dets]
            confs = [d["conf"] for d in cat_dets]
            indices = cv2.dnn.NMSBoxes(boxes, confs, score_threshold=0.0, nms_threshold=NMS_IOU)
            after_nms = [cat_dets[i] for i in indices.flatten()] if len(indices) > 0 else []

            # Containment dedup: if centre of A lies inside box of B (same category),
            # keep only the higher-confidence one. Catches column/head pairs where IoU is low.
            suppressed = set()
            after_nms_sorted = sorted(after_nms, key=lambda d: d["conf"], reverse=True)
            for i, a in enumerate(after_nms_sorted):
                if i in suppressed:
                    continue
                for j, b in enumerate(after_nms_sorted):
                    if j <= i or j in suppressed:
                        continue
                    # check if b's centre is inside a's box
                    if a["x1"] <= b["x"] <= a["x2"] and a["y1"] <= b["y"] <= a["y2"]:
                        suppressed.add(j)
                    # check if a's centre is inside b's box
                    elif b["x1"] <= a["x"] <= b["x2"] and b["y1"] <= a["y"] <= b["y2"]:
                        suppressed.add(i)
                        break
            kept.extend(d for idx, d in enumerate(after_nms_sorted) if idx not in suppressed)

        return kept

    def count_summary(self) -> dict:
        """Return {category: count} from current deduplicated detections."""
        counts: dict = {}
        for d in self.detections:
            counts[d["category"]] = counts.get(d["category"], 0) + 1
        return counts

    def print_summary(self):
        counts = self.count_summary()
        print("\n📍 Feature counts (deduplicated):")
        for cat, n in sorted(counts.items()):
            print(f"   {cat}: {n}")
        print()

    def save_summary(self, path: str):
        """Save feature counts to a text file."""
        counts = self.count_summary()
        with open(path, "w") as f:
            f.write("Feature counts \n")
            f.write("=" * 30 + "\n")
            for cat, n in sorted(counts.items()):
                f.write(f"{cat}: {n}\n")
        print(f"Feature summary saved: {path}")

    def render(self, frame: np.ndarray, categories: list = None) -> np.ndarray:
        """Draw feature markers on frame. Pass categories to show only a subset (e.g. ['Seating'])."""
        for det in self.detections:
            if categories is not None and det["category"] not in categories:
                continue
            color = FEATURE_COLORS.get(det["label"], (200, 200, 200))
            cv2.rectangle(frame, (det["x1"], det["y1"]), (det["x2"], det["y2"]), color, 4)
            cv2.circle(frame, (det["x"], det["y"]), 6, color, -1)
            cv2.putText(frame, det["label"], (det["x1"], det["y1"] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return frame

    def save(self, frame: np.ndarray, path: str, categories: list = None):
        """Save two PNGs: one overlaid on frame, one on black background (like paths)."""
        cv2.imwrite(path, self.render(frame.copy(), categories=categories))
        print(f"Features saved: {path}")
        black = np.zeros_like(frame)
        black_path = path.replace(".png", "_black.png")
        cv2.imwrite(black_path, self.render(black, categories=categories))
        print(f"Features (black bg) saved: {black_path}")

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
