import os
import cv2
import math
import torch
import argparse
import numpy as np
from collections import namedtuple
from datetime import datetime
from ultralytics import YOLO
from pathlib import Path
from boxmot import DeepOcSort
from heatmap import HeatmapAccumulator
from feature_detector import FeatureDetector
from path_tracer import PathTracer

parser = argparse.ArgumentParser(description="Run YOLO tracker on a video.")
parser.add_argument("video_path", type=str, help="Path to the input video file")
parser.add_argument("--show-config", action="store_true", default=False)
parser.add_argument("--heatmap", action="store_true", default=False, help="Overlay live heatmap and save at end")
parser.add_argument("--dwell", action="store_true", default=False, help="Heatmap counts only stationary positions (dwell mode)")
parser.add_argument("--no-boxes", action="store_true", default=False, help="Hide bounding boxes and labels")
parser.add_argument("--no-counter", action="store_true", default=False, help="Hide counts overlay in top right")
parser.add_argument("--features", action="store_true", default=False, help="Detect and overlay static park features (benches, trees, etc.)")
parser.add_argument("--features-filter", nargs="+", default=None, metavar="CAT", help="Show only these feature categories e.g. --features-filter Seating Lamps")
parser.add_argument("--paths", action="store_true", default=False, help="Overlay live path trails and save density image at end")
args = parser.parse_args()

DETECTOR_MODEL   = "yolo11x.pt"
IMGSZ            = 1440
NMS_IOU          = 0.85
HEATMAP_BLUR_RADIUS = 143  # None = auto (~2% of width); set e.g. w//20 for larger blobs
FEATURE_CONF  = 0.08
FEATURE_IMGSZ = 1280
PATH_TRAIL_LENGTH = 400  # frames to keep per live trail
PATH_MAX_JUMP     = 50   # px — larger jump treated as ID theft, trail restarted
PATH_DECAY_EVERY  = 10   # frames between decay steps for inactive trails (higher = longer persistence)

# --- DeepOcSort config (best known) ---
DS_REID_MODEL       = "osnet_x1_0_msmt17.pt"
DS_CONF             = 0.15
DS_MAX_AGE          = 150
DS_MIN_HITS         = 3
DS_DELTA_T          = 1
DS_INERTIA          = 0.2
DS_W_ASSOC_EMB      = 0.9
DS_ALPHA_FIXED_EMB  = 0.95
DS_ASSO_FUNC        = "giou"


print("\n🔍 --- SYSTEM INITIALIZING ---")
compute_device = "mps" if torch.backends.mps.is_available() else "cpu"

model = YOLO(DETECTOR_MODEL)

video_path = args.video_path
cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), "Error reading video file"

w, h, fps = (
    int(cap.get(x))
    for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS)
)

CONF = DS_CONF
tracker = DeepOcSort(
    reid_weights=Path(DS_REID_MODEL),
    device=torch.device(compute_device),
    half=False,
    delta_t=DS_DELTA_T,
    inertia=DS_INERTIA,
    w_association_emb=DS_W_ASSOC_EMB,
    alpha_fixed_emb=DS_ALPHA_FIXED_EMB,
    max_age=DS_MAX_AGE,
    min_hits=DS_MIN_HITS,
    asso_func=DS_ASSO_FUNC,
)
tracker_label = "DeepOcSort"

input_name = os.path.splitext(os.path.basename(video_path))[0]
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs("output", exist_ok=True)
output_path = f"output/{input_name}_{timestamp}_deepocsort.mp4"
video_writer = cv2.VideoWriter(
    output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
)

feature_detector = FeatureDetector(conf=FEATURE_CONF, imgsz=FEATURE_IMGSZ) if args.features else None
path_tracer = PathTracer(w, h, trail_length=PATH_TRAIL_LENGTH, max_jump=PATH_MAX_JUMP, decay_every=PATH_DECAY_EVERY) if args.paths else None
first_frame_done = False

track_history = {}
counted_ids = set()
heatmap = HeatmapAccumulator(w, h, blur_radius=HEATMAP_BLUR_RADIUS, dwell_only=args.dwell) if args.heatmap else None

# COCO classes: 0=person, 1=bicycle, 2=car, 3=motorcycle, 5=bus, 6=train, 7=truck
TARGET_CLASSES = [0, 1, 2, 3, 5, 6, 7]
CLASS_NAMES = {
    0: "People",
    1: "Bicycles",
    2: "Cars",
    3: "Motorcycles",
    5: "Buses",
    6: "Trains",
    7: "Trucks",
}
class_counts = {0: 0, 1: 0, 2: 0, 3: 0, 5: 0, 6: 0, 7: 0}

TIME_THRESHOLD = 7
DISTANCE_THRESHOLD = 10

CLASS_COLORS = {
    0: (237, 191, 148),  # Sky      #94BFED — people
    1: (79, 182, 235),   # Daylight #EBB64F — bicycles
    2: (36, 98, 208),    # Terracotta #D06224 — cars
    3: (0, 46, 234),     # Redline  #EA2E00 — motorcycles
    5: (162, 86, 26),    # Blueprint #1A56A2 — buses
    6: (61, 118, 52),    # Garden   #34763D — trains
    7: (61, 118, 52),    # Garden   #34763D — trucks
}

FrameSample = namedtuple("FrameSample", ["t", "in_frame", "cumulative"])
people_over_time: list[FrameSample] = []
frame_index = 0

TRACK_CLS_COL = 6   # column index for class ID in tracks array
PERSON_CLS_ID = 0   # COCO class ID for person

def log_people_stats(tracks: np.ndarray, frame_idx: int) -> None:
    in_frame = int(np.sum(tracks[:, TRACK_CLS_COL] == PERSON_CLS_ID)) if len(tracks) > 0 else 0
    people_over_time.append(FrameSample(t=frame_idx / fps, in_frame=in_frame, cumulative=len(counted_ids)))

def save_people_plot(samples: list[FrameSample], path: str) -> None:
    import matplotlib.pyplot as plt
    BG     = "#1A56A2"   # Blueprint
    C_LINE = "#FFF8E6"   # Stucco
    C_TEXT = "#FFF8E6"

    ts   = [s.t        for s in samples]
    live = [s.in_frame for s in samples]

    fig, ax = plt.subplots(figsize=(14, 5), facecolor=BG)
    ax.set_facecolor(BG)
    ax.plot(ts, live, color=C_LINE, linewidth=1.2)
    ax.set_xlabel("Video position (s)", color=C_TEXT)
    ax.set_ylabel("People in frame", color=C_TEXT)
    ax.tick_params(colors=C_TEXT)
    ax.set_xlim(ts[0], ts[-1])
    ax.set_ylim(0)
    for spine in ax.spines.values():
        spine.set_edgecolor(C_TEXT)
    fig.suptitle(f"{input_name} — People over time", color=C_TEXT)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=BG)
    plt.close(fig)
    print(f"People plot saved: {path}")

print(f"Running {tracker_label} Tracking...")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model.predict(
        frame,
        classes=TARGET_CLASSES,
        device=compute_device,
        conf=CONF,
        imgsz=IMGSZ,
        iou=NMS_IOU,
        verbose=False,
    )

    boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()
    clss = results[0].boxes.cls.cpu().numpy()

    if len(boxes_xyxy) > 0:
        dets = np.column_stack([boxes_xyxy, confs, clss]).astype(np.float32)
    else:
        dets = np.empty((0, 6), dtype=np.float32)

    try:
        tracks = tracker.update(dets, frame)
        if tracks is None or len(tracks) == 0:
            tracks = np.empty((0, 7))
        elif tracks.ndim == 1:
            tracks = tracks.reshape(1, -1)
    except IndexError as e:
        print(f"⚠️  IndexError on frame (skipping tracks): {e}")
        tracks = np.empty((0, 7))
    # tracks: [x1, y1, x2, y2, track_id, conf, cls, ...]

    log_people_stats(tracks, frame_index)
    frame_index += 1

    if heatmap is not None and len(tracks) > 0:
        heatmap.update(tracks, class_filter=[0])  # people only

    if path_tracer is not None and len(tracks) > 0:
        path_tracer.update(tracks, class_filter=[0])  # people only

    annotated_frame = heatmap.render(frame.copy()) if heatmap is not None else frame.copy()

    if path_tracer is not None:
        annotated_frame = path_tracer.render(annotated_frame)

    if feature_detector is not None:
        if not first_frame_done:
            feature_detector.detect(frame)
            feature_detector.print_summary()
            feature_detector.save(frame, f"output/{input_name}_{timestamp}_features.png", categories=args.features_filter)
            feature_detector.save_summary(f"output/{input_name}_{timestamp}_features.txt")
            first_frame_done = True
        annotated_frame = feature_detector.render(annotated_frame, categories=args.features_filter)

    if len(tracks) > 0:
        for track in tracks:
            x1, y1, x2, y2, track_id, conf, cls_id = (
                int(track[0]), int(track[1]), int(track[2]), int(track[3]),
                int(track[4]), float(track[5]), int(track[6]),
            )
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            if not args.no_boxes:
                color = CLASS_COLORS.get(cls_id, (200, 200, 200))
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                label = f"id:{track_id} {CLASS_NAMES.get(cls_id, '?')} {conf:.2f}"
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated_frame, (x1, y1 - lh - 10), (x1 + lw + 4, y1), color, -1)
                cv2.putText(annotated_frame, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if track_id not in track_history:
                track_history[track_id] = {
                    "start_x": cx,
                    "start_y": cy,
                    "frames_seen": 1,
                    "class": cls_id,
                }
            else:
                track_history[track_id]["frames_seen"] += 1

            if track_id not in counted_ids:
                history = track_history[track_id]
                if history["frames_seen"] >= TIME_THRESHOLD:
                    dx = cx - history["start_x"]
                    dy = cy - history["start_y"]
                    distance = math.sqrt(dx**2 + dy**2)
                    if distance >= DISTANCE_THRESHOLD:
                        counted_ids.add(track_id)
                        final_class = history["class"]
                        class_counts[final_class] += 1
                        class_name = CLASS_NAMES.get(final_class, "Unknown")
                        print(
                            f"🎯 {class_name} (ID {track_id}) counted! (Frames: {history['frames_seen']}, Displacement: {int(distance)}px)"
                        )

    if not args.no_counter:
        feature_counts = feature_detector.count_summary() if feature_detector is not None else {}
        feature_lines = [f"{cat}: {n}" for cat, n in sorted(feature_counts.items())]

        FS = 1.15
        LINE_H = 42
        PAD_X = w - 395
        font = cv2.FONT_HERSHEY_SIMPLEX

        MARGIN = 16
        counts_rows = 1 + len(TARGET_CLASSES) + 1  # header + classes + TOTAL
        counts_height = MARGIN + counts_rows * LINE_H + MARGIN
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (PAD_X - 15, 10), (w - 10, 10 + counts_height), (162, 86, 26), -1)
        cv2.addWeighted(overlay, 0.7, annotated_frame, 0.3, 0, annotated_frame)

        y = 10 + MARGIN + LINE_H - 6
        cv2.putText(annotated_frame, "> TRACKER:", (PAD_X, y), font, FS, (255, 255, 255), 2)
        y += LINE_H
        for cls_id in TARGET_CLASSES:
            cv2.putText(annotated_frame, f"{CLASS_NAMES[cls_id]}: {class_counts[cls_id]}",
                        (PAD_X, y), font, FS, (230, 248, 255), 1)
            y += LINE_H
        cv2.putText(annotated_frame, f"TOTAL: {len(counted_ids)}", (PAD_X, y), font, 1.25, (255, 255, 255), 2)

        if feature_lines:
            feat_top = 10 + counts_height + 10
            feat_rows = 1 + len(feature_lines)  # header + lines
            feat_height = MARGIN + feat_rows * LINE_H + MARGIN
            overlay2 = annotated_frame.copy()
            cv2.rectangle(overlay2, (PAD_X - 15, feat_top), (w - 10, feat_top + feat_height), (162, 86, 26), -1)
            cv2.addWeighted(overlay2, 0.7, annotated_frame, 0.3, 0, annotated_frame)

            y = feat_top + MARGIN + LINE_H - 6
            cv2.putText(annotated_frame, "> FEATURES:", (PAD_X, y), font, FS, (255, 255, 255), 2)
            y += LINE_H
            for line in feature_lines:
                cv2.putText(annotated_frame, line, (PAD_X, y), font, FS, (230, 248, 255), 1)
                y += LINE_H

    if args.show_config:
        config_lines = [
            f"tracker:     {tracker_label}",
            f"detector:    {DETECTOR_MODEL}",
            f"reid:        {DS_REID_MODEL}",
            f"asso_func:   {DS_ASSO_FUNC}",
            f"conf:        {DS_CONF}",
            f"imgsz:       {IMGSZ}",
            f"nms_iou:     {NMS_IOU}",
            f"max_age:     {DS_MAX_AGE}",
            f"min_hits:    {DS_MIN_HITS}",
            f"delta_t:     {DS_DELTA_T}",
            f"inertia:     {DS_INERTIA}",
            f"w_emb:       {DS_W_ASSOC_EMB}",
            f"alpha_emb:   {DS_ALPHA_FIXED_EMB}",
            f"heatmap:     {'dwell' if args.dwell else 'density' if args.heatmap else 'off'}",
            f"paths:       {'on' if args.paths else 'off'} trail={PATH_TRAIL_LENGTH} jump={PATH_MAX_JUMP} decay={PATH_DECAY_EVERY}",
            f"features:    {'on' if args.features else 'off'} conf={FEATURE_CONF}",
        ]
        font, fscale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        pad, line_h = 6, 18
        box_w = max(cv2.getTextSize(l, font, fscale, thick)[0][0] for l in config_lines) + pad * 2
        box_h = line_h * len(config_lines) + pad * 2
        cfg_overlay = annotated_frame.copy()
        cv2.rectangle(cfg_overlay, (10, 10), (10 + box_w, 10 + box_h), (30, 30, 30), -1)
        cv2.addWeighted(cfg_overlay, 0.7, annotated_frame, 0.3, 0, annotated_frame)
        for i, line in enumerate(config_lines):
            cv2.putText(annotated_frame, line, (10 + pad, 10 + pad + line_h * (i + 1) - 4),
                        font, fscale, (220, 220, 220), thick)

    video_writer.write(annotated_frame)
    cv2.imshow(f"{tracker_label} Tracker", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_writer.release()
cap.release()
cv2.destroyAllWindows()

if heatmap is not None:
    heatmap.save(f"output/{input_name}_{timestamp}_heatmap.png")

if path_tracer is not None:
    path_tracer.save(f"output/{input_name}_{timestamp}_paths.png")

if people_over_time:
    save_people_plot(people_over_time, f"output/{input_name}_{timestamp}_people.png")

print("\n📊 --- FINAL DATA --- 📊")
for cls_id in TARGET_CLASSES:
    print(f"{CLASS_NAMES[cls_id]}: {class_counts[cls_id]}")
print(f"Total Unique Objects: {len(counted_ids)}")
print(f"Video saved successfully as: {output_path}")
print("----------------------------\n")
