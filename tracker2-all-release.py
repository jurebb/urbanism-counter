import cv2
import math
import torch
from ultralytics import YOLO

print("\n🔍 --- SYSTEM INITIALIZING ---")
compute_device = "mps" if torch.backends.mps.is_available() else "cpu"

model = YOLO("yolov8x.pt") 

video_path = "/Users/jurebb/Movies/5min3neighbpt1.mp4"
cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), "Error reading video file"

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# ⬇️ --- NEW: Initialize the Video Writer --- ⬇️
output_path = "output_demo.mp4"
video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
# ⬆️ ---------------------------------------- ⬆️

track_history = {}  
counted_ids = set() 

# --- NEW: Class tracking setup ---
# COCO classes: 0=person, 1=bicycle, 2=car, 5=bus, 6=train(tram)
TARGET_CLASSES = [0, 1, 2, 5, 6]
CLASS_NAMES = {0: "People", 1: "Bicycles", 2: "Cars", 5: "Buses", 6: "Trams"}
class_counts = {0: 0, 1: 0, 2: 0, 5: 0, 6: 0}

# --- THE MAGIC THRESHOLDS ---
TIME_THRESHOLD = 7
DISTANCE_THRESHOLD = 10

print("🚀 Running Multi-Class Hybrid Tracking...")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # RUN TRACKING (Now looking for all 5 classes)
    results = model.track(frame, persist=True, classes=TARGET_CLASSES, tracker="bytetrack.yaml", device=compute_device, verbose=False)
    annotated_frame = results[0].plot()

    # HYBRID MATH LOGIC
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu() 
        track_ids = results[0].boxes.id.int().cpu().tolist()
        cls_ids = results[0].boxes.cls.int().cpu().tolist() # Pull the object type

        # Zip all three together
        for box, track_id, cls_id in zip(boxes, track_ids, cls_ids):
            x, y, bw, bh = box
            
            # 1. Initialize or update history (storing their class type too)
            if track_id not in track_history:
                track_history[track_id] = {'start_x': x, 'start_y': y, 'frames_seen': 1, 'class': cls_id}
            else:
                track_history[track_id]['frames_seen'] += 1
                
            # 2. Check the Gates
            if track_id not in counted_ids:
                history = track_history[track_id]
                
                # Did they survive the Time Gate?
                if history['frames_seen'] >= TIME_THRESHOLD:
                    
                    dx = x - history['start_x']
                    dy = y - history['start_y']
                    
                    # Calculate true 2D displacement
                    distance = math.sqrt(dx**2 + dy**2)
                    
                    # Did they survive the Distance Gate?
                    if distance >= DISTANCE_THRESHOLD:
                        counted_ids.add(track_id)
                        
                        # Add +1 to the specific class counter
                        final_class = history['class']
                        class_counts[final_class] += 1
                        
                        # Print to stdout with the name of the object!
                        class_name = CLASS_NAMES.get(final_class, "Unknown")
                        print(f"🎯 {class_name} (ID {track_id}) counted! (Frames: {history['frames_seen']}, Displacement: {int(distance)}px)")

    # --- THE NEW COMPACT HUD ---
    overlay = annotated_frame.copy()
    box_width = 200
    box_height = 190
    
    # Draw smaller background box
    cv2.rectangle(overlay, (w - box_width - 20, 20), (w - 20, 20 + box_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, annotated_frame, 0.3, 0, annotated_frame)
    
    # Draw the text breakdown
    y_offset = 45
    cv2.putText(annotated_frame, "--- COUNTS ---", (w - box_width - 5, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    y_offset += 25
    
    for cls_id in TARGET_CLASSES:
        name = CLASS_NAMES[cls_id]
        count = class_counts[cls_id]
        # Smaller, thinner text for the list items
        cv2.putText(annotated_frame, f"{name}: {count}", (w - box_width - 5, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 1)
        y_offset += 22
        
    # Bold Total at the bottom
    cv2.putText(annotated_frame, f"TOTAL: {len(counted_ids)}", (w - box_width - 5, y_offset + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

    # ⬇️ --- NEW: Write the frame to the MP4 file --- ⬇️
    video_writer.write(annotated_frame)
    # ⬆️ -------------------------------------------- ⬆️

    cv2.imshow("Hybrid Tracker Demo", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ⬇️ --- NEW: Release the Video Writer --- ⬇️
video_writer.release()
# ⬆️ ------------------------------------- ⬆️

cap.release()
cv2.destroyAllWindows()

print("\n📊 --- FINAL DATA --- 📊")
for cls_id in TARGET_CLASSES:
    print(f"{CLASS_NAMES[cls_id]}: {class_counts[cls_id]}")
print(f"Total Unique Objects: {len(counted_ids)}")
print(f"Video saved successfully as: {output_path}")
print("----------------------------\n")