import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import time
import csv
import numpy as np

# Load YOLOv8
model = YOLO("yolov8n.pt")
object_data = {}

# Class → Color mapping (BGR format)
CLASS_COLORS = {
    "person": (0, 255, 0),       # Green
    "bicycle": (255, 0, 0),      # Blue
    "car": (0, 0, 255),          # Red
    "motorcycle": (255, 255, 0), # Cyan
    "bus": (0, 255, 255),        # Yellow
    "truck": (255, 0, 255)       # Magenta
}
DEFAULT_COLOR = (200, 200, 200)  # Grey for other classes

# Initialize DeepSORT
tracker = DeepSort(max_age=30)

# Open video file
input_path = "video5.mp4"
cap = cv2.VideoCapture(input_path)

# Prepare output video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter("output_processed.mp4", fourcc, fps, (width, height))

# Prepare CSV logging
csv_file = open("tracking_log.csv", mode="w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["track_id", "label", "first_seen", "last_seen", "dwell_time_sec"])

# Initialize heatmap array
heatmap = np.zeros((height, width), dtype=np.float32)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO detection
    results = model(frame)
    detections = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]
            detections.append(((float(x1), float(y1), float(x2 - x1), float(y2 - y1)), conf, label))

    # DeepSORT tracking
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        label = track.get_det_class() if hasattr(track, "get_det_class") else "object"

        # Pick color based on class name
        color = CLASS_COLORS.get(label.lower(), DEFAULT_COLOR)

        # Draw bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID {track_id} {label}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Update object data
        now = time.time()
        if track_id not in object_data:
            object_data[track_id] = {
                "label": label,
                "first_seen": now,
                "last_seen": now
            }
        else:
            object_data[track_id]["last_seen"] = now

        # Update heatmap (add intensity for the area of the object)
        heatmap[y1:y2, x1:x2] += 1

    # Live analytics overlay
    unique_objects = len(object_data)
    class_counts = {}
    for obj in object_data.values():
        cls = obj["label"]
        class_counts[cls] = class_counts.get(cls, 0) + 1

    overlay_text = f"Unique Objects: {unique_objects} | " + \
                   " | ".join([f"{cls}: {count}" for cls, count in class_counts.items()])
    cv2.putText(frame, overlay_text, (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Write frame to output video
    out.write(frame)

    # Show video
    cv2.imshow("YOLOv8 + DeepSORT Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Save analytics to CSV
for tid, data in object_data.items():
    dwell_time = round(data["last_seen"] - data["first_seen"], 2)
    csv_writer.writerow([tid, data["label"], data["first_seen"], data["last_seen"], dwell_time])

# Generate and save heatmap image
heatmap_normalized = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
heatmap_normalized = heatmap_normalized.astype(np.uint8)
heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
cv2.imwrite("heatmap.png", heatmap_colored)

# Cleanup
csv_file.close()
cap.release()
out.release()
cv2.destroyAllWindows()

print("✅ Processing complete — output_processed.mp4, tracking_log.csv, and heatmap.png saved.")
