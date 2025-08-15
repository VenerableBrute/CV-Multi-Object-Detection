import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import time
import csv
import numpy as np

# Load YOLOv8
model = YOLO("yolov8n.pt")

# Class → Color mapping (BGR)
CLASS_COLORS = {
    "person": (0, 255, 0),
    "bicycle": (255, 0, 0),
    "car": (0, 0, 255),
    "motorcycle": (255, 255, 0),
    "bus": (0, 255, 255),
    "truck": (255, 0, 255)
}
DEFAULT_COLOR = (200, 200, 200)

def process_video(input_path, output_video_path="output/output_processed.mp4",
                  csv_file_path="output/tracking_log.csv", heatmap_path="output/heatmap.png"):

    tracker = DeepSort(max_age=30)
    object_data = {}

    cap = cv2.VideoCapture(input_path)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    csv_file = open(csv_file_path, mode="w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["track_id", "label", "first_seen", "last_seen", "dwell_time_sec"])

    heatmap = np.zeros((height, width), dtype=np.float32)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        detections = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = model.names[cls]
                detections.append(((float(x1), float(y1), float(x2 - x1), float(y2 - y1)), conf, label))

        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            label = track.get_det_class() if hasattr(track, "get_det_class") else "object"

            color = CLASS_COLORS.get(label.lower(), DEFAULT_COLOR)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID {track_id} {label}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            now = time.time()
            if track_id not in object_data:
                object_data[track_id] = {
                    "label": label,
                    "first_seen": now,
                    "last_seen": now
                }
            else:
                object_data[track_id]["last_seen"] = now

            heatmap[y1:y2, x1:x2] += 1

        unique_objects = len(object_data)
        class_counts = {}
        for obj in object_data.values():
            cls = obj["label"]
            class_counts[cls] = class_counts.get(cls, 0) + 1

        overlay_text = f"Unique: {unique_objects} | " + \
                       " | ".join([f"{cls}: {count}" for cls, count in class_counts.items()])
        cv2.putText(frame, overlay_text, (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        out.write(frame)

    for tid, data in object_data.items():
        dwell_time = round(data["last_seen"] - data["first_seen"], 2)
        csv_writer.writerow([tid, data["label"], data["first_seen"], data["last_seen"], dwell_time])

    csv_file.close()
    cap.release()
    out.release()

    heatmap_normalized = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_normalized = heatmap_normalized.astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
    cv2.imwrite(heatmap_path, heatmap_colored)

    return output_video_path, csv_file_path, heatmap_path
