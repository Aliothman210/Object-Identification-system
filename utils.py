# utils.py
from ultralytics import YOLO
import cv2
import numpy as np
import time
import threading
import warnings
warnings.filterwarnings("ignore")

# ------------------ Config ------------------ #

allowed_classes = [
    "person", "car", "dog", "cat", "bicycle", "motorcycle", "bus",
    "truck", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bird", "horse", "sheep", "cow"
]

# ------------------ Runtime State ------------------ #

# Lock to prevent race conditions when multiple threads update stats
_stats_lock = threading.Lock()

# Shared dictionary that holds the latest statistics sent to the /stats endpoint
_latest_stats = {
    "fps": 0.0,                      # Current frames per second
    "counts": {},                   # Object counts per class label
    "last_update_ts": 0.0,          # Timestamp of the last update
    "alert": {"message": "", "ts": 0.0}  # Alert message with timestamp
}

def _update_latest_stats(fps, counts, alert_message=None):
    """
    Update the FPS, object counts, and timestamp.
    Optionally updates the alert message if provided.
    """
    with _stats_lock:
        _latest_stats["fps"] = float(fps)
        _latest_stats["counts"] = dict(counts)
        _latest_stats["last_update_ts"] = time.time()

        if alert_message:
            _latest_stats["alert"] = {
                "message": alert_message,
                "ts": time.time()
            }

def _set_alert(message):
    """
    Set an alert message with the current timestamp.
    Used if only the alert needs to be updated.
    """
    with _stats_lock:
        _latest_stats["alert"] = {
            "message": message,
            "ts": time.time()
        }

def get_latest_stats():
    """
    Return a copy of the latest stats for the /stats endpoint.
    Ensures the frontend can't modify the shared state directly.
    """
    with _stats_lock:
        return {
            "fps": _latest_stats.get("fps", 0.0),
            "counts": dict(_latest_stats.get("counts", {})),
            "last_update_ts": _latest_stats.get("last_update_ts", 0.0),
            "alert": dict(_latest_stats.get("alert", {"message": "", "ts": 0.0}))
        }


# ------------------ Functions ------------------ #

# Load YOLO model from a .pt file
def load_model(model_path="yolov8n.pt"):
    return YOLO(model_path)

# Check if detected class is in the allowed classes list
def is_allowed_class(model, cls_id):
    cls_name = model.names[int(cls_id)]
    return cls_name in allowed_classes

# Draw bounding boxes and labels on allowed detections
def draw_boxes(frame, results, model):
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]

            if not is_allowed_class(model, cls_id):
                continue

            # get box coordinates [x1, y1, x2, y2]
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            label = f"{cls_name} {conf:.2f}"

            cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
            cv2.putText(frame, label, (xyxy[0], xyxy[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return frame

# Generator function to capture video frames, perform detection, and yield annotated frames
def generate_frames(model, source=0, scale=1):
    # open video source (camera or IP stream)
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # limit stored frames to reduce delay
    cap.set(cv2.CAP_PROP_FPS, 15)        # target FPS

    frame_count = 0  # counter to skip frames for performance
    last_results = None  # store latest detections
    smoothed_fps = 0.0  # average FPS value

    # for controlling alert frequency
    last_alert_time = 0
    alert_cooldown = 3  # seconds

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # stop if no frame is read

        frame_count += 1

        # run detection every 6 frames
        if frame_count % 6 == 0:
            t0 = time.time()
            results = model(frame, verbose=False, imgsz=480)  # run YOLO
            infer_dt = time.time() - t0

            # smooth out FPS spikes
            instant_fps = 1.0 / max(infer_dt, 1e-6)
            smoothed_fps = instant_fps if smoothed_fps == 0 else (
                0.9 * smoothed_fps + 0.1 * instant_fps
            )
            last_results = results

            # count allowed objects
            counts = {}
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    if not is_allowed_class(model, cls_id):
                        continue
                    cls_name = model.names[cls_id]
                    counts[cls_name] = counts.get(cls_name, 0) + 1

            if counts:
                current_time = time.time()
                # take most frequent detected class
                detected_class = max(counts.items(), key=lambda x: x[1])[0]
                count = counts[detected_class]
                print(f"Detected {count} {detected_class}(s)")

                # create alert only if cooldown passed
                alert_message = (
                    f"{count} {detected_class} detected"
                    if (current_time - last_alert_time) > alert_cooldown
                    else None
                )
                if alert_message:
                    last_alert_time = current_time
            else:
                alert_message = None

            _update_latest_stats(smoothed_fps, counts, alert_message)

        # draw boxes if we have detection results
        if last_results is not None:
            frame = draw_boxes(frame, last_results, model)

        # resize frame before sending
        frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

        # encode frame as JPEG for browser streaming
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # send frame chunk in multipart response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()  # release camera/stream when done

