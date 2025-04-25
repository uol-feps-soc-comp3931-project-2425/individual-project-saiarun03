import os
import cv2
import json
import numpy as np
from ultralytics import YOLO

# Load YOLO model
model_path = os.path.join('runs', 'detect', 'train5', 'weights', 'best.pt')
model = YOLO(model_path)

# Set up video and output dir
video_path = os.path.join('videos', 'test2.mp4')
os.makedirs("predictionsv2", exist_ok=True)

# Kalman Filter setup
def initialize_kalman_filter():
    kalman = cv2.KalmanFilter(4, 2)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                         [0, 1, 0, 0]], np.float32)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                        [0, 1, 0, 1],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]], np.float32)
    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    return kalman

kalman = initialize_kalman_filter()
cap = cv2.VideoCapture(video_path)
frame_id = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    results = model(frame)
    boxes = results[0].boxes

    ball_detected = False
    cx, cy = None, None

    if boxes and len(boxes.xyxy) > 0:
        for i in range(len(boxes.cls)):
            if int(boxes.cls[i].item()) == 0:
                x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                ball_detected = True
                break

    if not ball_detected:
        with open(os.path.join("predictionsv2", f"{frame_id:04d}_KF.json"), "w") as f:
            json.dump({}, f, indent=4)
        continue

    measurement = np.array([[np.float32(cx)], [np.float32(cy)]])
    kalman.correct(measurement)
    prediction = kalman.predict()
    pred_x, pred_y = int(prediction[0][0]), int(prediction[1][0])

    with open(os.path.join("predictionsv2", f"{frame_id:04d}_KF.json"), "w") as f:
        json.dump({"center_x": pred_x, "center_y": pred_y}, f, indent=4)

cap.release()
cv2.destroyAllWindows()
