import cv2
import time
import math
import os
from collections import deque
from ultralytics import YOLO
import numpy as np

class FixedSizeQueue:
    def __init__(self, max_size):
        self.queue = deque(maxlen=max_size)
    
    def add(self, item):
        self.queue.append(item)

    def get_queue(self):
        return self.queue
    
    def __len__(self):
        return len(self.queue)

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

# === Init YOLO & Video ===
model_path = os.path.join('runs','detect','train5','weights','best.pt')
model = YOLO(model_path)

video_path = os.path.join('videos','test3.mp4')
cap = cv2.VideoCapture(video_path)

kalman = initialize_kalman_filter()
centroid_history = FixedSizeQueue(10)
ret = True
prev_frame_time = 0
paused = False
angle = 0

while ret:
    ret, frame = cap.read()
    if not ret:
        break

    # FPS calc
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time) if prev_frame_time != 0 else 0
    prev_frame_time = new_frame_time

    # YOLO Detection
    results = model.track(frame, persist=True, conf=0.35, verbose=False)
    boxes = results[0].boxes
    if boxes and len(boxes.xyxy) > 0:
        for i in range(len(boxes.cls)):
            if int(boxes.cls[i].item()) == 0:  # Class 0 = ball
                x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                measurement = np.array([[np.float32(cx)], [np.float32(cy)]])
                kalman.correct(measurement)

                centroid_history.add((cx, cy))
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                break

    # Draw trail
    trail = list(centroid_history.get_queue())
    for i in range(1, len(trail)):
        cv2.line(frame, trail[i - 1], trail[i], (255, 0, 0), 2)

    # Kalman future prediction
    if len(trail) > 0:
        pred_pt = trail[-1]
        future_positions = [pred_pt]
        prediction = kalman.predict()
        for _ in range(4):  # Predict 4 steps into future
            prediction = kalman.predict()
            pred_x, pred_y = int(prediction[0][0]), int(prediction[1][0])
            future_positions.append((pred_x, pred_y))

        for i in range(1, len(future_positions)):
            cv2.line(frame, future_positions[i - 1], future_positions[i], (0, 255, 0), 2)
            cv2.circle(frame, future_positions[i], 3, (0, 0, 255), -1)

    # Text
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    frame_resized = cv2.resize(frame, (1000, 600))
    cv2.imshow("Kalman Visual", frame_resized)

    key = cv2.waitKey(80)  # Slow it down
    if key == ord('q'):
        break
    elif key == ord(' '):
        paused = not paused
        while paused:
            key = cv2.waitKey(30) & 0xFF
            if key == ord(' '):
                paused = not paused
            elif key == ord('q'):
                paused = False
                break

cap.release()
cv2.destroyAllWindows()
