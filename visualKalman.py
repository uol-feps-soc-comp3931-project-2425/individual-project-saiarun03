import os
import time
import math
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO

class FixedSizeQueue:
    """A fixed-length queue for storing recent items."""
    def __init__(self, max_size):
        self.queue = deque(maxlen=max_size)
    
    def add(self, item):
        self.queue.append(item)

    def get_queue(self):
        return self.queue
    
    def __len__(self):
        return len(self.queue)

def initialize_kalman_filter():
    """Configure a 4D Kalman filter (x, y, dx, dy)."""
    kalman = cv2.KalmanFilter(4, 2)
    # measurement: x, y
    kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                         [0, 1, 0, 0]], np.float32)
    # state transition: x += dx, y += dy
    kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                        [0, 1, 0, 1],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]], np.float32)
    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    return kalman

# Initialize model, video, and filter
model = YOLO(os.path.join('runs', 'detect', 'train5', 'weights', 'best.pt'))
cap = cv2.VideoCapture(os.path.join('videos', 'test2.mp4'))
kalman = initialize_kalman_filter()
centroid_history = FixedSizeQueue(10)

prev_time = 0
paused = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Compute FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time

    # Detect and track ball (class 0)
    results = model.track(frame, conf=0.35, persist=True, verbose=False)
    boxes = results[0].boxes
    if boxes and len(boxes.xyxy):
        for cls_idx, box in zip(boxes.cls, boxes.xyxy):
            if int(cls_idx) == 0:
                x1, y1, x2, y2 = map(int, box.tolist())
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # Update filter and history
                meas = np.array([[np.float32(cx)], [np.float32(cy)]])
                kalman.correct(meas)
                centroid_history.add((cx, cy))

                # Draw detection
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                break

    # Draw past trail
    trail = centroid_history.get_queue()
    for prev, curr in zip(trail, list(trail)[1:]):
        cv2.line(frame, prev, curr, (255, 0, 0), 2)

    # Predict and draw future positions
    if trail:
        future = []
        for _ in range(5):  # current + 4 future steps
            pred = kalman.predict()
            future.append((int(pred[0]), int(pred[1])))

        for p0, p1 in zip(future, future[1:]):
            cv2.line(frame, p0, p1, (0, 255, 0), 2)
            cv2.circle(frame, p1, 3, (0, 0, 255), -1)

    # Overlay FPS and show
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow("Kalman Visual", cv2.resize(frame, (1000, 600)))

    key = cv2.waitKey(80) & 0xFF
    if key == ord('q'):
        break
    if key == ord(' '):
        paused = not paused
        while paused:
            k = cv2.waitKey(30) & 0xFF
            if k in (ord(' '), ord('q')):
                paused = False
                break

cap.release()
cv2.destroyAllWindows()
