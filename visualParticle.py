import cv2
import time
import math
import os
import json
import numpy as np
from collections import deque
from ultralytics import YOLO

# === PF Tracker ===
class ParticleFilter:
    def __init__(self, num_particles, init_pos, frame_size):
        self.num_particles = num_particles
        self.particles = np.ones((num_particles, 2)) * np.array(init_pos)
        self.weights = np.ones(num_particles) / num_particles
        self.frame_size = frame_size

    def predict(self, std_dev=10):
        noise = np.random.randn(self.num_particles, 2) * std_dev
        self.particles += noise
        self.particles[:, 0] = np.clip(self.particles[:, 0], 0, self.frame_size[1])
        self.particles[:, 1] = np.clip(self.particles[:, 1], 0, self.frame_size[0])

    def update(self, measurement, std_dev=15):
        dists = np.linalg.norm(self.particles - measurement, axis=1)
        self.weights = np.exp(-dists**2 / (2 * std_dev**2)) + 1e-8
        self.weights /= np.sum(self.weights)

    def resample(self):
        indices = np.random.choice(self.num_particles, self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.weights.fill(1.0 / self.num_particles)

    def estimate(self):
        return np.average(self.particles, axis=0, weights=self.weights).astype(int)

# === Trail Queue ===
class FixedSizeQueue:
    def __init__(self, max_size):
        self.queue = deque(maxlen=max_size)
    
    def add(self, item):
        self.queue.append(item)

    def get_queue(self):
        return self.queue
    
    def __len__(self):
        return len(self.queue)

# === Init ===
model_path = os.path.join('runs', 'detect', 'train5', 'weights', 'best.pt')
model = YOLO(model_path)

video_path = os.path.join('videos', 'test2.mp4')
cap = cv2.VideoCapture(video_path)

frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

pf_filter = None
pf_trail = FixedSizeQueue(10)

prev_frame_time = 0
frame_id = 0
paused = False

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_id += 1

    new_time = time.time()
    fps = 1 / (new_time - prev_frame_time) if prev_frame_time else 0
    prev_frame_time = new_time

    # === YOLOv8 Detection ===
    results = model.track(frame, persist=True, conf=0.35, verbose=False)
    boxes = results[0].boxes

    detected = False
    cx, cy = None, None

    if boxes and len(boxes.xyxy) > 0:
        for i in range(len(boxes.cls)):
            if int(boxes.cls[i].item()) == 0:
                x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                detected = True
                break

    # === Init PF if detection found ===
    if detected:
        if pf_filter is None:
            pf_filter = ParticleFilter(300, (cx, cy), (frame_h, frame_w))

        pf_filter.predict()
        pf_filter.update(np.array([cx, cy]))
        pf_filter.resample()
        est_x, est_y = pf_filter.estimate()
        pf_trail.add((est_x, est_y))

        # Draw detection & estimate
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)  # Red = YOLO
        cv2.circle(frame, (est_x, est_y), 5, (255, 0, 0), -1)  # Blue = PF estimate

    # === Draw PF trail ===
    trail = list(pf_trail.get_queue())
    for i in range(1, len(trail)):
        cv2.line(frame, trail[i - 1], trail[i], (255, 0, 0), 2)  # Blue trail

    # === Predict Future Path (Linear) ===
    if len(trail) >= 2:
        x_diff = trail[-1][0] - trail[-2][0]
        y_diff = trail[-1][1] - trail[-2][1]

        future = [trail[-1]]
        for _ in range(4):
            next_pos = (future[-1][0] + x_diff, future[-1][1] + y_diff)
            future.append(next_pos)

        for i in range(1, len(future)):
            cv2.line(frame, future[i - 1], future[i], (0, 255, 0), 2)
            cv2.circle(frame, future[i], 3, (0, 0, 255), -1)

    # === Display ===
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    frame_resized = cv2.resize(frame, (1000, 600))
    cv2.imshow("Particle Filter Visual", frame_resized)

    key = cv2.waitKey(80)
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
