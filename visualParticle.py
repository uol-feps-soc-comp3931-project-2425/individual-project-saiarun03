import os
import time
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO

class ParticleFilter:
    """Particle filter for 2D position tracking."""
    def __init__(self, num_particles, init_pos, frame_size):
        self.num_particles = num_particles
        self.particles = np.tile(init_pos, (num_particles, 1))
        self.weights = np.full(num_particles, 1 / num_particles)
        self.frame_h, self.frame_w = frame_size

    def predict(self, std_dev=10):
        """Add Gaussian noise and clamp to frame bounds."""
        noise = np.random.randn(self.num_particles, 2) * std_dev
        self.particles += noise
        self.particles[:, 0] = np.clip(self.particles[:, 0], 0, self.frame_w)
        self.particles[:, 1] = np.clip(self.particles[:, 1], 0, self.frame_h)

    def update(self, measurement, std_dev=15):
        """Weight particles by distance to measurement."""
        dists = np.linalg.norm(self.particles - measurement, axis=1)
        self.weights = np.exp(-dists**2 / (2 * std_dev**2)) + 1e-8
        self.weights /= self.weights.sum()

    def resample(self):
        """Resample particles proportional to their weights."""
        idx = np.random.choice(
            self.num_particles, self.num_particles, p=self.weights
        )
        self.particles = self.particles[idx]
        self.weights.fill(1 / self.num_particles)

    def estimate(self):
        """Return weighted mean position."""
        return np.average(self.particles, axis=0, weights=self.weights).astype(int)

class FixedSizeQueue:
    """Fixed-length queue for recent positions."""
    def __init__(self, max_size):
        self.queue = deque(maxlen=max_size)
    
    def add(self, item):
        self.queue.append(item)

    def get_queue(self):
        return self.queue

# Initialize model and video
model = YOLO(os.path.join('runs', 'detect', 'train5', 'weights', 'best.pt'))
cap = cv2.VideoCapture(os.path.join('videos', 'test2.mp4'))
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

pf = None
trail = FixedSizeQueue(10)
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

    # Detect ball (class 0)
    results = model.track(frame, conf=0.35, persist=True, verbose=False)
    boxes = results[0].boxes
    detected = False

    if boxes and len(boxes.xyxy):
        for cls, xyxy in zip(boxes.cls, boxes.xyxy):
            if int(cls) == 0:
                x1, y1, x2, y2 = map(int, xyxy.tolist())
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                detected = True
                break

    if detected:
        # Initialize filter on first detection
        if pf is None:
            pf = ParticleFilter(300, (cx, cy), (frame_h, frame_w))

        # PF cycle
        pf.predict()
        pf.update(np.array([cx, cy]))
        pf.resample()
        ex, ey = pf.estimate()
        trail.add((ex, ey))

        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)  # YOLO
        cv2.circle(frame, (ex, ey), 5, (255, 0, 0), -1)  # PF estimate

    # Draw trail
    pts = list(trail.get_queue())
    for p0, p1 in zip(pts, pts[1:]):
        cv2.line(frame, p0, p1, (255, 0, 0), 2)

    # Linear future prediction
    if len(pts) >= 2:
        dx, dy = pts[-1][0] - pts[-2][0], pts[-1][1] - pts[-2][1]
        future = [pts[-1]]
        for _ in range(4):
            future.append((future[-1][0] + dx, future[-1][1] + dy))
        for p0, p1 in zip(future, future[1:]):
            cv2.line(frame, p0, p1, (0, 255, 0), 2)
            cv2.circle(frame, p1, 3, (0, 0, 255), -1)

    # Display
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.imshow("Particle Filter Visual", cv2.resize(frame, (1000, 600)))

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
