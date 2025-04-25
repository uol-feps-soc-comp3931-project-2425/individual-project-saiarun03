
import os
import cv2
import json
import numpy as np
from ultralytics import YOLO

# Particle Filter class
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

# Load YOLOv8 model
model_path = os.path.join('runs', 'detect', 'train5', 'weights', 'best.pt')
model = YOLO(model_path)

# Set paths
video_path = os.path.join('videos', 'test2.mp4')
os.makedirs("predictionsv2", exist_ok=True)

# Open video
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

frame_id = 0
particle_filter = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1

    # YOLO detection
    results = model(frame)
    boxes = results[0].boxes

    cx, cy = None, None
    if boxes and len(boxes.xyxy) > 0:
        for i in range(len(boxes.cls)):
            if int(boxes.cls[i].item()) == 0:  # Class 0 = ball
                x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                break

    # No detection → write empty prediction
    if cx is None or cy is None:
        with open(os.path.join("predictionsv2", f"{frame_id:04d}_PF.json"), "w") as f:
            json.dump({}, f, indent=4)
        continue

    # Initialize particle filter if first detection
    if particle_filter is None:
        particle_filter = ParticleFilter(300, (cx, cy), (frame_height, frame_width))

    # PF tracking step
    particle_filter.predict()
    particle_filter.update(np.array([cx, cy]))
    particle_filter.resample()
    pred_x, pred_y = particle_filter.estimate()

    # Save prediction with safe Python int conversion
    with open(os.path.join("predictionsv2", f"{frame_id:04d}_PF.json"), "w") as f:
        json.dump({
            "center_x": int(pred_x),
            "center_y": int(pred_y)
        }, f, indent=4)

cap.release()
cv2.destroyAllWindows()

