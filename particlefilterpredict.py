import cv2
import numpy as np
import json
import os
import psutil
import matplotlib.pyplot as plt

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
        self.weights = np.exp(-dists**2 / (2 * std_dev**2))
        self.weights += 1e-8  # Avoid zero weights
        self.weights /= np.sum(self.weights)
    
    def resample(self):
        indices = np.random.choice(self.num_particles, self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.weights.fill(1.0 / self.num_particles)
    
    def estimate(self):
        return np.average(self.particles, axis=0, weights=self.weights).astype(int)

def calculate_rmse(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)
    return np.sqrt(np.mean((actual - predicted) ** 2))

# Load video
video_path = os.path.join('videos', 'test2.mp4')
cap = cv2.VideoCapture(video_path)

# Prepare output directory
os.makedirs("predictions", exist_ok=True)

frame_id = 0
ret, frame = cap.read()
if not ret:
    print("Error: Cannot read video.")
    exit()

frame_height, frame_width = frame.shape[:2]
particle_filter = None
actual_positions, predicted_positions = [], []
rmse_x_values, rmse_y_values, rmse_total_values, memory_usage = [], [], [], []

while ret:
    frame_id += 1
    ret, frame = cap.read()
    if not ret:
        break

    ground_truth_file = os.path.join("ground_truth", f"frame_{frame_id:04d}_ground_truth.json")
    if os.path.exists(ground_truth_file):
        with open(ground_truth_file, 'r') as f:
            ground_truth_data = json.load(f)
            if ground_truth_data:
                gt_x, gt_y = ground_truth_data[0]['center_x'], ground_truth_data[0]['center_y']
            else:
                continue
    else:
        continue
    
    if particle_filter is None:
        particle_filter = ParticleFilter(num_particles=300, init_pos=(gt_x, gt_y), frame_size=(frame_height, frame_width))
    
    particle_filter.predict()
    particle_filter.update(np.array([gt_x, gt_y]))
    particle_filter.resample()
    predicted_x, predicted_y = particle_filter.estimate()
    
    actual_positions.append((gt_x, gt_y))
    predicted_positions.append((predicted_x, predicted_y))
    
    cv2.circle(frame, (gt_x, gt_y), 5, (0, 0, 255), -1)
    cv2.circle(frame, (predicted_x, predicted_y), 5, (0, 255, 0), -1)
    
    for p in particle_filter.particles:
        cv2.circle(frame, tuple(p.astype(int)), 1, (255, 255, 255), -1)
    
    rmse_x = calculate_rmse([p[0] for p in actual_positions], [p[0] for p in predicted_positions])
    rmse_y = calculate_rmse([p[1] for p in actual_positions], [p[1] for p in predicted_positions])
    rmse_total = np.sqrt(rmse_x**2 + rmse_y**2)
    rmse_x_values.append(rmse_x)
    rmse_y_values.append(rmse_y)
    rmse_total_values.append(rmse_total)
    
    memory_usage.append(psutil.Process().memory_info().rss / 1024 ** 2)
    
    cv2.putText(frame, f'RMSE (X): {rmse_x:.2f} px', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f'RMSE (Y): {rmse_y:.2f} px', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f'Memory: {memory_usage[-1]:.2f} MB', (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    frame_resized = cv2.resize(frame, (1000, 600))
    cv2.imshow('Particle Filter Tracking', frame_resized)
    
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    
    prediction_data = {"center_x": predicted_x, "center_y": predicted_y}
    # Convert NumPy int64 to Python int
    prediction_data = {
        "center_x": int(predicted_x),
        "center_y": int(predicted_y)
    }

    with open(os.path.join("predictions", f"{frame_id:04d}_PF.json"), "w") as f:
        json.dump(prediction_data, f, indent=4)

cap.release()
cv2.destroyAllWindows()

plt.plot(rmse_x_values, label='RMSE (X)')
plt.plot(rmse_y_values, label='RMSE (Y)')
plt.plot(rmse_total_values, label='RMSE (Total)')
plt.xlabel('Frame')
plt.ylabel('RMSE (px)')
plt.title('RMSE of Particle Filter Predictions')
plt.legend()
plt.show()

plt.plot(memory_usage, label='Memory Usage (MB)')
plt.xlabel('Frame')
plt.ylabel('Memory Usage (MB)')
plt.title('Memory Usage Over Time')
plt.legend()
plt.show()
