import os
import cv2
import json
import numpy as np
import psutil
import matplotlib.pyplot as plt

# Ensure the "predictions" directory exists
os.makedirs("predictions", exist_ok=True)

def initialize_kalman_filter():
    kalman = cv2.KalmanFilter(4, 2)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    return kalman

def calculate_rmse(actual, predicted):
    actual, predicted = np.array(actual), np.array(predicted)
    return np.sqrt(np.mean((actual - predicted) ** 2))

def load_ground_truth(frame_id):
    gt_path = os.path.join("ground_truth", f"frame_{frame_id:04d}_ground_truth.json")
    if os.path.exists(gt_path):
        with open(gt_path, "r") as f:
            data = json.load(f)
        if data:
            return int(data[0]["center_x"]), int(data[0]["center_y"])
    return None, None

# Initialize video capture
video_path = os.path.join('videos', 'test2.mp4')
cap = cv2.VideoCapture(video_path)

# Initialize Kalman filter
kalman = initialize_kalman_filter()

# Initialize lists for error tracking
rmse_x_values, rmse_y_values, rmse_total_values = [], [], []
memory_usage = []

frame_id = 0
actual_positions, predicted_positions = [], []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_id += 1
    gt_x, gt_y = load_ground_truth(frame_id)
    if gt_x is None or gt_y is None:
        continue  # Skip frames without ground truth
    
    # Kalman filter prediction
    measurement = np.array([[np.float32(gt_x)], [np.float32(gt_y)]])
    kalman.correct(measurement)
    prediction = kalman.predict()
    pred_x, pred_y = int(prediction[0]), int(prediction[1])
    
    actual_positions.append((gt_x, gt_y))
    predicted_positions.append((pred_x, pred_y))
    
    # Save predictions as JSON
    prediction_data = {"center_x": int(pred_x), "center_y": int(pred_y)}
    with open(os.path.join("predictions", f"{frame_id:04d}_KF.json"), "w") as f:
        json.dump(prediction_data, f, indent=4)
    
    # Draw actual and predicted positions
    cv2.circle(frame, (gt_x, gt_y), 4, (0, 0, 255), -1)  # Red for ground truth
    cv2.circle(frame, (pred_x, pred_y), 4, (0, 255, 0), -1)  # Green for KF
    
    # RMSE Calculation
    if actual_positions and predicted_positions:
        actual_x, actual_y = zip(*actual_positions)
        pred_x_vals, pred_y_vals = zip(*predicted_positions)
        
        rmse_x, rmse_y = calculate_rmse(actual_x, pred_x_vals), calculate_rmse(actual_y, pred_y_vals)
        rmse_total = np.sqrt(rmse_x**2 + rmse_y**2)
        
        rmse_x_values.append(rmse_x)
        rmse_y_values.append(rmse_y)
        rmse_total_values.append(rmse_total)
    
    # Memory Usage
    memory_usage.append(psutil.Process().memory_info().rss / 1024 ** 2)
    
    cv2.imshow('Kalman Filter Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# RMSE Graph
plt.plot(rmse_x_values, label='RMSE (X)')
plt.plot(rmse_y_values, label='RMSE (Y)')
plt.plot(rmse_total_values, label='RMSE (Total)')
plt.xlabel('Frame')
plt.ylabel('RMSE (px)')
plt.title('RMSE of Kalman Filter Predictions')
plt.legend()
plt.show()

# Memory Usage Graph
plt.plot(memory_usage, label='Memory Usage (MB)')
plt.xlabel('Frame')
plt.ylabel('Memory Usage (MB)')
plt.title('Memory Usage Over Time')
plt.legend()
plt.show()
