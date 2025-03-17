import os
import cv2
import json
import numpy as np
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Load YOLO model
model_path = os.path.join('runs', 'detect', 'train5', 'weights', 'best.pt')
model = YOLO(model_path)

# Initialize DeepSORT Tracker
deep_sort = DeepSort("deep_sort/deep/checkpoint/ckpt.t7")

# Load video
video_path = os.path.join('videos', 'test2.mp4')
cap = cv2.VideoCapture(video_path)

# Ensure prediction directory exists
os.makedirs("predictions", exist_ok=True)

# RMSE Data Storage
actual_positions = []
predicted_positions = []
rmse_x_values, rmse_y_values, rmse_total_values = [], [], []

frame_id = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_id += 1
    results = model(frame)
    detections = []  # Stores (x1, y1, x2, y2, conf, class_id)
    
    for result in results:
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box[:4])
            conf = box[4].item()
            class_id = int(box[5].item())
            
            if class_id == 0:  # Assuming class 0 is the ball
                detections.append([x1, y1, x2, y2, conf])
    
    # Convert detections to NumPy format
    dets = np.array(detections)
    
    # Update DeepSORT tracker
    if len(dets) > 0:
        outputs = deep_sort.update(dets, frame)
        
        for track in outputs:
            track_id, x1, y1, x2, y2 = track[:5]
            center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
            
            # Store predicted positions
            predicted_positions.append((center_x, center_y))
            
            # Draw tracking box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.circle(frame, (center_x, center_y), 3, (0, 0, 255), -1)
            
            # Save predictions
            prediction_data = {"center_x": center_x, "center_y": center_y}
            with open(os.path.join("predictions", f"{frame_id:04d}_DeepSORT.json"), "w") as f:
                json.dump(prediction_data, f, indent=4)
    
    # Load ground truth
    gt_path = os.path.join("ground_truth", f"frame_{frame_id:04d}_ground_truth.json")
    if os.path.exists(gt_path):
        with open(gt_path, "r") as f:
            gt_data = json.load(f)
            if gt_data:
                actual_x, actual_y = gt_data[0]["center_x"], gt_data[0]["center_y"]
                actual_positions.append((actual_x, actual_y))
    
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()

# Compute RMSE
actual_positions = np.array(actual_positions)
predicted_positions = np.array(predicted_positions)
if len(actual_positions) > 0 and len(predicted_positions) > 0:
    rmse_x = np.sqrt(np.mean((actual_positions[:, 0] - predicted_positions[:, 0]) ** 2))
    rmse_y = np.sqrt(np.mean((actual_positions[:, 1] - predicted_positions[:, 1]) ** 2))
    rmse_total = np.sqrt(rmse_x**2 + rmse_y**2)
    
    print(f"RMSE X: {rmse_x:.2f} pixels")
    print(f"RMSE Y: {rmse_y:.2f} pixels")
    print(f"Total RMSE: {rmse_total:.2f} pixels")
    
    # Plot RMSE Graphs
    plt.plot(rmse_x_values, label='RMSE (X)')
    plt.plot(rmse_y_values, label='RMSE (Y)')
    plt.plot(rmse_total_values, label='Total RMSE')
    plt.xlabel('Frame')
    plt.ylabel('RMSE (px)')
    plt.title('DeepSORT Tracking RMSE')
    plt.legend()
    plt.show()
