from collections import deque
from ultralytics import YOLO
import math
import time
import cv2
import os
import numpy as np
import psutil
import matplotlib.pyplot as plt

def angle_between_lines(m1, m2=1):
    if m1 != -1/m2:
        angle = math.degrees(math.atan(abs((m2 - m1) / (1 + m1 * m2))))
        return angle
    else:
        return 90.0

class FixedSizeQueue:
    def __init__(self, max_size):
        self.queue = deque(maxlen=max_size)
    
    def add(self, item):
        self.queue.append(item)
    
    def pop(self):
        self.queue.popleft()
        
    def clear(self):
        self.queue.clear()

    def get_queue(self):
        return self.queue
    
    def __len__(self):
        return len(self.queue)

def initialize_kalman_filter():
    kalman = cv2.KalmanFilter(4, 2)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03
    return kalman

def calculate_rmse(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)
    return np.sqrt(np.mean((actual - predicted) ** 2))

# Initialize YOLO model
model_path = os.path.join('runs', 'detect', 'train5', 'weights', 'best.pt')
model = YOLO(model_path)

# Initialize video capture
video_path = os.path.join('videos', 'test1.mp4')
cap = cv2.VideoCapture(video_path)
ret = True
prevTime = 0
centroid_history = FixedSizeQueue(10)
start_time = time.time()
interval = 0.6
paused = False
angle = 0
prev_frame_time = 0 
new_frame_time = 0

# Initialize Kalman filter
kalman = initialize_kalman_filter()

# Initialize lists to store actual and predicted positions
actual_positions = []
predicted_positions = []

# Initialize lists to store RMSE values and memory usage
rmse_x_values = []
rmse_y_values = []
rmse_total_values = []
memory_usage = []

while ret:
    ret, frame = cap.read()
    if ret:
        new_frame_time = time.time() 
        fps = 1/(new_frame_time-prev_frame_time) 
        prev_frame_time = new_frame_time 
        fps = int(fps)  
        fps = str(fps)
        print(list(centroid_history.queue))
        current_time = time.time()
        if current_time - start_time >= interval and len(centroid_history) > 0:
            centroid_history.pop()
            start_time = current_time
        
        results = model.track(frame, persist=True, conf=0.35, verbose=False)
        boxes = results[0].boxes
        box = boxes.xyxy
        rows, cols = box.shape
        if len(box) != 0:
            for i in range(rows):
                x1, y1, x2, y2 = box[i]
                x1, y1, x2, y2 = x1.item(), y1.item(), x2.item(), y2.item()
                
                centroid_x = int((x1 + x2) / 2)
                centroid_y = int((y1 + y2) / 2)
                
                centroid_history.add((centroid_x, centroid_y))
                cv2.circle(frame, (centroid_x, centroid_y), radius=3, color=(0, 0, 255), thickness=-1)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                
                # Kalman filter prediction
                measurement = np.array([[np.float32(centroid_x)], [np.float32(centroid_y)]])
                kalman.correct(measurement)
                prediction = kalman.predict()
                predicted_x, predicted_y = int(prediction[0]), int(prediction[1])
                predicted_positions.append((predicted_x, predicted_y))
                actual_positions.append((centroid_x, centroid_y))
                
                # Draw predicted position
                cv2.circle(frame, (predicted_x, predicted_y), radius=3, color=(0, 255, 0), thickness=-1)
                
        if len(centroid_history) > 1:
            centroid_list = list(centroid_history.get_queue())
            for i in range(1, len(centroid_history)):
                cv2.line(frame, centroid_history.get_queue()[i-1], centroid_history.get_queue()[i], (255, 0, 0), 4)    
                
        if len(centroid_history) > 1:
            centroid_list = list(centroid_history.get_queue())
            x_diff = centroid_list[-1][0] - centroid_list[-2][0]
            y_diff = centroid_list[-1][1] - centroid_list[-2][1]
            if x_diff != 0:
                m1 = y_diff / x_diff
                if m1 == 1:
                    angle = 90
                elif m1 != 0:
                    angle = 90 - angle_between_lines(m1)
                if angle >= 45:
                    print("ball bounced")
            future_positions = [centroid_list[-1]]
            for i in range(1, 5):
                future_positions.append(
                    (
                        centroid_list[-1][0] + x_diff * i,
                        centroid_list[-1][1] + y_diff * i
                    )
                )
            print("Future Positions: ", future_positions)
            for i in range(1, len(future_positions)):
                cv2.line(frame, future_positions[i-1], future_positions[i], (0, 255, 0), 4)
                cv2.circle(frame, future_positions[i], radius=3, color=(0, 0, 255), thickness=-1)
                
        text = "Angle: {:.2f} degrees".format(angle)
        cv2.putText(frame, text, (20, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
        cv2.putText(frame, f'FPS: {fps}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2) 
        
        # Calculate RMSE for x and y coordinates
        if len(actual_positions) > 0 and len(predicted_positions) > 0:
            actual_x = [pos[0] for pos in actual_positions]
            actual_y = [pos[1] for pos in actual_positions]
            predicted_x = [pos[0] for pos in predicted_positions]
            predicted_y = [pos[1] for pos in predicted_positions]
            
            rmse_x = calculate_rmse(actual_x, predicted_x)
            rmse_y = calculate_rmse(actual_y, predicted_y)
            rmse_total = np.sqrt(rmse_x**2 + rmse_y**2)  # Combined RMSE
            
            # Store RMSE values for graphing
            rmse_x_values.append(rmse_x)
            rmse_y_values.append(rmse_y)
            rmse_total_values.append(rmse_total)
            
            # Display RMSE
            cv2.putText(frame, f'RMSE (X): {rmse_x:.2f} px', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame, f'RMSE (Y): {rmse_y:.2f} px', (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame, f'RMSE (Total): {rmse_total:.2f} px', (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # Display memory usage
        memory_usage.append(psutil.Process().memory_info().rss / 1024 ** 2)  # Memory usage in MB
        cv2.putText(frame, f'Memory: {memory_usage[-1]:.2f} MB', (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        frame_resized = cv2.resize(frame, (1000, 600))
        cv2.imshow('frame', frame_resized)
         
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        elif key & 0xFF == ord(' '):
            paused = not paused
            
            while paused:
                key = cv2.waitKey(30) & 0xFF
                if key == ord(' '):
                    paused = not paused
                elif key == ord('q'):
                    break

cap.release()
cv2.destroyAllWindows()

# Generate RMSE graph
plt.plot(rmse_x_values, label='RMSE (X)')
plt.plot(rmse_y_values, label='RMSE (Y)')
plt.plot(rmse_total_values, label='RMSE (Total)')
plt.xlabel('Frame')
plt.ylabel('RMSE (px)')
plt.title('RMSE of Kalman Filter Predictions')
plt.legend()
plt.show()

# Generate memory usage graph
plt.plot(memory_usage, label='Memory Usage (MB)')
plt.xlabel('Frame')
plt.ylabel('Memory Usage (MB)')
plt.title('Memory Usage Over Time')
plt.legend()
plt.show()