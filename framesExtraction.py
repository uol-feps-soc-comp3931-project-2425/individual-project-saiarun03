import cv2
import os

# Set video path
video_path = "videos/test2.mp4"  # Change this to your video file
output_folder = "frames3 "  # Folder to save frames
frame_rate = 1  # Extract one frame every 5 frames

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Open video file
cap = cv2.VideoCapture(video_path)

frame_count = 0
saved_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Stop if video ends
    
    # Save frame at specified interval
    if frame_count % frame_rate == 0:
        frame_filename = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        saved_count += 1
        print(f"Saved {frame_filename}")

    frame_count += 1

cap.release()
print(f"Extracted {saved_count} frames and saved to '{output_folder}'")
