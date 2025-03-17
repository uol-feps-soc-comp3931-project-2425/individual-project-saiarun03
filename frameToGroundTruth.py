import os
import json

# Paths
image_folder = "dataset_ordered/images"
label_folder = "dataset_ordered/labels"
output_folder = "ground_truth"

os.makedirs(output_folder, exist_ok=True)

# Get image dimensions (assuming all images are the same size)
import cv2
sample_img = cv2.imread(os.path.join(image_folder, os.listdir(image_folder)[0]))
img_height, img_width, _ = sample_img.shape  # Get height & width of image

# Process each label file
for label_file in sorted(os.listdir(label_folder)):
    label_path = os.path.join(label_folder, label_file)
    frame_number = label_file.split(".")[0]  # Get frame ID
    
    with open(label_path, "r") as file:
        lines = file.readlines()
    
    ground_truth_data = []
    
    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])  # Object class (should be ball)
        x_center, y_center, width, height = map(float, parts[1:])
        
        # Convert to absolute pixel values
        x_center *= img_width
        y_center *= img_height
        width *= img_width
        height *= img_height

        # Bounding box (x1, y1, x2, y2)
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)

        ground_truth_data.append({
            "class_id": class_id,
            "center_x": int(x_center),
            "center_y": int(y_center),
            "bbox": [x1, y1, x2, y2]
        })
    
    # Save as JSON file for each frame
    with open(os.path.join(output_folder, f"{frame_number}_ground_truth.json"), "w") as json_file:
        json.dump(ground_truth_data, json_file, indent=4)

print("✅ YOLO labels converted to pixel coordinates & saved as JSON!")
