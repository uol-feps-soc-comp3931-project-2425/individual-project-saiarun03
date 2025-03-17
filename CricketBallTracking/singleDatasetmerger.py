import os
import shutil
import re

# Define dataset paths
dataset_paths = ["train", "test", "valid"]  # Folders to merge
output_images = "dataset_ordered/images"
output_labels = "dataset_ordered/labels"

# Create merged output folders
os.makedirs(output_images, exist_ok=True)
os.makedirs(output_labels, exist_ok=True)

# Extract numerical frame number from filename
def get_frame_number(filename):
    match = re.search(r"(\d+)", filename)  # Extract number from filename
    return int(match.group()) if match else float('inf')

# Collect and sort all images and labels
all_images = []
all_labels = []

for dataset in dataset_paths:
    image_path = os.path.join(dataset, "images")
    label_path = os.path.join(dataset, "labels")
    
    if os.path.exists(image_path):
        all_images.extend([os.path.join(image_path, f) for f in os.listdir(image_path)])
    if os.path.exists(label_path):
        all_labels.extend([os.path.join(label_path, f) for f in os.listdir(label_path)])

# Sort images and labels by frame number
all_images = sorted(all_images, key=get_frame_number)
all_labels = sorted(all_labels, key=get_frame_number)

# Rename and move files to new ordered dataset
for idx, image in enumerate(all_images):
    new_image_name = f"frame_{idx:04d}.jpg"
    new_label_name = f"frame_{idx:04d}.txt"
    
    shutil.copy(image, os.path.join(output_images, new_image_name))
    
    # Find matching label (if exists)
    matching_label = next((lbl for lbl in all_labels if os.path.splitext(os.path.basename(image))[0] in lbl), None)
    if matching_label:
        shutil.copy(matching_label, os.path.join(output_labels, new_label_name))

print("✅ Dataset merged, sorted, and renamed sequentially!")
