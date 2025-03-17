import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Paths to data folders
ground_truth_path = "ground_truth"
predictions_path = "predictions"

# Function to calculate RMSE
def calculate_rmse(actual, predicted):
    return np.sqrt(np.mean((np.array(actual) - np.array(predicted)) ** 2))

# Function to calculate MAE
def calculate_mae(actual, predicted):
    return np.mean(np.abs(np.array(actual) - np.array(predicted)))

# Load ground truth data
ground_truth = {}
for file in sorted(os.listdir(ground_truth_path)):
    if file.endswith(".json"):
        frame_id = file.split("_")[1]  # Extract frame number
        with open(os.path.join(ground_truth_path, file), "r") as f:
            data = json.load(f)
            if data:  # Ensure it's not empty
                ground_truth[frame_id] = (data[0]["center_x"], data[0]["center_y"])

# Load KF and PF predictions
kf_predictions = {}
pf_predictions = {}

for file in sorted(os.listdir(predictions_path)):
    if file.endswith("_KF.json"):
        frame_id = file.split("_")[0]  # Extract frame number
        with open(os.path.join(predictions_path, file), "r") as f:
            data = json.load(f)
            kf_predictions[frame_id] = (data["center_x"], data["center_y"])
    
    elif file.endswith("_PF.json"):
        frame_id = file.split("_")[0]  # Extract frame number
        with open(os.path.join(predictions_path, file), "r") as f:
            data = json.load(f)
            pf_predictions[frame_id] = (data["center_x"], data["center_y"])

# Compute RMSE and MAE
kf_rmse_values, pf_rmse_values = [], []
kf_mae_values, pf_mae_values = [], []
frames = []

for frame_id in ground_truth.keys():
    if frame_id in kf_predictions and frame_id in pf_predictions:
        gt_x, gt_y = ground_truth[frame_id]
        kf_x, kf_y = kf_predictions[frame_id]
        pf_x, pf_y = pf_predictions[frame_id]

        # Compute errors
        kf_rmse = calculate_rmse((gt_x, gt_y), (kf_x, kf_y))
        pf_rmse = calculate_rmse((gt_x, gt_y), (pf_x, pf_y))

        kf_mae = calculate_mae((gt_x, gt_y), (kf_x, kf_y))
        pf_mae = calculate_mae((gt_x, gt_y), (pf_x, pf_y))

        # Store values
        frames.append(int(frame_id))
        kf_rmse_values.append(kf_rmse)
        pf_rmse_values.append(pf_rmse)
        kf_mae_values.append(kf_mae)
        pf_mae_values.append(pf_mae)

# Convert lists to numpy arrays for better analysis
kf_rmse_values = np.array(kf_rmse_values)
pf_rmse_values = np.array(pf_rmse_values)
kf_mae_values = np.array(kf_mae_values)
pf_mae_values = np.array(pf_mae_values)

# Compute mean RMSE and MAE
mean_kf_rmse, mean_pf_rmse = np.mean(kf_rmse_values), np.mean(pf_rmse_values)
mean_kf_mae, mean_pf_mae = np.mean(kf_mae_values), np.mean(pf_mae_values)

# 📊 PLOTS

# 1️⃣ Line Plot: RMSE over frames
plt.figure(figsize=(10, 5))
plt.plot(frames, kf_rmse_values, label="KF RMSE", marker="o")
plt.plot(frames, pf_rmse_values, label="PF RMSE", marker="s")
plt.xlabel("Frame Number")
plt.ylabel("RMSE (px)")
plt.title("RMSE Over Frames")
plt.legend()
plt.grid()
plt.show()

# 2️⃣ Bar Chart: Mean RMSE comparison
plt.figure(figsize=(6, 4))
plt.bar(["Kalman Filter", "Particle Filter"], [mean_kf_rmse, mean_pf_rmse], color=["blue", "green"])
plt.ylabel("Mean RMSE (px)")
plt.title("Average RMSE Comparison")
plt.show()

# 3️⃣ Box Plot: RMSE distribution
plt.figure(figsize=(6, 4))
plt.boxplot([kf_rmse_values, pf_rmse_values], labels=["KF RMSE", "PF RMSE"], patch_artist=True)
plt.ylabel("RMSE (px)")
plt.title("RMSE Distribution")
plt.show()

# 4️⃣ Box Plot: MAE distribution
plt.figure(figsize=(6, 4))
plt.boxplot([kf_mae_values, pf_mae_values], labels=["KF MAE", "PF MAE"], patch_artist=True)
plt.ylabel("MAE (px)")
plt.title("MAE Distribution")
plt.show()

# Print summary
print("\n📊 **Summary of Results** 📊")
print(f"✅ Mean RMSE - KF: {mean_kf_rmse:.2f} px | PF: {mean_pf_rmse:.2f} px")
print(f"✅ Mean MAE  - KF: {mean_kf_mae:.2f} px | PF: {mean_pf_mae:.2f} px")
