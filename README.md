# Cricket Ball Tracking & Trajectory Analysis

A complete pipeline for detecting, tracking and forecasting cricket ball trajectories using YOLOv8, Kalman filters, particle filters, and benchmarking scripts.

---

## Table of Contents

1. [Project Structure](#project-structure)  
2. [Prerequisites](#prerequisites)  
3. [Dataset Acquisition](#dataset-acquisition)  
4. [Installation](#installation)  
5. [Usage](#usage)  
6. [Module Descriptions](#module-descriptions)  
7. [Benchmarking & Evaluation](#benchmarking--evaluation)  
8. [Testing](#testing)  
9. [License](#license)  

---

## Project Structure

```
.
├── CricketBallTrackingv2
│   ├── README.dataset.txt
│   ├── README.roboflow.txt
│   ├── data.yaml
│   ├── singleDatasetmerge.py # Merge multiple label sets into one
│   ├── test
│   ├── train
│   └── valid
├── CricketBallTrajectoryv2
│   ├── README.dataset.txt
│   ├── README.roboflow.txt
│   ├── data.yaml
│   ├── singleDatasetmerge.py # Merge multiple label sets into one
│   ├── test
│   ├── train
│   └── valid
├── Evaluation  # Scripts to compute tracking & prediction errors
│   ├── benchmark.py
│   ├── globalerror.py
│   ├── results
│   └── results.py
├── Frames Extraction for Video
│   ├── frameToGroundTruth.py
│   └── framesExtraction.py
├── README.md
├── YOLOv8 Training
│   ├── ball_tracking_train.py
│   ├── cricket_ball_data
│   ├── modelSave.py
│   └── runs
├── benchmarking
│   ├── Fast-Bowling_MAE_per_frame.png
│   ├── Fast-Bowling_MSE_per_frame.png
│   ├── Fast-Bowling_acceleration_vs_time.png
│   ├── Fast-Bowling_memory_over_time.png
│   ├── Spin-Bowling_MAE_per_frame.png
│   ├── Spin-Bowling_MSE_per_frame.png
│   ├── Spin-Bowling_acceleration_vs_time.png
│   ├── Spin-Bowling_memory_over_time.png
│   ├── global_average_mae.png
│   ├── global_average_rmse.png
│   └── summary_metrics.csv
├── data.yaml
├── dataset_orderedv2
│   ├── images
│   └── labels
├── dataset_orderedv3
│   ├── images
│   └── labels
├── gen_readme.py
├── ground_truthv2
│   ├── frame_0000_ground_truth.json
│   ├── frame_0001_ground_truth.json
│   ├── frame_0002_ground_truth.json
├── ground_truthv3
│   ├── frame_0000_ground_truth.json
│   ├── frame_0001_ground_truth.json
│   ├── frame_0002_ground_truth.json
├── images
│   └── predicting_ball_path.png
├── kalmanpredict.py  # Apply Kalman filter to detections
├── particlefilterpredict.py # Apply particle filter to detections
├── predictions # KF & PF output JSON v1
│   ├── 0001_KF.json
│   ├── 0001_PF.json
│   ├── 0002_KF.json
│   ├── 0002_PF.json
├── predictionsv2 # KF & PF output JSON v2
│   ├── 0001_KF.json
│   ├── 0001_PF.json
│   ├── 0002_KF.json
│   ├── 0002_PF.json
├── pytests.py # Unit tests for core modules
├── requirements.txt # Python dependencies
├── videos # Sample test videos
│   ├── test2.mp4
│   └── test3.mp4
├── visualKalman.py
├── visualParticle.py
└── youtubeVideoExtractor.py # Download video frames from YouTube
```

## Prerequisites

- **Python** 3.8 or higher  
- **CUDA-enabled GPU** (optional, for YOLOv8 training)  
- **Kaggle CLI** (for dataset download):  
  ```bash
  pip install kaggle
  ```

## Dataset Acquisition
This project uses the Cricket Ball Dataset for YOLO. To download and prepare:
1. Configure the Kaggle CLI with your API token (~/.kaggle/kaggle.json)
2 Run:
```bash
kaggle datasets download -d kushagra3204/cricket-ball-dataset-for-yolo
unzip cricket-ball-dataset-for-yolo.zip -d data/
```
3. Edit CricketBallTrackingv2/data.yaml and CricketBallTrajectoryv2/data.yaml to point at data/.

## Installation
1. Clone the repository:
```bash
git clone <repo-url>
cd <repo-directory>
```
2. Install dependencies:
 ```bash
 pip install -r requirements.txt
```
## Usage
1.All scripts are standalone. To execute any module:
```bash
python <script_name>.py
```
Add --help to view flags, for example:
```bash
python ball_tracking_train.py --help
```

## Module Descriptions
- **singleDatasetmerge.py** 
Merge multiple Roboflow exports into one consolidated YOLO dataset (images + labels).

- **ball_tracking_train.py** 
Train YOLOv8 detector on the cricket ball dataset. Outputs weights under runs/.

- **modelSave.py** 
Save & export the best-performing YOLOv8 model for inference.

- **kalmanpredict.py / particlefilterpredict.py** 
Consume YOLO detections (JSON) and apply Kalman or particle filtering to forecast the ball’s path.

- **visualKalman.py / visualParticle.py** 
Overlay predicted trajectories on original video frames and export annotated MP4.

- **benchmark.py, globalerror.py, results.py** 
Compute per-frame MAE/MSE, aggregate global errors, and generate summary plots (saved under benchmarking/).

- **framesExtraction.py & frameToGroundTruth.py** 
Extract frames from test videos and align each frame with its ground-truth JSON.

- **youtubeVideoExtractor.py** 
Download and prepare cricket match clips from YouTube for testing.

- **pytests.py** 
Unit tests covering dataset parsing, filter implementations, and utility functions.

## Benchmarking & Evaluation
Precomputed metrics and performance charts are located in benchmarking/.
To re-run benchmarks:
```bash
python Evaluation/benchmark.py
python Evaluation/results.py
```
This will regenerate MAE/MSE plots and update benchmarking/summary_metrics.csv.

## Testing
Command to run the unit test:
```bash
python pytests.py
```
## License 
This project is released under the MIT License.
