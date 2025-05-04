# Project Structure



```
.
├── CricketBallTrackingv2
│   ├── README.dataset.txt
│   ├── README.roboflow.txt
│   ├── data.yaml
│   ├── singleDatasetmerge.py
│   ├── test
│   ├── train
│   └── valid
├── CricketBallTrajectoryv2
│   ├── README.dataset.txt
│   ├── README.roboflow.txt
│   ├── data.yaml
│   ├── singleDatasetmerge.py
│   ├── test
│   ├── train
│   └── valid
├── Evaluation 
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
├── kalmanpredict.py
├── particlefilterpredict.py
├── predictions
│   ├── 0001_KF.json
│   ├── 0001_PF.json
│   ├── 0002_KF.json
│   ├── 0002_PF.json
├── predictionsv2
│   ├── 0001_KF.json
│   ├── 0001_PF.json
│   ├── 0002_KF.json
│   ├── 0002_PF.json
├── pytests.py
├── requirements.txt
├── videos
│   ├── test2.mp4
│   └── test3.mp4
├── visualKalman.py
├── visualParticle.py
└── youtubeVideoExtractor.py

28 directories, 642 files
```

