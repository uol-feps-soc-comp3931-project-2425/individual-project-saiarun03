# fast_eval_with_trackers.py

import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ─── USER SETTINGS ─────────────────────────────────────────────────────────────
# Toggle these at the top of the script:
SAVE_PLOTS = True      # ← set to False (or comment out this line) to disable writing PNG files
SHOW_PLOTS = False     # ← set to True  (or comment out this line) to pop up each figure interactively
# ────────────────────────────────────────────────────────────────────────────────

MODEL_PATH = os.path.join('runs','detect','train5','weights','best.pt')
YOLO_MODEL = YOLO(MODEL_PATH)

# Clip configurations: label → (video, ground-truth-dir, predictions-dir)
CLIPS = {
    "Fast‑Bowling": ("videos/test2.mp4", "ground_truthv2", "predictionsv2"),
    "Spin‑Bowling": ("videos/test3.mp4", "ground_truthv3", "predictions")
}

# Confidence thresholds to sweep
THRESHOLDS = np.linspace(0.0, 1.0, 101)

# Where to dump results
RESULTS_ROOT = "results"

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def load_gt_mask_and_positions(gt_folder, total_frames):
    """Returns a boolean mask of frames with ground-truth and a dict of positions."""
    mask = np.zeros(total_frames, dtype=bool)
    pos  = {}
    for fn in os.listdir(gt_folder):
        if not fn.endswith('.json'): continue
        fid = int(fn.split('_')[1])
        data = json.load(open(os.path.join(gt_folder, fn)))
        if isinstance(data, list) and data:
            x,y = data[0]['center_x'], data[0]['center_y']
            mask[fid-1] = True
            pos[fid]     = (x,y)
    return mask, pos

def collect_confidences(video_file):
    """Runs YOLO on every frame (no threshold) to record the ball confidence each frame."""
    cap   = cv2.VideoCapture(video_file)
    N     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    confs = np.zeros(N, dtype=float)

    for idx in range(N):
        ret, frame = cap.read()
        if not ret: break
        res = YOLO_MODEL(frame, conf=0.0)[0]
        cls_ids = res.boxes.cls.cpu().numpy().astype(int)
        scores  = res.boxes.conf.cpu().numpy().astype(float)
        ball_scores = scores[cls_ids == 0]
        confs[idx]   = ball_scores.max() if ball_scores.size>0 else 0.0

    cap.release()
    return confs

def compute_detector_curves(confs, gt_mask):
    """Sweep thresholds, return Precision, Recall, F1 curves and TP frame sets."""
    precisions, recalls, f1s = [], [], []
    tp_frames_map = {}
    for thr in THRESHOLDS:
        det_mask = confs >= thr
        TP = int((det_mask & gt_mask).sum())
        FP = int((det_mask & ~gt_mask).sum())
        FN = int((~det_mask & gt_mask).sum())
        P  = TP/(TP+FP) if TP+FP else 0.0
        R  = TP/(TP+FN) if TP+FN else 0.0
        F1 = 2*P*R/(P+R)    if P+R   else 0.0

        precisions.append(P)
        recalls.append(R)
        f1s.append(F1)
        tp_frames_map[thr] = set(np.nonzero(det_mask & gt_mask)[0] + 1)

    return np.array(precisions), np.array(recalls), np.array(f1s), tp_frames_map

def load_predictions(pred_folder, suffix):
    """Read {frame: center} for files *_<suffix>.json."""
    out = {}
    for fn in os.listdir(pred_folder):
        if not fn.endswith(f"_{suffix}.json"): continue
        fid = int(fn.split('_')[0])
        data = json.load(open(os.path.join(pred_folder, fn)))
        if 'center_x' in data and 'center_y' in data:
            out[fid] = (data['center_x'], data['center_y'])
    return out

# def compute_tracker_errors(gt_pos, pred_pos, eval_frames):
#     """On the subset of TP frames, compute RMSE and MAE."""
#     y_true, y_pred = [], []
#     for fid in eval_frames:
#         if fid in gt_pos and fid in pred_pos:
#             y_true.append(gt_pos[fid])
#             y_pred.append(pred_pos[fid])
#     if not y_true:
#         return np.nan, np.nan
#     y_true = np.array(y_true)
#     y_pred = np.array(y_pred)
#     rmse = np.sqrt(mean_squared_error(y_true, y_pred))
#     mae  = mean_absolute_error(y_true, y_pred)
#     return rmse, mae

if __name__ == "__main__":
    ensure_dir(RESULTS_ROOT)

    for clip_name, (vid, gt_folder, pred_folder) in CLIPS.items():
        print(f"\n\n===== {clip_name} =====")
        cap = cv2.VideoCapture(vid)
        N   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        out_dir = ensure_dir(os.path.join(RESULTS_ROOT, clip_name.replace(" ", "_")))

        # 1) Load GT
        gt_mask, gt_pos = load_gt_mask_and_positions(gt_folder, N)

        # 2) Gather raw confidences
        print("Collecting YOLO confidences…")
        confs = collect_confidences(vid)

        # 3) Build P/R/F1 curves
        print("Computing P/R/F1 curves…")
        P_curve, R_curve, F1_curve, tp_frames_map = compute_detector_curves(confs, gt_mask)

        # 4) Plot & optionally save/show detection curves
        for arr, label, fname, color in [
            (P_curve,  "Precision",    "precision.png", "tab:blue"),
            (R_curve,  "Recall",       "recall.png",    "tab:green"),
            (F1_curve, "F₁ Score",     "f1_score.png",  "tab:purple"),
        ]:
            plt.figure(figsize=(6,4))
            plt.plot(THRESHOLDS, arr, color=color)
            plt.title(f"{clip_name}: {label} vs Confidence")
            plt.xlabel("Confidence Threshold")
            plt.ylabel(label)
            plt.ylim(0,1.02)
            plt.grid(True)
            plt.tight_layout()
            if SAVE_PLOTS:
                # ↓ comment out this line to disable saving
                plt.savefig(os.path.join(out_dir, fname))
            if SHOW_PLOTS:
                # ↓ comment out this line to disable interactive display
                plt.show()
            plt.close()

        # 5) Pick best threshold by F₁
        best_idx = int(np.argmax(F1_curve))
        best_thr = float(THRESHOLDS[best_idx])
        print(f"→ Best F₁ @ thr={best_thr:.3f}: P={P_curve[best_idx]:.3f}, "
              f"R={R_curve[best_idx]:.3f}, F₁={F1_curve[best_idx]:.3f}")

        eval_frames = tp_frames_map[best_thr]

        # # 6) Compute KF/PF errors
        # errors = {}
        # for tracker in ("KF","PF"):
        #     preds = load_predictions(pred_folder, tracker)
        #     rmse, mae = compute_tracker_errors(gt_pos, preds, eval_frames)
        #     errors[tracker] = (rmse, mae)
        #     print(f"   [{tracker}] on {len(eval_frames)} TP‑frames → "
        #           f"RMSE={rmse:.2f}px, MAE={mae:.2f}px")

        # # 7) Plot tracker errors
        # x      = np.arange(len(errors))
        # rmse_v = [errors[t][0] for t in errors]
        # mae_v  = [errors[t][1] for t in errors]
        # labels = list(errors.keys())
        # w      = 0.35

        # plt.figure(figsize=(6,4))
        # plt.bar(x - w/2, rmse_v, w, label="RMSE")
        # plt.bar(x + w/2, mae_v,  w, label="MAE")
        # plt.xticks(x, labels)
        # plt.ylabel("Error (px)")
        # plt.title(f"{clip_name}: Tracker Errors @ thr={best_thr:.3f}")
        # plt.legend()
        # plt.grid(axis='y')
        # plt.tight_layout()
        # if SAVE_PLOTS:
        #     # ↓ comment out this line to disable saving tracker-errors plot
        #     plt.savefig(os.path.join(out_dir, "tracker_errors.png"))
        # if SHOW_PLOTS:
        #     # ↓ comment out this line to disable interactive display
        #     plt.show()
        # plt.close()
