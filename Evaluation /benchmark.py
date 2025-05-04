import os
import time
import csv
import psutil
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO

# ─── Configuration ───────────────────────────────────────────────────────────
OUTPUT_DIR = "benchmarking"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join('runs','detect','train5','weights','best.pt')
VIDEOS = {
    "Fast-Bowling": ("videos/test2.mp4", "ground_truthv2"),
    "Spin-Bowling": ("videos/test3.mp4", "ground_truthv3"),
}
TRACKERS = ["KF", "PF"]
# ────────────────────────────────────────────────────────────────────────────

def get_mem():
    return psutil.Process(os.getpid()).memory_info().rss / 1e6

def initialize_kf():
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
    kf.transitionMatrix    = np.array([[1,0,1,0],
                                       [0,1,0,1],
                                       [0,0,1,0],
                                       [0,0,0,1]], np.float32)
    kf.processNoiseCov     = np.eye(4, dtype=np.float32)*0.03
    return kf

class PF:
    def __init__(self, N, pos, fs):
        self.p = np.ones((N,2)) * pos
        self.w = np.ones(N)/N
        self.fs = fs
    def predict(self, σ=10):
        self.p += np.random.randn(*self.p.shape)*σ
        self.p[:,0] = np.clip(self.p[:,0], 0, self.fs[1])
        self.p[:,1] = np.clip(self.p[:,1], 0, self.fs[0])
    def update(self, m, σ=15):
        d = np.linalg.norm(self.p - m, axis=1)
        self.w = np.exp(-d**2/(2*σ**2)) + 1e-8
        self.w /= self.w.sum()
    def resample(self):
        idx = np.random.choice(len(self.p), len(self.p), p=self.w)
        self.p = self.p[idx]
        self.w = np.ones(len(self.p)) / len(self.p)
    def estimate(self):
        return np.average(self.p, axis=0, weights=self.w).astype(int)

def run_tracker(video_path, mode):
    model = YOLO(MODEL_PATH)
    start_time = time.time()
    frames, mem, pts, times = 0, [], [], []
    kf = None
    pf = None

    for res in model.track(source=video_path, stream=True, conf=0.35, verbose=False):
        t = time.time() - start_time
        times.append(t)
        frames += 1
        mem.append(get_mem())

        # detect ball centroid
        bx = res.boxes
        cx = cy = None
        if bx and len(bx.cls) > 0:
            cls = bx.cls.cpu().numpy().astype(int)
            for i, c in enumerate(cls):
                if c == 0:
                    x1,y1,x2,y2 = map(int, bx.xyxy[i].tolist())
                    cx, cy = ((x1+x2)//2, (y1+y2)//2)
                    break

        if cx is None:
            pts.append(None)
            continue

        if mode == "KF":
            if kf is None:
                kf = initialize_kf()
            meas = np.array([[np.float32(cx)], [np.float32(cy)]])
            kf.correct(meas)
            p = kf.predict()
            pts.append((int(p[0,0]), int(p[1,0])))
        else:
            if pf is None:
                pf = PF(300, np.array([cx,cy]), res.orig_shape)
            pf.predict()
            pf.update(np.array([cx,cy]))
            pf.resample()
            e = pf.estimate()
            pts.append((int(e[0]), int(e[1])))

    fps = frames / (time.time() - start_time) if frames else 0
    return frames, np.array(mem), pts, fps, np.array(times)

def load_ground_truth(clip_key):
    video_path, gt_dir = VIDEOS[clip_key]
    cap = cv2.VideoCapture(video_path)
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()

    arr = [None]*N
    for fn in os.listdir(gt_dir):
        if fn.endswith('.json'):
            fid = int(fn.split('_')[1]) - 1
            data = json.load(open(os.path.join(gt_dir, fn)))
            if isinstance(data, list) and data:
                arr[fid] = (data[0]['center_x'], data[0]['center_y'])
    return arr

def compute_errors(gt, pts):
    mae=[]; mse=[]
    for g,p in zip(gt, pts):
        if g and p:
            dx,dy = p[0]-g[0], p[1]-g[1]
            mae.append(np.hypot(dx,dy))
            mse.append(dx*dx+dy*dy)
        else:
            mae.append(np.nan)
            mse.append(np.nan)
    return np.array(mae), np.array(mse)

def compute_acceleration(pts):
    vel=[None]
    for i in range(1,len(pts)):
        if pts[i] and pts[i-1]:
            vel.append((pts[i][0]-pts[i-1][0], pts[i][1]-pts[i-1][1]))
        else:
            vel.append(None)
    acc=[np.nan]
    for i in range(1,len(vel)):
        if vel[i] and vel[i-1]:
            ax = vel[i][0] - vel[i-1][0]
            ay = vel[i][1] - vel[i-1][1]
            acc.append(np.hypot(ax,ay))
        else:
            acc.append(np.nan)
    return np.array(acc)

def save_csv(name, header, rows):
    path = os.path.join(OUTPUT_DIR, name)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print("Wrote", path)

def plot_combined(x1,y1,x2,y2,labels,xlabel,ylabel,title,filename,colors):
    fig,ax = plt.subplots(figsize=(8,4))
    ax.plot(x1,y1, color=colors[0], label=labels[0], linewidth=2)
    ax.plot(x2,y2, color=colors[1], label=labels[1], linewidth=2)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path)
    plt.close(fig)
    print("Saved", path)

# ─── Main ────────────────────────────────────────────────────────────────────
summary_rows = []

for clip in VIDEOS:
    print("Processing:", clip)
    gt = load_ground_truth(clip)

    # collect KF & PF data
    results = {}
    for trk in TRACKERS:
        frames, mem, pts, fps, times = run_tracker(VIDEOS[clip][0], trk)
        mae, mse = compute_errors(gt, pts)
        acc      = compute_acceleration(pts)
        results[trk] = {
            "frames": frames,
            "mem": mem,
            "mae": mae,
            "mse": mse,
            "acc": acc,
            "times": times,
            "fps": fps
        }

    # 1. Acceleration vs Time
    mask_kf = ~np.isnan(results["KF"]["acc"])
    mask_pf = ~np.isnan(results["PF"]["acc"])
    plot_combined(
        results["KF"]["times"][mask_kf],
        results["KF"]["acc"][mask_kf],
        results["PF"]["times"][mask_pf],
        results["PF"]["acc"][mask_pf],
        ["Kalman Filter","Particle Filter"],
        "Time (s)", "Acceleration (px/frame²)",
        f"Acceleration of Tracking Algorithms Over Time\n{clip}",
        f"{clip}_acceleration_vs_time.png",
        ["C0","C1"]
    )

    # 2. MAE per Frame
    idx = np.arange(1, results["KF"]["frames"]+1)
    mask_kf = ~np.isnan(results["KF"]["mae"])
    mask_pf = ~np.isnan(results["PF"]["mae"])
    plot_combined(
        idx[mask_kf],
        results["KF"]["mae"][mask_kf],
        idx[mask_pf],
        results["PF"]["mae"][mask_pf],
        ["Kalman Filter","Particle Filter"],
        "Frame", "MAE (px)",
        f"Mean Absolute Error per Frame\n{clip}",
        f"{clip}_MAE_per_frame.png",
        ["C2","C3"]
    )

    # 3. MSE per Frame
    mask_kf = ~np.isnan(results["KF"]["mse"])
    mask_pf = ~np.isnan(results["PF"]["mse"])
    plot_combined(
        idx[mask_kf],
        results["KF"]["mse"][mask_kf],
        idx[mask_pf],
        results["PF"]["mse"][mask_pf],
        ["Kalman Filter","Particle Filter"],
        "Frame", "MSE (px²)",
        f"Mean Squared Error per Frame\n{clip}",
        f"{clip}_MSE_per_frame.png",
        ["C4","C5"]
    )

    # 4. Memory Usage Over Time
    plot_combined(
        idx,
        results["KF"]["mem"],
        idx,
        results["PF"]["mem"],
        ["Kalman Filter","Particle Filter"],
        "Frame", "Memory Usage (MB)",
        f"Memory Usage Over Time\n{clip}",
        f"{clip}_memory_over_time.png",
        ["C6","C7"]
    )

    # summary row
    valid = ~np.isnan(results["KF"]["mae"])
    gm_mae  = np.nanmean(results["KF"]["mae"][valid])
    gm_rmse = np.sqrt(np.nanmean(results["KF"]["mse"][valid]))
    summary_rows.append([clip, "Kalman Filter", f"{gm_rmse:.2f}", f"{gm_mae:.2f}", f"{results['KF']['fps']:.2f}"])
    valid = ~np.isnan(results["PF"]["mae"])
    pm_mae = np.nanmean(results["PF"]["mae"][valid])
    pm_rmse = np.sqrt(np.nanmean(results["PF"]["mse"][valid]))
    summary_rows.append([clip, "Particle Filter", f"{pm_rmse:.2f}", f"{pm_mae:.2f}", f"{results['PF']['fps']:.2f}"])

# write summary CSV
save_csv(
    "summary_metrics.csv",
    ["Clip","Tracker","Global_RMSE(px)","Global_MAE(px)","Average_FPS"],
    summary_rows
)

print("✅ Done—all plots and summary CSV in", OUTPUT_DIR)
