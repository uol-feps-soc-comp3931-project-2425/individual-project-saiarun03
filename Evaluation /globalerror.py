import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ─── Configuration ───────────────────────────────────────────────────────────
SUMMARY_CSV = "benchmarking/summary_metrics.csv"
OUTPUT_DIR  = "benchmarking"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ────────────────────────────────────────────────────────────────────────────

# Read the summary CSV
df = pd.read_csv(SUMMARY_CSV)

# Pivot for RMSE and MAE
rmse_df = df.pivot(index="Clip", columns="Tracker", values="Global_RMSE(px)")
mae_df  = df.pivot(index="Clip", columns="Tracker", values="Global_MAE(px)")

# Common plotting routine
def plot_grouped_bar(data: pd.DataFrame, ylabel: str, title: str, out_fname: str, colors):
    clips = data.index.tolist()
    trackers = data.columns.tolist()  # should be ['Kalman Filter','Particle Filter']
    x = np.arange(len(clips))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(x - width/2, data[trackers[0]], width, label=trackers[0], color=colors[0])
    ax.bar(x + width/2, data[trackers[1]], width, label=trackers[1], color=colors[1])

    ax.set_xticks(x)
    ax.set_xticklabels(clips, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    fig.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, out_fname)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")

# Plot Global Average MAE
plot_grouped_bar(
    mae_df,
    ylabel="Global MAE (px)",
    title="Global Average MAE by Tracker and Video",
    out_fname="global_average_mae.png",
    colors=["#4F81BD", "#F28E2B"],
)

# Plot Global Average RMSE
plot_grouped_bar(
    rmse_df,
    ylabel="Global RMSE (px)",
    title="Global Average RMSE by Tracker and Video",
    out_fname="global_average_rmse.png",
    colors=["#C0504D", "#9BBB59"],
)
