# Script to generate a realistic sample V9 dashboard plot with 50 epochs of random synthetic training data
import csv
import math
import random
from pathlib import Path

from ultralytics.utils.oa26_region_refine.training_plot_v9 import render_v9_training_dashboard

def main():
    random.seed(42)
    output_dir = Path("/Users/francistommy/Desktop/BugHunter/Project/OApheno/tests/custom")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = output_dir / "sample_results.csv"
    png_path = output_dir / "sample_v9_plot.png"

    headers = [
        "epoch", "time",
        "train/box_loss", "train/pose_loss",
        "train/hm_loss", "train/hm_coord_loss",
        "metrics/mAP50(B)", "metrics/mAP50-95(B)",
        "metrics/mAP50(P)", "metrics/mAP50-95(P)",
    ]

    rows = []
    num_epochs = 50
    for ep in range(1, num_epochs + 1):
        # Synthetic loss curves (decreasing)
        box_loss = 0.75 * math.exp(-ep / 18.0) + 0.08 + random.gauss(0, 0.008)
        pose_loss = 0.55 * math.exp(-ep / 16.0) + 0.05 + random.gauss(0, 0.006)
        hm_loss = 0.35 * math.exp(-ep / 14.0) + 0.03 + random.gauss(0, 0.004)
        hm_coord_loss = 0.25 * math.exp(-ep / 12.0) + 0.02 + random.gauss(0, 0.003)

        # Synthetic mAP curves (increasing)
        det_map50 = 0.86 / (1.0 + math.exp(-(ep - 14) / 5.0)) + random.gauss(0, 0.007)
        det_map50_95 = 0.63 / (1.0 + math.exp(-(ep - 17) / 5.0)) + random.gauss(0, 0.006)
        pose_map50 = 0.89 / (1.0 + math.exp(-(ep - 12) / 5.0)) + random.gauss(0, 0.007)
        pose_map50_95 = 0.71 / (1.0 + math.exp(-(ep - 15) / 5.0)) + random.gauss(0, 0.006)

        rows.append([
            ep,
            ep * 12.5,
            max(0.001, box_loss),
            max(0.001, pose_loss),
            max(0.001, hm_loss),
            max(0.001, hm_coord_loss),
            min(1.0, max(0.0, det_map50)),
            min(1.0, max(0.0, det_map50_95)),
            min(1.0, max(0.0, pose_map50)),
            min(1.0, max(0.0, pose_map50_95)),
        ])

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    result_path = render_v9_training_dashboard(csv_path, png_path)
    print(f"Sample dashboard saved to: {result_path}")

if __name__ == "__main__":
    main()
