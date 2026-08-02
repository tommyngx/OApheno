# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
import tempfile
from pathlib import Path
import csv

from ultralytics.utils.oa26_region_refine.training_plot_v9 import (
    render_v9_training_dashboard,
    render_pose_detection_performance,
)

def test_render_v9_training_dashboard():
    with tempfile.TemporaryDirectory() as directory:
        csv_path = Path(directory) / "results.csv"
        headers = [
            "epoch", "time",
            "train/box_loss", "train/pose_loss",
            "train/hm_loss", "train/hm_coord_loss",
            "metrics/mAP50(B)", "metrics/mAP50-95(B)",
            "metrics/mAP50(P)", "metrics/mAP50-95(P)",
        ]
        
        rows = [
            [1, 10, 0.50, 0.40, 0.30, 0.25, 0.40, 0.20, 0.30, 0.10],
            [2, 20, 0.40, 0.30, 0.22, 0.18, 0.55, 0.32, 0.45, 0.22],
            [3, 30, 0.35, 0.25, 0.19, 0.14, 0.65, 0.40, 0.52, 0.29],
            [4, 40, 0.30, 0.20, 0.15, 0.10, 0.70, 0.48, 0.60, 0.38],
            [5, 50, 0.28, 0.18, 0.12, 0.08, 0.68, 0.46, 0.58, 0.36],
        ]
        
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)
            
        out_file = Path(directory) / "dashboard_v9.png"
        res = render_v9_training_dashboard(csv_path, out_file)
        
        assert res is not None
        assert res.exists()
        assert res.stat().st_size > 10_000
        
        # Test backward compatible wrapper
        out_compat = Path(directory) / "compat.png"
        res_compat = render_pose_detection_performance(csv_path, out_compat)
        assert res_compat is not None
        assert res_compat.exists()

if __name__ == "__main__":
    test_render_v9_training_dashboard()
    print("SUCCESS: test_render_v9_training_dashboard passed!")
