# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""V9-only epoch plot for detection and pose validation performance."""

from __future__ import annotations

import csv
import math
from pathlib import Path

from ultralytics.utils import RANK
from ultralytics.utils.torch_utils import unwrap_model


PLOT_METRICS = (
    ("metrics/mAP50(B)", "Detection mAP50", "#E63946"),
    ("metrics/mAP50-95(B)", "Detection mAP50–95", "#F4A261"),
    ("metrics/mAP50(P)", "Pose mAP50", "#2A9D8F"),
    ("metrics/mAP50-95(P)", "Pose mAP50–95", "#457B9D"),
)
TOP3_COLORS = ("#D4AF37", "#A7A7AD", "#CD7F32")


def render_pose_detection_performance(csv_path: str | Path, output_path: str | Path) -> Path | None:
    """Render a 2x2 metric dashboard from Ultralytics results.csv, annotating each metric's top three."""
    csv_path, output_path = Path(csv_path), Path(output_path)
    if not csv_path.exists():
        return None
    with csv_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows or not all(key in rows[0] for key, _, _ in PLOT_METRICS):
        return None

    import matplotlib.pyplot as plt

    epochs = [int(float(row["epoch"])) for row in rows]
    plt.style.use("fivethirtyeight")
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), constrained_layout=True)
    fig.patch.set_facecolor("#f7f7f7")
    fig.suptitle("YOLO26 Pose v9 — Detection & Pose Performance", fontsize=20, fontweight="bold")

    for ax, (key, title, line_color) in zip(axes.flat, PLOT_METRICS):
        values = [float(row[key]) for row in rows]
        ax.set_facecolor("#f7f7f7")
        ax.plot(epochs, values, color=line_color, linewidth=2.5, marker="o", markersize=4, label=title)
        candidates = [index for index, value in enumerate(values) if math.isfinite(value)]
        top3 = sorted(candidates, key=lambda index: (-values[index], epochs[index]))[:3]
        for rank, index in enumerate(top3, start=1):
            ax.scatter(
                epochs[index], values[index], s=130, color=TOP3_COLORS[rank - 1], edgecolor="#222831", zorder=5
            )
            ax.annotate(
                f"#{rank}  {values[index]:.4f}\nEpoch {epochs[index]}",
                (epochs[index], values[index]),
                xytext=(8, 10 + (rank - 1) * 15),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold",
                color="#222831",
                bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": TOP3_COLORS[rank - 1], "alpha": 0.9},
            )
        ax.set_title(title, color="#222831", fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("mAP")
        finite_max = max((values[index] for index in candidates), default=0.0)
        ax.set_ylim(bottom=0.0, top=max(1.0, finite_max * 1.12))
        ax.grid(True, linestyle="--", alpha=0.45, color="navy")
        ax.legend(facecolor="white", edgecolor="#222831")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
            spine.set_color("#161A1F")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


def plot_v9_performance_on_epoch_end(trainer) -> None:
    """Update the v9 dashboard after the current epoch metrics have been written."""
    if RANK not in {-1, 0} or not getattr(trainer, "csv", None):
        return
    model = unwrap_model(trainer.model)
    if hasattr(model, "student_model"):
        model = model.student_model
    head = model.model[-1]
    if type(head).__name__ != "OA26RegionRefinePose":
        return
    output = render_pose_detection_performance(
        trainer.csv, Path(trainer.save_dir) / "pose_detection_performance.png"
    )
    if output is not None:
        trainer.on_plot(output)
