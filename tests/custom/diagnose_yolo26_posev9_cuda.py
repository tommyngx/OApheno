# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Standalone CUDA forward/loss/backward probe for diagnosing hard v9 notebook kernel exits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import PoseModel
from ultralytics.utils.oa26_region_refine import REGION_KEYPOINT_COUNTS


def report(stage: str) -> None:
    """Synchronize CUDA and print allocator state immediately so the last successful stage survives a crash."""
    torch.cuda.synchronize()
    gib = 1 << 30
    print(
        f"[{stage}] allocated={torch.cuda.memory_allocated() / gib:.3f} GiB, "
        f"reserved={torch.cuda.memory_reserved() / gib:.3f} GiB, "
        f"peak={torch.cuda.max_memory_allocated() / gib:.3f} GiB",
        flush=True,
    )


def synthetic_batch(batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
    """Build four MESKO class instances per image using the padded 51-point label contract."""
    count = batch_size * 4
    keypoints = torch.zeros(count, 51, 3, device=device)
    for image_id in range(batch_size):
        for class_id, valid_points in enumerate(REGION_KEYPOINT_COUNTS):
            row = image_id * 4 + class_id
            keypoints[row, :valid_points, 0] = torch.linspace(0.2, 0.8, valid_points, device=device)
            keypoints[row, :valid_points, 1] = 0.35 + class_id * 0.1
            keypoints[row, :valid_points, 2] = 1
    return {
        "batch_idx": torch.arange(batch_size, device=device).repeat_interleave(4).float(),
        "cls": torch.arange(4, device=device).repeat(batch_size).float().view(-1, 1),
        "bboxes": torch.tensor([[0.5, 0.5, 0.65, 0.5]], device=device).repeat(count, 1),
        "keypoints": keypoints,
    }


def main() -> None:
    """Run each training stage with synchronization and flushed allocator reports."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgsz", type=int, default=896)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this environment")

    device = torch.device("cuda:0")
    print(f"torch={torch.__version__}, cuda={torch.version.cuda}, gpu={torch.cuda.get_device_name(0)}", flush=True)
    import torchvision

    print(f"torchvision={torchvision.__version__}, source={ROOT}", flush=True)
    model = PoseModel(
        str(ROOT / "ultralytics/cfg/models/26oa/yolo26-posev9.yaml"),
        ch=3,
        nc=4,
        data_kpt_shape=(51, 3),
        verbose=False,
    ).to(device).train()
    model.args = get_cfg(overrides={"task": "pose", "mode": "train", "model": "v9", "data": "mesko"})
    report("model-built")

    target = synthetic_batch(args.batch, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=not args.no_amp)
    torch.cuda.reset_peak_memory_stats()
    detached = None
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        image = torch.randn(args.batch, 3, args.imgsz, args.imgsz, device=device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=not args.no_amp):
            predictions = model(image)
        report(f"step-{step}-forward")
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=not args.no_amp):
            loss, detached = model.loss(target, predictions)
        print(f"[step-{step}-loss] shape={tuple(loss.shape)}, finite={bool(torch.isfinite(loss).all())}", flush=True)
        scaler.scale(loss.sum()).backward()
        scaler.step(optimizer)
        scaler.update()
        report(f"step-{step}-complete")
        del image, predictions, loss
    print(f"PASS detached_loss_shape={tuple(detached.shape)}", flush=True)


if __name__ == "__main__":
    main()
