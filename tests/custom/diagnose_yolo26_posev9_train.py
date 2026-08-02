# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Run one real v9 Trainer job in a child process with crash-surviving stage markers."""

from __future__ import annotations

import argparse
import faulthandler
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.models.yolo.pose.train import PoseTrainer
from ultralytics.utils.oa26_region_refine.debug import debug_event


class V9DebugPoseTrainer(PoseTrainer):
    """Add outer Trainer markers without changing the normal trainer implementation."""

    def setup_model(self):
        debug_event("trainer-setup-model-enter")
        result = super().setup_model()
        debug_event("trainer-setup-model-complete")
        return result

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        debug_event("trainer-dataloader-enter", mode=mode, batch=batch_size)
        result = super().get_dataloader(dataset_path, batch_size, rank, mode)
        debug_event("trainer-dataloader-complete", mode=mode, batches=len(result))
        return result

    def build_optimizer(self, *args, **kwargs):
        debug_event("trainer-optimizer-build-enter")
        result = super().build_optimizer(*args, **kwargs)
        debug_event("trainer-optimizer-build-complete")
        return result

    def get_validator(self):
        debug_event("trainer-validator-build-enter")
        result = super().get_validator()
        debug_event("trainer-validator-build-complete")
        return result

    def preprocess_batch(self, batch):
        debug_event("trainer-preprocess-enter", images=tuple(batch["img"].shape))
        result = super().preprocess_batch(batch)
        debug_event("trainer-preprocess-complete", images=tuple(result["img"].shape))
        return result

    def optimizer_step(self):
        debug_event("trainer-optimizer-step-enter")
        result = super().optimizer_step()
        debug_event("trainer-optimizer-step-complete")
        return result

    def validate(self):
        debug_event("trainer-validation-enter")
        result = super().validate()
        debug_event("trainer-validation-complete")
        return result


def callback(stage):
    """Create an Ultralytics callback that writes one flushed marker."""
    return lambda trainer: debug_event(stage, epoch=getattr(trainer, "epoch", -1))


def main() -> None:
    """Launch a minimal real-data train in this standalone process."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "Reference/yolo_mesko4GF2/data.yaml"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--device", default="0")
    parser.add_argument("--name", default="trace")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--val", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    run_dir = ROOT / "runs/v9_debug" / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_V9_DEBUG"] = "1"
    os.environ["YOLO_V9_DEBUG_SYNC"] = "1"
    os.environ["YOLO_V9_DEBUG_FILE"] = str(run_dir / "v9_debug.log")
    faulthandler.enable(all_threads=True)
    debug_event("process-start", python=sys.version.split()[0], root=ROOT)

    model = YOLO(str(ROOT / "ultralytics/cfg/models/26oa/yolo26-posev9.yaml"))
    for event in (
        "on_pretrain_routine_start",
        "on_pretrain_routine_end",
        "on_train_start",
        "on_train_epoch_start",
        "on_train_batch_start",
        "on_train_batch_end",
        "on_train_epoch_end",
        "on_val_start",
        "on_val_batch_start",
        "on_val_batch_end",
        "on_val_end",
        "on_fit_epoch_end",
        "on_model_save",
        "on_train_end",
        "teardown",
    ):
        model.add_callback(event, callback(f"callback-{event}"))
    debug_event("yolo-train-enter")
    model.train(
        trainer=V9DebugPoseTrainer,
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        amp=args.amp,
        val=args.val,
        plots=args.plots,
        project=str(ROOT / "runs/v9_debug"),
        name=args.name,
        exist_ok=True,
        cache=False,
    )
    debug_event("process-complete")


if __name__ == "__main__":
    main()
