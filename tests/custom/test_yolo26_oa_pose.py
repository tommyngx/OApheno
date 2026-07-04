# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Smoke tests for experimental YOLO26 OA pose variants."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ultralytics.cfg import get_cfg
from ultralytics.nn.modules.oa26 import OA26HeatmapPose, OA26SimCCPose
from ultralytics.nn.tasks import PoseModel


CFG_DIR = ROOT / "ultralytics/cfg/models/26oa"
KPT_SHAPE = (129, 3)


def _make_model(name: str) -> PoseModel:
    cfg = CFG_DIR / name
    model = PoseModel(str(cfg), ch=3, nc=1, data_kpt_shape=KPT_SHAPE, verbose=False)
    model.args = get_cfg(overrides={"task": "pose", "mode": "train", "model": str(cfg), "data": "coco8-pose.yaml"})
    return model


def _synthetic_batch() -> dict[str, torch.Tensor]:
    keypoints = torch.zeros(1, KPT_SHAPE[0], KPT_SHAPE[1])
    keypoints[..., 0] = torch.linspace(0.2, 0.8, KPT_SHAPE[0])
    keypoints[..., 1] = 0.5
    keypoints[..., 2] = 1
    keypoints[:, ::5, 2] = 0
    return {
        "batch_idx": torch.tensor([0.0]),
        "cls": torch.tensor([0.0]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.5, 0.5]]),
        "keypoints": keypoints,
    }


def test_build_all_oa26_pose_models():
    models = {
        "yolo26-posev1.yaml": OA26HeatmapPose,
        "yolo26-posev2.yaml": OA26HeatmapPose,
        "yolo26-posev3.yaml": OA26SimCCPose,
    }
    for name, head_type in models.items():
        model = _make_model(name)
        assert isinstance(model.model[-1], head_type)
        assert tuple(model.model[-1].kpt_shape) == KPT_SHAPE


def test_forward_outputs_keep_standard_pose_and_auxiliary():
    x = torch.randn(1, 3, 64, 64)
    for name in ("yolo26-posev1.yaml", "yolo26-posev2.yaml", "yolo26-posev3.yaml"):
        model = _make_model(name)
        model.train()
        preds = model(x)
        branch = preds["one2many"]
        assert branch["kpts"].shape[1] == KPT_SHAPE[0] * KPT_SHAPE[1]
        if isinstance(model.model[-1], OA26HeatmapPose):
            assert {"heatmaps", "hm_kpts"}.issubset(branch)
            assert branch["heatmaps"].shape[1] == KPT_SHAPE[0]
        else:
            assert {"simcc_x", "simcc_y", "simcc_kpts"}.issubset(branch)
            assert branch["simcc_kpts"].shape == (1, KPT_SHAPE[0], 3)

        model.eval()
        with torch.no_grad():
            y, _ = model(x)
        assert y.shape[-1] == 6 + KPT_SHAPE[0] * KPT_SHAPE[1]


def test_loss_runs_for_129_keypoints():
    batch = _synthetic_batch()
    x = torch.randn(1, 3, 64, 64)
    for name in ("yolo26-posev1.yaml", "yolo26-posev2.yaml", "yolo26-posev3.yaml"):
        model = _make_model(name)
        model.train()
        preds = model(x)
        loss, detached = model.loss(batch, preds)
        assert loss.shape == detached.shape
        assert torch.isfinite(loss).all()


def test_simcc_logits_and_decode_shapes():
    model = _make_model("yolo26-posev3.yaml")
    model.train()
    preds = model(torch.randn(1, 3, 64, 64))["one2many"]
    assert preds["simcc_x"].shape == (1, KPT_SHAPE[0], 1792)
    assert preds["simcc_y"].shape == (1, KPT_SHAPE[0], 1792)
    assert preds["simcc_kpts"].shape == (1, KPT_SHAPE[0], 3)


if __name__ == "__main__":
    test_build_all_oa26_pose_models()
    test_forward_outputs_keep_standard_pose_and_auxiliary()
    test_loss_runs_for_129_keypoints()
    test_simcc_logits_and_decode_shapes()
