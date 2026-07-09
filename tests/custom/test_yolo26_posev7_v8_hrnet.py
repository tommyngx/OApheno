# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Smoke tests for canonical HRNet-W32/W48 YOLO26 OA pose variants."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ultralytics.nn.modules.head import Pose26
from ultralytics.nn.modules.oa26 import HRNet
from ultralytics.nn.tasks import PoseModel


KPT_SHAPE = (129, 3)
CONFIGS = (
    ("w32", ROOT / "ultralytics/cfg/models/26oa/yolo26-posev7.yaml"),
    ("w48", ROOT / "ultralytics/cfg/models/26oa/yolo26-posev8.yaml"),
)


def test_canonical_hrnet_outputs_expected_896_shapes():
    for variant, _ in CONFIGS:
        backbone = HRNet(variant).eval()
        with torch.no_grad():
            outputs = backbone(torch.randn(1, 3, 896, 896))
        assert [tuple(x.shape) for x in outputs] == [
            (1, 128, 224, 224),
            (1, 256, 112, 112),
            (1, 512, 56, 56),
            (1, 512, 28, 28),
        ]


def test_yolo26_canonical_hrnet_builds_and_keeps_pose26():
    for _, config in CONFIGS:
        model = PoseModel(str(config), ch=3, nc=1, data_kpt_shape=KPT_SHAPE, verbose=False).eval()
        assert isinstance(model.model[-1], Pose26)
        assert tuple(model.model[-1].kpt_shape) == KPT_SHAPE
        with torch.no_grad():
            y, raw = model(torch.randn(1, 3, 128, 128))
        assert y.shape[-1] == 6 + KPT_SHAPE[0] * KPT_SHAPE[1]
        assert isinstance(raw, dict)


def test_optional_slow_canonical_hrnet_896_forward():
    if os.getenv("OA26_SLOW_HRNET") != "1":
        return
    for _, config in CONFIGS:
        model = PoseModel(str(config), ch=3, nc=1, data_kpt_shape=KPT_SHAPE, verbose=False).eval()
        with torch.no_grad():
            y, _ = model(torch.randn(1, 3, 896, 896))
        assert y.shape[-1] == 6 + KPT_SHAPE[0] * KPT_SHAPE[1]


if __name__ == "__main__":
    test_canonical_hrnet_outputs_expected_896_shapes()
    test_yolo26_canonical_hrnet_builds_and_keeps_pose26()
    test_optional_slow_canonical_hrnet_896_forward()
