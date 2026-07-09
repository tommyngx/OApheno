# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Smoke tests for YOLO26 OA pose v4 HRNet backbone."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.nn.modules.head import Pose26
from ultralytics.nn.modules.oa26 import HRNetLite, OA26HeatmapPose, OA26SimCCPose
from ultralytics.nn.tasks import PoseModel


V4_CFG = ROOT / "ultralytics/cfg/models/26oa/yolo26-posev4.yaml"
ORIGINAL_CFG = ROOT / "ultralytics/cfg/models/26/yolo26-pose.yaml"
KPT_SHAPE = (129, 3)


def test_yolo26_posev4_hrnet_builds_from_yaml():
    model = YOLO(str(V4_CFG), task="pose")
    head = model.model.model[-1]
    assert isinstance(head, Pose26)
    assert not isinstance(head, (OA26HeatmapPose, OA26SimCCPose))
    assert tuple(head.kpt_shape) == KPT_SHAPE


def test_hrnet_lite_backbone_outputs_expected_896_shapes():
    backbone = HRNetLite().eval()
    with torch.no_grad():
        outputs = backbone(torch.randn(1, 3, 896, 896))
    assert [tuple(x.shape) for x in outputs] == [
        (1, 128, 224, 224),
        (1, 256, 112, 112),
        (1, 512, 56, 56),
        (1, 512, 28, 28),
    ]


def test_yolo26_posev4_forward_keeps_standard_pose_output():
    model = PoseModel(str(V4_CFG), ch=3, nc=1, data_kpt_shape=KPT_SHAPE, verbose=False).eval()
    with torch.no_grad():
        y, raw = model(torch.randn(1, 3, 128, 128))
    assert y.shape[-1] == 6 + KPT_SHAPE[0] * KPT_SHAPE[1]
    assert isinstance(raw, dict)


def test_original_yolo26_pose_still_builds():
    model = PoseModel(str(ORIGINAL_CFG), ch=3, nc=1, data_kpt_shape=(17, 3), verbose=False)
    assert isinstance(model.model[-1], Pose26)


def test_optional_slow_896_forward():
    if os.getenv("OA26_SLOW_HRNET") != "1":
        return
    model = PoseModel(str(V4_CFG), ch=3, nc=1, data_kpt_shape=KPT_SHAPE, verbose=False).eval()
    with torch.no_grad():
        y, _ = model(torch.randn(1, 3, 896, 896))
    assert y.shape[-1] == 6 + KPT_SHAPE[0] * KPT_SHAPE[1]


if __name__ == "__main__":
    test_yolo26_posev4_hrnet_builds_from_yaml()
    test_hrnet_lite_backbone_outputs_expected_896_shapes()
    test_yolo26_posev4_forward_keeps_standard_pose_output()
    test_original_yolo26_pose_still_builds()
    test_optional_slow_896_forward()
