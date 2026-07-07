# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Smoke tests for YOLO26 OA pose v6 ViTRefine."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.nn.modules.head import Pose26
from ultralytics.nn.modules.oa26 import OA26HeatmapPose, OA26SimCCPose, ViTRefine
from ultralytics.nn.tasks import PoseModel


V6_CFG = ROOT / "ultralytics/cfg/models/26oa/yolo26-posev6.yaml"
V2_CFG = ROOT / "ultralytics/cfg/models/26oa/yolo26-posev2.yaml"
ORIGINAL_CFG = ROOT / "ultralytics/cfg/models/26/yolo26-pose.yaml"
KPT_SHAPE = (129, 3)


def test_vitrefine_preserves_p4_shape():
    block = ViTRefine(512, embed_dim=64, num_heads=4, depth=1).eval()
    x = torch.randn(1, 512, 56, 56)
    with torch.no_grad():
        y = block(x)
    assert tuple(y.shape) == tuple(x.shape)


def test_vitrefine_max_tokens_pooling_path_preserves_shape():
    block = ViTRefine(128, embed_dim=32, num_heads=4, depth=1, max_tokens=64).eval()
    x = torch.randn(1, 128, 20, 20)
    with torch.no_grad():
        y = block(x)
    assert tuple(y.shape) == tuple(x.shape)


def test_yolo26_posev6_builds_from_yaml():
    model = YOLO(str(V6_CFG), task="pose")
    head = model.model.model[-1]
    assert isinstance(head, Pose26)
    assert not isinstance(head, (OA26HeatmapPose, OA26SimCCPose))
    assert tuple(head.kpt_shape) == KPT_SHAPE


def test_yolo26_posev6_forward_keeps_standard_pose_output():
    model = PoseModel(str(V6_CFG), ch=3, nc=1, data_kpt_shape=KPT_SHAPE, verbose=False).eval()
    with torch.no_grad():
        y, raw = model(torch.randn(1, 3, 128, 128))
    assert y.shape[-1] == 6 + KPT_SHAPE[0] * KPT_SHAPE[1]
    assert isinstance(raw, dict)


def test_existing_pose_yamls_still_build():
    v2 = PoseModel(str(V2_CFG), ch=3, nc=1, data_kpt_shape=KPT_SHAPE, verbose=False)
    original = PoseModel(str(ORIGINAL_CFG), ch=3, nc=1, data_kpt_shape=(17, 3), verbose=False)
    assert isinstance(v2.model[-1], OA26HeatmapPose)
    assert isinstance(original.model[-1], Pose26)


if __name__ == "__main__":
    test_vitrefine_preserves_p4_shape()
    test_vitrefine_max_tokens_pooling_path_preserves_shape()
    test_yolo26_posev6_builds_from_yaml()
    test_yolo26_posev6_forward_keeps_standard_pose_output()
    test_existing_pose_yamls_still_build()
