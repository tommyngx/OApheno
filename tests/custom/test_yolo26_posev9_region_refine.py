# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Tests for YOLO26 OA pose v9 class-local region refinement."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ultralytics.cfg import get_cfg
from ultralytics.nn.modules.oa26_region_refine import (
    OA26RegionLocalizationHead,
    OA26RegionRefinePose,
    OA26RegionROIExtractor,
    OA26RegionTransformer,
)
from ultralytics.nn.tasks import PoseModel
from ultralytics.utils.oa26_region_refine import (
    NUM_REGIONS,
    REGION_KEYPOINT_COUNTS,
    class_keypoint_mask,
    render_pose_detection_performance,
    validate_region_schema,
)


V1_CFG = ROOT / "ultralytics/cfg/models/26oa/yolo26-posev1.yaml"
V9_CFG = ROOT / "ultralytics/cfg/models/26oa/yolo26-posev9.yaml"
KPT_SHAPE = (51, 3)
NC = 4


def _make_model(path: Path = V9_CFG) -> PoseModel:
    model = PoseModel(str(path), ch=3, nc=NC, data_kpt_shape=KPT_SHAPE, verbose=False)
    model.args = get_cfg(
        overrides={"task": "pose", "mode": "train", "model": str(path), "data": "coco8-pose.yaml"}
    )
    return model


def _synthetic_batch() -> dict[str, torch.Tensor]:
    keypoints = torch.zeros(NC, *KPT_SHAPE)
    for class_id, count in enumerate(REGION_KEYPOINT_COUNTS):
        keypoints[class_id, :count, 0] = torch.linspace(0.2, 0.8, count)
        keypoints[class_id, :count, 1] = 0.35 + class_id * 0.1
        keypoints[class_id, :count, 2] = 1
    return {
        "batch_idx": torch.zeros(NC),
        "cls": torch.arange(NC, dtype=torch.float32).view(-1, 1),
        "bboxes": torch.tensor([[0.5, 0.5, 0.65, 0.5]]).repeat(NC, 1),
        "keypoints": keypoints,
    }


def test_region_schema_matches_mesko4gf2_classes():
    validate_region_schema(NC, KPT_SHAPE)
    mask = class_keypoint_mask(torch.arange(NC))
    assert mask.shape == (NC, KPT_SHAPE[0])
    assert tuple(mask.sum(1).tolist()) == REGION_KEYPOINT_COUNTS == (45, 51, 24, 9)


def test_direct_v9_model_loss_initializes_default_args():
    model = PoseModel(str(V9_CFG), ch=3, nc=NC, data_kpt_shape=KPT_SHAPE, verbose=False).train()
    assert not hasattr(model, "args")
    predictions = model(torch.randn(1, 3, 64, 64))
    loss, detached = model.loss(_synthetic_batch(), predictions)
    assert hasattr(model.criterion, "one2many") and hasattr(model.criterion, "one2one")
    assert model.criterion.one2many.hyp.task == "pose" and hasattr(model, "args")
    assert loss.shape == (14,) and detached.shape == (5,) and torch.isfinite(loss).all()
    loss.sum().backward()


def test_roi_align_shape_boundary_and_empty_input():
    extractor = OA26RegionROIExtractor(16, d_model=24, output_size=(12, 10)).eval()
    feature = torch.randn(2, 16, 16, 20, requires_grad=True)
    boxes = torch.tensor([[0.0, 0.0, 40.0, 48.0], [30.0, 20.0, 79.0, 63.0]])
    output = extractor(feature, boxes, torch.tensor([0, 1]), spatial_scale=0.25)
    assert output.shape == (2, 24, 12, 10)
    output.sum().backward()
    assert feature.grad is not None and torch.isfinite(feature.grad).all()
    empty = extractor(feature.detach(), boxes[:0], torch.empty(0, dtype=torch.long), spatial_scale=0.25)
    assert empty.shape == (0, 24, 12, 10)

    amp_feature = torch.randn(1, 16, 16, 20, requires_grad=True)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        amp_output = extractor(amp_feature, boxes[:1], torch.tensor([0]), spatial_scale=0.25)
    assert amp_output.dtype == torch.bfloat16
    amp_output.float().mean().backward()
    assert amp_feature.grad is not None and torch.isfinite(amp_feature.grad).all()


def test_tiny_thop_feature_map_bypasses_native_roi_align():
    extractor = OA26RegionROIExtractor(16, d_model=24, output_size=(20, 20)).eval()
    feature = torch.randn(1, 16, 2, 2, requires_grad=True)
    boxes = torch.tensor([[0.0, 0.0, 8.0, 8.0]]).repeat(NC, 1)
    with patch("ultralytics.nn.modules.oa26_region_refine.roi_feature_extractor.roi_align") as native_roi:
        output = extractor(feature, boxes, torch.zeros(NC, dtype=torch.long), spatial_scale=0.25)
    native_roi.assert_not_called()
    assert output.shape == (NC, 24, 20, 20)
    output.mean().backward()
    assert feature.grad is not None and torch.isfinite(feature.grad).all()


def test_eager_roi_align_fallback_signature_and_backward():
    """Cover the eager path used when Torchvision would otherwise invoke hidden Triton compilation."""
    extractor = OA26RegionROIExtractor(8, d_model=8, output_size=(4, 4)).eval()
    feature = torch.randn(1, 8, 8, 8, requires_grad=True)
    boxes = torch.tensor([[0.0, 0.0, 28.0, 28.0]])
    with (
        patch("ultralytics.nn.modules.oa26_region_refine.roi_feature_extractor._has_ops", return_value=False),
        patch("ultralytics.nn.modules.oa26_region_refine.roi_feature_extractor.roi_align") as lazy_roi,
    ):
        output = extractor(feature, boxes, torch.zeros(1, dtype=torch.long), spatial_scale=0.25)
    lazy_roi.assert_not_called()
    assert output.shape == (1, 8, 4, 4)
    output.mean().backward()
    assert feature.grad is not None and torch.isfinite(feature.grad).all()


def test_refinement_detaches_predicted_roi_coordinates():
    model = _make_model().train()
    head = model.model[-1].region_refine_head
    boxes = torch.tensor(
        [[2.0, 2.0, 30.0, 30.0], [3.0, 3.0, 31.0, 31.0], [4.0, 4.0, 32.0, 32.0], [5.0, 5.0, 33.0, 33.0]],
        requires_grad=True,
    )
    coarse = torch.rand(NC, *KPT_SHAPE, requires_grad=True)
    output = head(
        torch.randn(1, 128, 4, 4, requires_grad=True),
        boxes,
        coarse,
        torch.arange(NC),
        torch.zeros(NC, dtype=torch.long),
        torch.arange(NC).view(1, NC),
        (torch.tensor(64.0), torch.tensor(64.0)),
        16.0,
    )
    assert not output["region_boxes"].requires_grad
    output["refined_region_kpts"].sum().backward()
    assert boxes.grad is None
    assert coarse.grad is not None


def test_transformer_does_not_mix_region_instances():
    transformer = OA26RegionTransformer(d_model=24, num_heads=4, num_layers=2, dropout=0.0).eval()
    queries = torch.randn(2, 5, 24)
    image_tokens = torch.randn(2, 16, 24)
    valid = torch.ones(2, 5, dtype=torch.bool)
    with torch.no_grad():
        baseline = transformer(queries, image_tokens, valid)
        changed_queries, changed_image = queries.clone(), image_tokens.clone()
        changed_queries[1].add_(100)
        changed_image[1].mul_(20)
        changed = transformer(changed_queries, changed_image, valid)
    assert torch.allclose(baseline[0], changed[0], atol=1e-6, rtol=1e-6)


def test_roi_localization_probability_coordinate_and_gradient():
    head = OA26RegionLocalizationHead(d_model=24, temperature=0.2)
    landmarks = torch.randn(2, 5, 24, requires_grad=True)
    image_tokens = torch.randn(2, 12 * 10, 24, requires_grad=True)
    valid = torch.ones(2, 5, dtype=torch.bool)
    _, probability, xy = head(landmarks, image_tokens, (12, 10), torch.rand(2, 5, 2), valid)
    assert probability.shape == (2, 5, 12, 10)
    assert torch.allclose(probability.sum(dim=(-2, -1)), torch.ones(2, 5), atol=1e-5)
    assert ((xy >= 0) & (xy <= 1)).all()
    xy.sum().backward()
    assert image_tokens.grad is not None and image_tokens.grad.abs().sum() > 0


def test_v9_build_forward_loss_and_backward():
    model = _make_model().train()
    assert isinstance(model.model[-1], OA26RegionRefinePose)
    predictions = model(torch.randn(1, 3, 64, 64))
    branch = predictions["one2many"]
    assert branch["coarse_region_kpts"].shape == (NC, *KPT_SHAPE)
    assert branch["refined_region_kpts"].shape == (NC, *KPT_SHAPE)
    assert branch["region_boxes"].shape == (NC, 4)
    assert branch["region_heatmaps"].shape == (NC, KPT_SHAPE[0], 20, 20)
    assert tuple(branch["region_valid_mask"].sum(1).tolist()) == REGION_KEYPOINT_COUNTS
    valid_probability = branch["region_heatmaps"].sum(dim=(-2, -1))[branch["region_valid_mask"]]
    assert torch.allclose(valid_probability, torch.ones_like(valid_probability), atol=1e-5)
    loss, detached = model.loss(_synthetic_batch(), predictions)
    assert loss.shape == (14,) and detached.shape == (5,)
    assert torch.isfinite(loss).all()
    loss.sum().backward()
    gradient = model.model[-1].region_refine_head.localization_head.query_projection.weight.grad
    assert gradient is not None and gradient.abs().sum() > 0


def test_v9_public_inference_layout_and_v1_checkpoint_compatibility():
    model = _make_model().eval()
    with torch.no_grad():
        predictions, raw = model(torch.randn(1, 3, 64, 64))
    assert raw["one2one"]["refined_region_kpts"].shape == (NC, *KPT_SHAPE)
    assert predictions.shape[-1] == 6 + KPT_SHAPE[0] * KPT_SHAPE[1] == 159

    v1 = _make_model(V1_CFG)
    result = model.load_state_dict(v1.state_dict(), strict=False)
    assert not result.unexpected_keys
    assert result.missing_keys and all("region_refine_head" in key for key in result.missing_keys)


def test_v9_postprocess_selects_pose_for_returned_class():
    head = _make_model().model[-1]
    anchors = NC
    boxes = torch.arange(anchors * 4, dtype=torch.float32).view(1, anchors, 4)
    scores = torch.full((1, anchors, NC), -10.0)
    scores[0, torch.arange(NC), torch.arange(NC)] = torch.arange(NC, dtype=torch.float32) + 1
    coarse = torch.zeros(1, anchors, head.nk)
    refined = torch.stack(
        [torch.full(KPT_SHAPE, float(class_id + 1)) for class_id in range(NC)]
    )
    raw = {
        "refined_region_kpts": refined,
        "region_selected_anchor_indices": torch.arange(NC).view(1, NC),
    }
    output = head._postprocess_refined(torch.cat((boxes, scores, coarse), dim=-1), raw)
    assert output.shape == (1, anchors, 6 + head.nk)
    for detection in output[0]:
        class_id = int(detection[5].item())
        assert torch.all(detection[6:] == float(class_id + 1))


def test_epoch_dashboard_contains_four_metrics():
    with tempfile.TemporaryDirectory() as directory:
        csv_path = Path(directory) / "results.csv"
        csv_path.write_text(
            "epoch,time,metrics/mAP50(B),metrics/mAP50-95(B),metrics/mAP50(P),metrics/mAP50-95(P)\n"
            "1,1,0.40,0.20,0.30,0.10\n2,2,0.50,0.25,0.45,0.22\n3,3,0.48,0.24,0.43,0.21\n",
            encoding="utf-8",
        )
        output = render_pose_detection_performance(csv_path, Path(directory) / "pose_detection_performance.png")
        assert output is not None and output.is_file() and output.stat().st_size > 10_000


if __name__ == "__main__":
    test_region_schema_matches_mesko4gf2_classes()
    test_direct_v9_model_loss_initializes_default_args()
    test_roi_align_shape_boundary_and_empty_input()
    test_tiny_thop_feature_map_bypasses_native_roi_align()
    test_eager_roi_align_fallback_signature_and_backward()
    test_refinement_detaches_predicted_roi_coordinates()
    test_transformer_does_not_mix_region_instances()
    test_roi_localization_probability_coordinate_and_gradient()
    test_v9_build_forward_loss_and_backward()
    test_v9_public_inference_layout_and_v1_checkpoint_compatibility()
    test_v9_postprocess_selects_pose_for_returned_class()
    test_epoch_dashboard_contains_four_metrics()
