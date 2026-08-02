# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Dedicated Pose26 integration for per-class MESKO4GF2 region refinement."""

from __future__ import annotations

import copy

import torch

from ultralytics.nn.modules.oa26.pose_heads import OA26HeatmapPose
from ultralytics.utils.oa26_region_refine.region_schema import (
    MAX_REGION_KEYPOINTS,
    NUM_REGIONS,
    validate_region_schema,
)
from ultralytics.utils.tal import make_anchors

from .refinement_head import OA26PerRegionRefinementHead


class OA26RegionRefinePose(OA26HeatmapPose):
    """Keep v1 paths and add an isolated P4 refiner for each of the four detected bones."""

    def __init__(
        self,
        nc: int = NUM_REGIONS,
        kpt_shape: tuple = (MAX_REGION_KEYPOINTS, 3),
        region_config: dict | None = None,
        reg_max: int = 1,
        end2end: bool = False,
        ch: tuple = (),
    ):
        """Initialize the unchanged v1 head and the v9-only region branch."""
        validate_region_schema(nc, kpt_shape)
        cfg = region_config or {}
        super().__init__(
            nc,
            kpt_shape,
            0,
            float(cfg.get("coarse_heatmap_temperature", 1.0)),
            reg_max,
            end2end,
            ch,
        )
        self.region_refine_enabled = bool(cfg.get("enabled", True))
        if str(cfg.get("feature_level", "P4")).upper() != "P4":
            raise ValueError("OA26 region refinement currently supports feature_level=P4 only")
        if int(cfg.get("num_regions", NUM_REGIONS)) != NUM_REGIONS:
            raise ValueError(f"MESKO4GF2 defines exactly {NUM_REGIONS} region classes")
        if len(ch) < 3:
            raise ValueError("OA26RegionRefinePose requires P2, P3, and P4 feature levels")
        self.region_refine_head = OA26PerRegionRefinementHead(
            in_channels=ch[2],
            num_classes=nc,
            kpt_shape=kpt_shape,
            d_model=int(cfg.get("d_model", 192)),
            num_heads=int(cfg.get("num_heads", 6)),
            num_layers=int(cfg.get("num_layers", 3)),
            roi_output_size=tuple(cfg.get("roi_output_size", (24, 24))),
            roi_sampling_ratio=int(cfg.get("roi_sampling_ratio", 2)),
            roi_padding=float(cfg.get("roi_padding", 0.25)),
            min_roi_size_px=float(cfg.get("min_roi_size_px", 48.0)),
            heatmap_temperature=float(cfg.get("heatmap_temperature", 0.1)),
            use_coarse_prior=bool(cfg.get("use_coarse_spatial_prior", False)),
            coarse_prior_sigma=float(cfg.get("coarse_prior_sigma", 0.25)),
            coarse_prior_gain=float(cfg.get("coarse_prior_gain", 0.5)),
            dropout=float(cfg.get("dropout", 0.1)),
        )
        if end2end:
            self.one2one_region_refine_head = copy.deepcopy(self.region_refine_head)

    @property
    def one2many(self):
        """Return v1 one-to-many modules plus the separate v9 refiner."""
        heads = super().one2many
        heads["region_refine_head"] = self.region_refine_head
        return heads

    @property
    def one2one(self):
        """Return v1 one-to-one modules plus the separate v9 refiner."""
        heads = super().one2one
        heads["region_refine_head"] = self.one2one_region_refine_head
        return heads

    def _select_class_instances(
        self,
        x: list[torch.Tensor],
        boxes: torch.Tensor,
        scores: torch.Tensor,
        raw_kpts: torch.Tensor,
        image_size: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select and decode the highest-confidence anchor for every class in every image."""
        batch_size = boxes.shape[0]
        image_h, image_w = image_size
        selected = scores.sigmoid().argmax(dim=2)  # B x C; one region instance per MESKO class.
        batch_grid = torch.arange(batch_size, device=boxes.device)[:, None].expand(-1, self.nc)
        class_grid = torch.arange(self.nc, device=boxes.device)[None].expand(batch_size, -1)

        if bool((self.stride > 0).all()):
            anchors, stride_tensor = make_anchors(x, self.stride, 0.5)
            decoded_boxes = self.decode_bboxes(self.dfl(boxes), anchors.t().unsqueeze(0)) * stride_tensor.t()
            raw = raw_kpts.view(batch_size, *self.kpt_shape, -1).permute(0, 3, 1, 2)
            decoded_xy = (raw[..., :2] + anchors[None, :, None]) * stride_tensor[None, :, None]
            decoded_conf = raw[..., 2:3].sigmoid()
            decoded_kpts = torch.cat((decoded_xy, decoded_conf), dim=-1)
            instance_boxes = decoded_boxes.transpose(1, 2)[batch_grid, selected]
            coarse_kpts = decoded_kpts[batch_grid, selected]
        else:
            # PoseModel runs a stride-discovery forward before prediction heads are calibrated.
            full = boxes.new_tensor((0.0, 0.0, float(image_w), float(image_h)))
            instance_boxes = full.view(1, 1, 4).expand(batch_size, self.nc, -1)
            coarse_kpts = raw_kpts.new_zeros((batch_size, self.nc, *self.kpt_shape))
            coarse_kpts[..., 0] = image_w * 0.5
            coarse_kpts[..., 1] = image_h * 0.5
            coarse_kpts[..., 2] = 0.5

        instance_boxes = instance_boxes.reshape(-1, 4)
        x1 = instance_boxes[:, 0].clamp(0, image_w)
        y1 = instance_boxes[:, 1].clamp(0, image_h)
        x2 = instance_boxes[:, 2].clamp(0, image_w)
        y2 = instance_boxes[:, 3].clamp(0, image_h)
        x2 = torch.maximum(x2, x1 + 2.0).clamp(max=image_w)
        y2 = torch.maximum(y2, y1 + 2.0).clamp(max=image_h)
        instance_boxes = torch.stack((x1, y1, x2, y2), dim=-1)
        return (
            instance_boxes,
            coarse_kpts.reshape(-1, *self.kpt_shape),
            class_grid.reshape(-1),
            batch_grid.reshape(-1),
            selected,
        )

    def forward_head(
        self,
        x: list[torch.Tensor],
        box_head: torch.nn.Module,
        cls_head: torch.nn.Module,
        pose_head: torch.nn.Module,
        kpts_head: torch.nn.Module,
        kpts_sigma_head: torch.nn.Module,
        heatmap_head: torch.nn.Module | None = None,
        region_refine_head: torch.nn.Module | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return standard v1 predictions plus one independent refinement row per class."""
        preds = super().forward_head(x, box_head, cls_head, pose_head, kpts_head, kpts_sigma_head, heatmap_head)
        if not self.region_refine_enabled or region_refine_head is None or "kpts" not in preds:
            return preds

        stride0 = self.stride[0].to(device=x[0].device, dtype=x[0].dtype)
        stride0 = torch.where(stride0 > 0, stride0, stride0.new_tensor(4.0))
        image_h = x[0].new_tensor(float(x[0].shape[-2])) * stride0
        image_w = x[0].new_tensor(float(x[0].shape[-1])) * stride0
        boxes, coarse, class_ids, batch_ids, selected = self._select_class_instances(
            x, preds["boxes"], preds["scores"], preds["kpts"], (image_h, image_w)
        )
        p4_stride = self.stride[2] if self.stride[2] > 0 else self.stride.new_tensor(16.0)
        preds.update(
            region_refine_head(
                x[2], boxes, coarse, class_ids, batch_ids, selected, (image_h, image_w), p4_stride
            )
        )
        return preds

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Append internal class-specific poses while retaining the normal Pose26 channel prefix."""
        preds = super()._inference(x)
        if "refined_region_kpts" not in x:
            return preds
        base_channels = 4 + self.nc
        coarse = preds[:, base_channels:]
        batch_size, _, num_anchors = coarse.shape
        class_kpts = coarse[:, None].expand(-1, self.nc, -1, -1).clone()
        refined = x["refined_region_kpts"].reshape(batch_size, self.nc, self.nk)
        selected = x["region_selected_anchor_indices"]
        scatter_index = selected[:, :, None, None].expand(-1, -1, self.nk, 1)
        class_kpts.scatter_(3, scatter_index, refined.unsqueeze(-1))
        return torch.cat((preds, class_kpts.reshape(batch_size, self.nc * self.nk, num_anchors)), dim=1)

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        """Return the exact v1 public layout, selecting refinement by detection class."""
        standard = 4 + self.nc + self.nk
        if preds.shape[-1] == standard:
            return super().postprocess(preds)
        boxes, scores, _, class_kpts = preds.split([4, self.nc, self.nk, self.nc * self.nk], dim=-1)
        scores, classes, indices = self.get_topk_index(scores, self.max_det)
        boxes = boxes.gather(1, indices.expand(-1, -1, 4))
        batch_size, detections = indices.shape[:2]
        class_kpts = class_kpts.view(batch_size, -1, self.nc, self.nk)
        class_kpts = class_kpts.gather(
            1, indices[:, :, None, :].expand(-1, -1, self.nc, self.nk)
        )
        class_index = classes.long()[:, :, None].expand(-1, -1, 1, self.nk)
        kpts = class_kpts.gather(2, class_index).reshape(batch_size, detections, self.nk)
        return torch.cat((boxes, scores, classes, kpts), dim=-1)

    def fuse(self) -> None:
        """Discard only training-time one-to-many v9 modules during inference fusion."""
        super().fuse()
        self.region_refine_head = None
