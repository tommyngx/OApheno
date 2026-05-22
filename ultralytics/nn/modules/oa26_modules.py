# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""OA26 custom pose modules."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from .conv import Conv
from .head import Pose26


class OA26HeatmapPose(Pose26):
    """YOLO26 pose head with an auxiliary stride-4 heatmap and soft-argmax decoder."""

    def __init__(
        self,
        nc: int = 80,
        kpt_shape: tuple = (128, 3),
        heatmap_channels: int = 128,
        heatmap_temperature: float = 1.0,
        reg_max: int = 1,
        end2end: bool = False,
        ch: tuple = (),
    ):
        super().__init__(nc, kpt_shape, reg_max, end2end, ch)
        self.heatmap_channels = int(heatmap_channels or kpt_shape[0])
        self.heatmap_temperature = float(heatmap_temperature)
        self.heatmap_stride = 4.0

        c = max(ch[0] // 2, self.heatmap_channels)
        self.hm_head = nn.Sequential(
            Conv(ch[0], c, 3),
            Conv(c, c, 3),
            nn.Conv2d(c, self.heatmap_channels, 1),
        )
        if end2end:
            self.one2one_hm_head = copy.deepcopy(self.hm_head)

    @property
    def one2many(self):
        heads = super().one2many
        heads["heatmap_head"] = self.hm_head
        return heads

    @property
    def one2one(self):
        heads = super().one2one
        heads["heatmap_head"] = self.one2one_hm_head
        return heads

    def forward_head(
        self,
        x: list[torch.Tensor],
        box_head: torch.nn.Module,
        cls_head: torch.nn.Module,
        pose_head: torch.nn.Module,
        kpts_head: torch.nn.Module,
        kpts_sigma_head: torch.nn.Module,
        heatmap_head: torch.nn.Module | None = None,
    ) -> dict[str, torch.Tensor]:
        """Concatenate detection outputs and add auxiliary heatmap predictions."""
        preds = super().forward_head(x, box_head, cls_head, pose_head, kpts_head, kpts_sigma_head)
        if heatmap_head is not None:
            heatmaps = heatmap_head(x[0])
            preds["heatmaps"] = heatmaps
            preds["hm_kpts"] = self.soft_argmax_2d(heatmaps)
        return preds

    def soft_argmax_2d(self, heatmaps: torch.Tensor) -> torch.Tensor:
        """Decode heatmaps to image-space xy coordinates with sub-pixel precision."""
        b, k, h, w = heatmaps.shape
        logits = heatmaps.view(b, k, -1) / max(self.heatmap_temperature, 1e-6)
        prob = logits.softmax(dim=-1)
        dtype, device = heatmaps.dtype, heatmaps.device
        xs = torch.arange(w, device=device, dtype=dtype)
        ys = torch.arange(h, device=device, dtype=dtype)
        y_grid, x_grid = torch.meshgrid(ys, xs, indexing="ij")
        x = (prob * x_grid.reshape(1, 1, -1)).sum(dim=-1)
        y = (prob * y_grid.reshape(1, 1, -1)).sum(dim=-1)
        return torch.stack((x, y), dim=-1) * self.heatmap_stride

    def fuse(self) -> None:
        """Remove one-to-many heads for end-to-end inference optimization."""
        super().fuse()
        self.hm_head = None
