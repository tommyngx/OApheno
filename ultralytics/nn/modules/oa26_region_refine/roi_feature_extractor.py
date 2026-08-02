# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""P4 ROI feature extraction for OA26 per-region refinement."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.ops import roi_align

from ultralytics.nn.modules.conv import Conv


class OA26RegionROIExtractor(nn.Module):
    """Extract a fixed-resolution projected feature map for every anatomical region ROI."""

    def __init__(
        self,
        in_channels: int,
        d_model: int = 192,
        output_size: tuple[int, int] = (24, 24),
        sampling_ratio: int = 2,
        aligned: bool = True,
    ):
        """Initialize the feature projection and ROIAlign settings."""
        super().__init__()
        self.output_size = tuple(int(value) for value in output_size)
        self.sampling_ratio = int(sampling_ratio)
        self.aligned = bool(aligned)
        self.projection = nn.Sequential(Conv(in_channels, d_model, 1), Conv(d_model, d_model, 3))

    def forward(
        self,
        feature: torch.Tensor,
        boxes: torch.Tensor,
        batch_indices: torch.Tensor,
        spatial_scale: float,
    ) -> torch.Tensor:
        """Return M x d_model x Hroi x Wroi features, including a safe empty-ROI path."""
        projected = self.projection(feature)
        if boxes.numel() == 0:
            return projected.new_empty((0, projected.shape[1], *self.output_size))
        rois = torch.cat((batch_indices.to(boxes.dtype).unsqueeze(1), boxes), dim=1)
        return roi_align(
            projected,
            rois,
            output_size=self.output_size,
            spatial_scale=float(spatial_scale),
            sampling_ratio=self.sampling_ratio,
            aligned=self.aligned,
        )
