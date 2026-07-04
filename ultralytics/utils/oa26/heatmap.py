# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Heatmap target helpers for OA26 pose experiments."""

from __future__ import annotations

import torch


def extract_image_keypoints(
    batch: dict[str, torch.Tensor],
    batch_size: int,
    num_keypoints: int,
    image_size: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one normalized-label landmark set per image as image-space xy plus visibility mask."""
    gt_xy = torch.zeros(batch_size, num_keypoints, 2, device=device, dtype=dtype)
    valid = torch.zeros(batch_size, num_keypoints, device=device, dtype=torch.bool)
    keypoints = batch.get("keypoints")
    batch_idx = batch.get("batch_idx")
    if keypoints is None or batch_idx is None or keypoints.numel() == 0:
        return gt_xy, valid

    keypoints = keypoints.to(device=device, dtype=dtype).clone()
    batch_idx = batch_idx.to(device=device).long().flatten()
    h, w = image_size.to(device=device, dtype=dtype)
    keypoints[..., 0] *= w
    keypoints[..., 1] *= h
    n = min(num_keypoints, keypoints.shape[1])

    for image_i in range(batch_size):
        object_ids = (batch_idx == image_i).nonzero(as_tuple=False).flatten()
        if object_ids.numel() == 0:
            continue
        sample = keypoints[object_ids[0], :n]
        gt_xy[image_i, :n] = sample[:, :2]
        valid[image_i, :n] = sample[:, 2] > 0 if sample.shape[-1] == 3 else True
    return gt_xy, valid


def gaussian_heatmap_targets(
    gt_xy: torch.Tensor,
    valid: torch.Tensor,
    heatmap_hw: tuple[int, int],
    image_size: torch.Tensor,
    sigma: float = 1.5,
) -> torch.Tensor:
    """Generate Gaussian heatmaps from image-space keypoints."""
    h, w = heatmap_hw
    b, k = valid.shape
    device, dtype = gt_xy.device, gt_xy.dtype
    image_h, image_w = image_size.to(device=device, dtype=dtype).clamp(min=1)
    ys = torch.arange(h, device=device, dtype=dtype)
    xs = torch.arange(w, device=device, dtype=dtype)
    y_grid, x_grid = torch.meshgrid(ys, xs, indexing="ij")

    x0 = (gt_xy[..., 0] * w / image_w).clamp(0, w - 1).view(b, k, 1, 1)
    y0 = (gt_xy[..., 1] * h / image_h).clamp(0, h - 1).view(b, k, 1, 1)
    dist = (x_grid.view(1, 1, h, w) - x0).pow(2) + (y_grid.view(1, 1, h, w) - y0).pow(2)
    target = torch.exp(-dist / (2 * max(float(sigma), 1e-6) ** 2))
    return target * valid.view(b, k, 1, 1).to(dtype)
