# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""OA26 pose losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ultralytics.utils.loss import PoseLoss26


class OA26PoseLoss(PoseLoss26):
    """Pose26 loss plus heatmap, soft-coordinate, neighbour and curvature terms."""

    def __init__(self, model: torch.nn.Module, tal_topk: int = 10, tal_topk2: int | None = None):
        super().__init__(model, tal_topk, tal_topk2)
        cfg = getattr(model, "yaml", {}).get("oa26_loss", {})
        self.heatmap_gain = float(cfg.get("heatmap", 1.0))
        self.coord_gain = float(cfg.get("coord", 0.5))
        self.neighbour_gain = float(cfg.get("neighbour", 0.2))
        self.curve_gain = float(cfg.get("curve", 0.2))
        self.sigma = float(cfg.get("sigma", 1.5))

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate base pose loss and optional OA26 auxiliary losses."""
        base_loss, base_detach = super().loss(preds, batch)
        if "heatmaps" not in preds or "hm_kpts" not in preds:
            return base_loss, base_detach

        batch_size = preds["heatmaps"].shape[0]
        aux = self.auxiliary_loss(preds, batch)
        total = torch.cat((base_loss, aux * batch_size))
        detach = torch.cat((base_detach, aux.detach()))
        return total, detach

    def auxiliary_loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute heatmap, coordinate, neighbour and curvature auxiliary losses."""
        heatmaps = preds["heatmaps"]
        pred_xy = preds["hm_kpts"]
        gt_xy, valid = self._image_level_keypoints(batch, heatmaps)

        loss = torch.zeros(4, device=self.device, dtype=heatmaps.dtype)
        if not valid.any():
            return loss

        target_heatmaps = self._make_heatmap_targets(gt_xy, valid, heatmaps.shape[-2:])
        loss[0] = F.mse_loss(heatmaps.sigmoid(), target_heatmaps) * self.heatmap_gain

        pred_visible = pred_xy[valid]
        gt_visible = gt_xy[valid]
        loss[1] = F.smooth_l1_loss(pred_visible, gt_visible, beta=2.0) * self.coord_gain

        neighbour_mask = valid[:, 1:] & valid[:, :-1]
        if neighbour_mask.any():
            pred_vec = pred_xy[:, 1:] - pred_xy[:, :-1]
            gt_vec = gt_xy[:, 1:] - gt_xy[:, :-1]
            loss[2] = F.smooth_l1_loss(pred_vec[neighbour_mask], gt_vec[neighbour_mask], beta=2.0)
            loss[2] *= self.neighbour_gain

        curve_mask = valid[:, 2:] & valid[:, 1:-1] & valid[:, :-2]
        if curve_mask.any():
            pred_curve = pred_xy[:, :-2] - 2 * pred_xy[:, 1:-1] + pred_xy[:, 2:]
            gt_curve = gt_xy[:, :-2] - 2 * gt_xy[:, 1:-1] + gt_xy[:, 2:]
            loss[3] = F.smooth_l1_loss(pred_curve[curve_mask], gt_curve[curve_mask], beta=2.0)
            loss[3] *= self.curve_gain

        return loss

    def _image_level_keypoints(
        self, batch: dict[str, torch.Tensor], heatmaps: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build one landmark set per image for the auxiliary heatmap branch."""
        b, k, h, w = heatmaps.shape
        keypoints = batch["keypoints"].to(self.device).float().clone()
        batch_idx = batch["batch_idx"].to(self.device).long().flatten()
        imgsz = torch.tensor([h * 4.0, w * 4.0], device=self.device, dtype=heatmaps.dtype)
        gt_xy = torch.zeros(b, k, 2, device=self.device, dtype=heatmaps.dtype)
        valid = torch.zeros(b, k, device=self.device, dtype=torch.bool)

        if keypoints.numel() == 0:
            return gt_xy, valid

        keypoints[..., 0] *= imgsz[1]
        keypoints[..., 1] *= imgsz[0]
        n = min(k, keypoints.shape[1])
        for image_i in range(b):
            object_ids = (batch_idx == image_i).nonzero(as_tuple=False).flatten()
            if object_ids.numel() == 0:
                continue
            sample = keypoints[object_ids[0], :n]
            gt_xy[image_i, :n] = sample[:, :2]
            valid[image_i, :n] = sample[:, 2] != 0 if sample.shape[-1] == 3 else True
        return gt_xy, valid

    def _make_heatmap_targets(self, gt_xy: torch.Tensor, valid: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
        """Generate Gaussian heatmap targets for visible landmarks."""
        h, w = hw
        b, k = valid.shape
        dtype = gt_xy.dtype
        ys = torch.arange(h, device=self.device, dtype=dtype)
        xs = torch.arange(w, device=self.device, dtype=dtype)
        y_grid, x_grid = torch.meshgrid(ys, xs, indexing="ij")
        x0 = (gt_xy[..., 0] / 4.0).clamp(0, w - 1).view(b, k, 1, 1)
        y0 = (gt_xy[..., 1] / 4.0).clamp(0, h - 1).view(b, k, 1, 1)
        dist = (x_grid.view(1, 1, h, w) - x0).pow(2) + (y_grid.view(1, 1, h, w) - y0).pow(2)
        target = torch.exp(-dist / (2 * self.sigma**2))
        return target * valid.view(b, k, 1, 1).to(dtype)
