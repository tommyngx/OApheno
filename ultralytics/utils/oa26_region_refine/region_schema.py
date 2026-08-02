# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""MESKO4GF2 class-local landmark schema used by OA26 region refinement."""

from __future__ import annotations

import torch


# Verified against Reference/yolo_mesko4GF2/data.yaml, summary.json, and all 442 training labels.
REGION_NAMES: tuple[str, ...] = ("femur", "tibia", "fibula", "patella")
REGION_KEYPOINT_COUNTS: tuple[int, ...] = (45, 51, 24, 9)
SOURCE_KEYPOINT_RANGES: tuple[tuple[int, int], ...] = ((0, 44), (45, 95), (96, 119), (120, 128))
NUM_REGIONS = len(REGION_NAMES)
MAX_REGION_KEYPOINTS = 51


def validate_region_schema(num_classes: int = 4, kpt_shape: tuple[int, int] = (51, 3)) -> None:
    """Validate the four-class, padded-51-keypoint MESKO label contract."""
    if num_classes != NUM_REGIONS:
        raise ValueError(f"MESKO region schema requires nc={NUM_REGIONS}, got {num_classes}")
    if tuple(kpt_shape) != (MAX_REGION_KEYPOINTS, 3):
        raise ValueError(f"MESKO region schema requires kpt_shape=[{MAX_REGION_KEYPOINTS}, 3], got {kpt_shape}")
    if any(count > MAX_REGION_KEYPOINTS for count in REGION_KEYPOINT_COUNTS):
        raise ValueError("A region keypoint count exceeds the padded keypoint tensor")


def class_keypoint_mask(class_ids: torch.Tensor, max_keypoints: int = MAX_REGION_KEYPOINTS) -> torch.Tensor:
    """Return a boolean local-keypoint mask for each region class ID."""
    counts = class_ids.new_tensor(REGION_KEYPOINT_COUNTS)
    safe_ids = class_ids.long().clamp(0, NUM_REGIONS - 1)
    local_ids = torch.arange(max_keypoints, device=class_ids.device).view(1, -1)
    return local_ids < counts[safe_ids].view(-1, 1)


validate_region_schema()
