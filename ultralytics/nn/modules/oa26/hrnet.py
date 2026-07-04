# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Clean-room HRNet-style backbone adapter for OA26 pose experiments."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    """Convolution, batch normalization and ReLU activation."""

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1, act: bool = True):
        """Initialize a convolutional block."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, padding=k // 2, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution, normalization and activation."""
        return self.act(self.bn(self.conv(x)))


class BasicBlock(nn.Module):
    """Minimal residual block used by the HRNet-style branches."""

    def __init__(self, channels: int):
        """Initialize the residual block."""
        super().__init__()
        self.cv1 = ConvBNAct(channels, channels, 3, 1)
        self.cv2 = ConvBNAct(channels, channels, 3, 1, act=False)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a residual block."""
        return self.act(x + self.cv2(self.cv1(x)))


class HRFusionStage(nn.Module):
    """Parallel high-resolution branches with repeated multi-scale fusion."""

    def __init__(self, channels: tuple[int, int, int, int], num_blocks: int = 1):
        """Initialize one HRNet-style fusion stage."""
        super().__init__()
        self.channels = channels
        self.branches = nn.ModuleList(
            nn.Sequential(*(BasicBlock(c) for _ in range(num_blocks))) for c in channels
        )
        self.fuse = nn.ModuleList(
            nn.ModuleList(self._make_fuse_layer(j, i) for j in range(len(channels))) for i in range(len(channels))
        )
        self.act = nn.ReLU(inplace=True)

    def _make_fuse_layer(self, source: int, target: int) -> nn.Module:
        """Create source-to-target resolution/channel conversion."""
        if source == target:
            return nn.Identity()
        if source > target:
            return ConvBNAct(self.channels[source], self.channels[target], 1, 1, act=False)

        layers = []
        in_channels = self.channels[source]
        for step in range(target - source):
            out_channels = self.channels[target] if step == target - source - 1 else in_channels
            layers.append(ConvBNAct(in_channels, out_channels, 3, 2, act=step != target - source - 1))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        """Fuse the four parallel branches."""
        x = [branch(xi) for branch, xi in zip(self.branches, x)]
        fused = []
        for target, target_tensor in enumerate(x):
            y = self.fuse[target][target](target_tensor)
            for source, source_tensor in enumerate(x):
                if source == target:
                    continue
                z = self.fuse[target][source](source_tensor)
                if source > target:
                    z = F.interpolate(z, size=target_tensor.shape[-2:], mode="nearest")
                y = y + z
            fused.append(self.act(y))
        return fused


class HRNet(nn.Module):
    """HRNet-style W32 backbone that returns P2/P3/P4/P5 features."""

    _VARIANTS = {"w32": (32, 64, 128, 256)}

    def __init__(
        self,
        variant: str = "w32",
        pretrained: bool = False,
        out_channels: tuple[int, int, int, int] | list[int] = (128, 256, 512, 512),
        return_p2: bool = True,
    ):
        """Initialize the HRNet-style backbone adapter."""
        super().__init__()
        variant = str(variant).lower()
        if variant not in self._VARIANTS:
            raise ValueError(
                f"Unsupported HRNet variant '{variant}'. Supported variants: {self._VARIANTS}."
            )
        if pretrained:
            raise NotImplementedError(
                "HRNet pretrained weights are not bundled for this clean-room adapter."
            )

        self.variant = variant
        self.return_p2 = bool(return_p2)
        branch_channels = self._VARIANTS[variant]
        out_channels = tuple(int(c) for c in out_channels)
        if len(out_channels) != 4:
            raise ValueError("out_channels must contain four values for P2, P3, P4 and P5.")

        self.stem = nn.Sequential(
            ConvBNAct(3, 64, 3, 2),  # stride 2
            ConvBNAct(64, 64, 3, 2),  # stride 4
            BasicBlock(64),
            BasicBlock(64),
        )
        self.transition = nn.ModuleList(
            (
                ConvBNAct(64, branch_channels[0], 3, 1),
                ConvBNAct(64, branch_channels[1], 3, 2),
                nn.Sequential(
                    ConvBNAct(64, branch_channels[1], 3, 2),
                    ConvBNAct(branch_channels[1], branch_channels[2], 3, 2),
                ),
                nn.Sequential(
                    ConvBNAct(64, branch_channels[1], 3, 2),
                    ConvBNAct(branch_channels[1], branch_channels[2], 3, 2),
                    ConvBNAct(branch_channels[2], branch_channels[3], 3, 2),
                ),
            )
        )
        self.stages = nn.ModuleList(HRFusionStage(branch_channels, num_blocks=1) for _ in range(3))
        self.adapters = nn.ModuleList(ConvBNAct(c1, c2, 1, 1) for c1, c2 in zip(branch_channels, out_channels))

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return `[P2, P3, P4, P5]` feature maps for the YOLO neck."""
        x = self.stem(x)  # B x 64 x H/4 x W/4
        features = [transition(x) for transition in self.transition]
        for stage in self.stages:
            features = stage(features)

        p2, p3, p4, p5 = [adapter(feature) for adapter, feature in zip(self.adapters, features)]
        # For 896x896 input: P2=224x224, P3=112x112, P4=56x56, P5=28x28.
        return [p2, p3, p4, p5] if self.return_p2 else [p3, p4, p5]
