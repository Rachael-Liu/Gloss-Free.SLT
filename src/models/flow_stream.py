from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small


def _infer_backbone_out_channels(backbone_features: nn.Module) -> int:
    last = None
    for m in reversed(list(backbone_features.modules())):
        if isinstance(m, nn.Conv2d):
            last = m.out_channels
            break
    if last is None:
        raise RuntimeError("Could not infer out_channels from backbone features")
    return int(last)


class FlowStream(nn.Module):
    """
    Optical-flow dynamic feature stream.

    Minimal version:
    - MobileNetV3 feature extractor (per-frame) but flow is 2-channel
      so we pad a zero channel to make it 3-channel.
    - Linear projection to d_model
    - 1D CNN over time
    """

    def __init__(
        self,
        d_model: int = 256,
        cnn_dropout: float = 0.1,
        backbone_out_channels: Optional[int] = None,
    ) -> None:
        super().__init__()

        backbone = mobilenet_v3_small(weights=None)
        self.features = backbone.features
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        feat_ch = backbone_out_channels or _infer_backbone_out_channels(self.features)
        self.proj = nn.Sequential(
            nn.Linear(feat_ch, d_model),
            nn.Dropout(cnn_dropout),
        )

        # 1D conv over time: input [B, d_model, T]
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(cnn_dropout),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
        )
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, flow: torch.Tensor) -> torch.Tensor:
        """
        flow: [B, T, 2, 224, 224]
        return: [B, T, d_model]
        """
        b, t, c, h, w = flow.shape
        if c != 2:
            raise ValueError(f"Expected flow channel=2, got {c}")

        x = flow.view(b * t, c, h, w)  # [B*T, 2, H, W]
        zero = torch.zeros((b * t, 1, h, w), dtype=x.dtype, device=x.device)
        x = torch.cat([x, zero], dim=1)  # [B*T, 3, H, W]

        x = self.features(x)  # [B*T, feat_ch, h', w']
        x = self.avg_pool(x).flatten(1)  # [B*T, feat_ch]
        x = self.proj(x)  # [B*T, d_model]
        x = x.view(b, t, -1)  # [B, T, d_model]

        # Temporal conv expects channels-first.
        x = x.permute(0, 2, 1).contiguous()  # [B, d_model, T]
        x = self.temporal_conv(x)
        x = x.permute(0, 2, 1).contiguous()  # [B, T, d_model]
        x = self.out_norm(x)
        return x

