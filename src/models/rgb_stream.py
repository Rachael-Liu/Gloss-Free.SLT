from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small


def _infer_backbone_out_channels(backbone_features: nn.Module) -> int:
    # Try to find the last Conv2d out_channels in the features stack.
    last = None
    for m in reversed(list(backbone_features.modules())):
        if isinstance(m, nn.Conv2d):
            last = m.out_channels
            break
    if last is None:
        raise RuntimeError("Could not infer out_channels from backbone features")
    return int(last)


class RGBStream(nn.Module):
    """
    RGB static feature stream.

    Minimal version:
    - MobileNetV3 feature extractor (per-frame)
    - Linear projection to d_model
    - BiLSTM over time
    """

    def __init__(
        self,
        d_model: int = 256,
        lstm_dropout: float = 0.1,
        backbone_out_channels: Optional[int] = None,
    ) -> None:
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError("d_model must be even for BiLSTM output")

        backbone = mobilenet_v3_small(weights=None)
        self.features = backbone.features
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        feat_ch = backbone_out_channels or _infer_backbone_out_channels(self.features)
        self.proj = nn.Sequential(
            nn.Linear(feat_ch, d_model),
            nn.Dropout(lstm_dropout),
        )

        lstm_hidden = d_model // 2
        self.bilstm = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        rgb: [B, T, 3, 224, 224]
        return: [B, T, d_model]
        """
        b, t, c, h, w = rgb.shape
        x = rgb.view(b * t, c, h, w)
        x = self.features(x)
        x = self.avg_pool(x).flatten(1)  # [B*T, feat_ch]
        x = self.proj(x)  # [B*T, d_model]
        x = x.view(b, t, -1)  # [B, T, d_model]
        x, _ = self.bilstm(x)
        x = self.out_norm(x)
        return x

