from __future__ import annotations

import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    """
    Cross-stream attention fusion (CS-MHA, minimal).

    Query: RGB stream features
    Key/Value: Flow stream features
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        rgb_seq: torch.Tensor,
        flow_seq: torch.Tensor,
        flow_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        rgb_seq: [B, T, d_model]
        flow_seq: [B, T, d_model]
        flow_padding_mask: [B, T] bool, True means padding positions
        return: [B, T, d_model]
        """
        attn_out, _ = self.attn(
            query=rgb_seq,
            key=flow_seq,
            value=flow_seq,
            key_padding_mask=flow_padding_mask,
            need_weights=False,
        )
        fused = self.norm(rgb_seq + self.dropout(attn_out))
        return fused

