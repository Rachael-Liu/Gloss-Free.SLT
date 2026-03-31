from __future__ import annotations

import torch
import torch.nn as nn

from .rgb_stream import RGBStream
from .flow_stream import FlowStream
from .cross_attention import CrossAttentionFusion
from .decoder import TransformerTextDecoder


class OFTSNet(nn.Module):
    """
    Minimal OFTS-Net:
    - RGB stream encoder
    - Flow stream encoder
    - Cross attention fusion
    - Transformer text decoder
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        pad_id: int = 0,
        max_len: int = 512,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model

        self.rgb_stream = RGBStream(d_model=d_model, lstm_dropout=dropout)
        self.flow_stream = FlowStream(d_model=d_model, cnn_dropout=dropout)
        self.fusion = CrossAttentionFusion(d_model=d_model, nhead=nhead, dropout=dropout)
        self.decoder = TransformerTextDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            pad_id=pad_id,
            max_len=max_len,
        )

    def forward(
        self,
        rgb: torch.Tensor,
        flow: torch.Tensor,
        tgt_input: torch.Tensor,
        src_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        rgb: [B, T, 3, 224, 224]
        flow: [B, T, 2, 224, 224]
        tgt_input: [B, L] (shifted tokens)
        src_lengths: [B] (unpadded lengths, in [1..T])
        return: logits [B, L, vocab]
        """
        b, t, _, _, _ = rgb.shape

        rgb_seq = self.rgb_stream(rgb)   # [B, T, d_model]
        flow_seq = self.flow_stream(flow)  # [B, T, d_model]

        # Padding mask: True means padding.
        ar = torch.arange(t, device=src_lengths.device)[None, :]  # [1, T]
        memory_padding_mask = ar >= src_lengths[:, None]  # [B, T]

        fused = self.fusion(rgb_seq, flow_seq, flow_padding_mask=memory_padding_mask)

        tgt_padding_mask = tgt_input.eq(self.pad_id)  # [B, L]
        logits = self.decoder(
            tgt_input=tgt_input,
            memory=fused,
            tgt_padding_mask=tgt_padding_mask,
            memory_padding_mask=memory_padding_mask,
        )
        return logits

