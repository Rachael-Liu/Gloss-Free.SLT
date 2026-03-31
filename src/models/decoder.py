from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, d_model]
        l = x.size(1)
        return x + self.pe[:, :l].to(x.dtype)


class TransformerTextDecoder(nn.Module):
    """
    Transformer decoder for gloss-free translation (minimal).

    Inputs:
    - tgt_input: [B, L] token ids (already shifted, includes <sos> at position 0)
    - memory: [B, S, d_model] fused features from the two-stream encoder
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        pad_id: int = 0,
        max_len: int = 512,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_len)
        self.dropout = nn.Dropout(dropout)

        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)

    @staticmethod
    def _generate_causal_mask(l: int, device: torch.device) -> torch.Tensor:
        # True means "masked" in PyTorch bool mask.
        return torch.triu(torch.ones(l, l, device=device, dtype=torch.bool), diagonal=1)

    def forward(
        self,
        tgt_input: torch.Tensor,
        memory: torch.Tensor,
        tgt_padding_mask: Optional[torch.Tensor] = None,
        memory_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        tgt_input: [B, L]
        memory: [B, S, d_model]
        tgt_padding_mask: [B, L] bool (True for pad positions)
        memory_padding_mask: [B, S] bool (True for pad positions)
        return logits: [B, L, vocab]
        """
        b, l = tgt_input.shape
        device = tgt_input.device

        x = self.embedding(tgt_input) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_enc(x)
        x = self.dropout(x)

        tgt_mask = self._generate_causal_mask(l, device=device)
        out = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_padding_mask,
        )
        logits = self.lm_head(out)
        return logits

