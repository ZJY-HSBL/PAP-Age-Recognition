from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .attention import AdditiveTemporalAttention


class PAPLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 1,
        attention_dim: int = 128,
        bidirectional: bool = False,
        causal_attention: bool = True,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.0 if num_layers == 1 else 0.1,
        )
        out_size = hidden_size * (2 if bidirectional else 1)
        self.attention = AdditiveTemporalAttention(out_size, attention_dim, causal=causal_attention)
        self.age_head = nn.Sequential(
            nn.Linear(out_size * 2, out_size),
            nn.ReLU(),
            nn.Linear(out_size, 1),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, mask: torch.Tensor):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        hidden, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=x.shape[1])
        context, weights = self.attention(hidden, mask)
        pred = self.age_head(torch.cat([hidden, context], dim=-1)).squeeze(-1)
        return pred, weights
