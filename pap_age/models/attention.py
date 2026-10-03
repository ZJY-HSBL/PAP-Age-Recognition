from __future__ import annotations

import torch
from torch import nn


class AdditiveTemporalAttention(nn.Module):
    def __init__(self, hidden_size: int, attention_dim: int, causal: bool = True):
        super().__init__()
        self.key = nn.Linear(hidden_size, attention_dim, bias=False)
        self.query = nn.Linear(hidden_size, attention_dim, bias=False)
        self.score = nn.Linear(attention_dim, 1, bias=False)
        self.causal = causal

    def forward(self, hidden: torch.Tensor, valid_mask: torch.Tensor):
        # hidden: [B,T,H], valid_mask: [B,T]
        k = self.key(hidden)[:, None, :, :]     # [B,1,T,A]
        q = self.query(hidden)[:, :, None, :]   # [B,T,1,A]
        scores = self.score(torch.tanh(k + q)).squeeze(-1)  # [B,T,T]

        key_mask = valid_mask[:, None, :].expand_as(scores)
        query_mask = valid_mask[:, :, None].expand_as(scores)
        mask = key_mask & query_mask
        if self.causal:
            t = hidden.shape[1]
            causal = torch.tril(torch.ones((t, t), dtype=torch.bool, device=hidden.device))
            mask = mask & causal[None, :, :]

        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        weights = torch.where(query_mask, weights, torch.zeros_like(weights))
        context = torch.bmm(weights, hidden)
        return context, weights
