"""
Rotary positional embeddings (RoPE) for transformer attention.
Applies rotations to Q/K per head as used in RoFormer / GPT-style models.
"""

import torch
import torch.nn as nn


class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, d: int, base: int = 10_000):
        super().__init__()
        if d % 2 != 0:
            raise ValueError("RoPE dimension must be even.")
        self.base = base
        self.d = d
        self.register_buffer("cos_cached", None, persistent=False)
        self.register_buffer("sin_cached", None, persistent=False)

    def _build_cache(self, x: torch.Tensor):
        seq_len = x.shape[0]
        if self.cos_cached is not None and seq_len <= self.cos_cached.shape[0]:
            return

        theta = 1.0 / (self.base ** (torch.arange(0, self.d, 2, device=x.device).float() / self.d))
        seq_idx = torch.arange(seq_len, device=x.device).float()
        idx_theta = torch.einsum("n,d->nd", seq_idx, theta)
        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=-1)

        cos_cached = idx_theta2.cos()
        sin_cached = idx_theta2.sin()

        self.cos_cached = cos_cached[:, None, None, :]
        self.sin_cached = sin_cached[:, None, None, :]

    def _neg_half(self, x: torch.Tensor) -> torch.Tensor:
        d_2 = self.d // 2
        return torch.cat([-x[..., d_2:], x[..., :d_2]], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [seq_len, batch, n_heads, d]
        self._build_cache(x)
        cos = self.cos_cached[: x.shape[0]]
        sin = self.sin_cached[: x.shape[0]]
        return x * cos + self._neg_half(x) * sin
