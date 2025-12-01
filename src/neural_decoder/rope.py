"""
Rotary positional embeddings (RoPE) for rotary attention (RoFormer/GPT-style).
Applies a rotation to Q/K per-head representations.
"""
import torch
import torch.nn as nn


class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, d: int, base: int = 10_000):
        """
        Rotary positional embeddings for last dimension of size d (d must be even).
        Expects input of shape [seq_len, batch, n_heads, d] and returns the same shape.
        """
        super().__init__()
        self.base = base
        self.d = d
        self.cos_cached = None  # [max_seq, 1, 1, d]
        self.sin_cached = None  # [max_seq, 1, 1, d]

    def _build_cache(self, x: torch.Tensor):
        seq_len = x.shape[0]
        if self.cos_cached is not None and seq_len <= self.cos_cached.shape[0]:
            return

        theta = 1.0 / (self.base ** (torch.arange(0, self.d, 2, device=x.device).float() / self.d))  # [d/2]
        seq_idx = torch.arange(seq_len, device=x.device).float()  # [seq_len]
        idx_theta = torch.einsum("n,d->nd", seq_idx, theta)  # [seq_len, d/2]
        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=-1)  # [seq_len, d]

        cos_cached = idx_theta2.cos()
        sin_cached = idx_theta2.sin()

        self.cos_cached = cos_cached[:, None, None, :]  # [seq_len, 1, 1, d]
        self.sin_cached = sin_cached[:, None, None, :]  # [seq_len, 1, 1, d]

    def _neg_half(self, x: torch.Tensor) -> torch.Tensor:
        d_2 = self.d // 2
        return torch.cat([-x[..., d_2:], x[..., :d_2]], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [seq_len, batch, n_heads, d]
        returns: same shape with rotary position embedding applied
        """
        self._build_cache(x)
        cos = self.cos_cached[: x.shape[0]]
        sin = self.sin_cached[: x.shape[0]]

        neg_half_x = self._neg_half(x)
        x_rope = x * cos + neg_half_x * sin
        return x_rope
