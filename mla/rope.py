"""Rotary position embeddings (paper Section 2.2, Eq. 4).

The paper writes RoPE(x_m, m) = x_m * e^{i m theta} and describes theta as
"learned frequencies". The method it cites (RoFormer, Su et al.) uses *fixed*
frequencies theta_i = base^{-2i/dim}. See PAPER_ANALYSIS.md issue #4.

Both are implemented here. Fixed is the default because it is what the cited
method does and what every reference implementation uses; `learned=True`
follows the paper's literal wording and makes the inverse frequencies
trainable parameters.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Split the last dim in half and rotate: (x1, x2) -> (-x2, x1)."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embedding to `x` of shape (B, H, T, D).

    `cos` and `sin` are (T, D) or broadcastable to (1, 1, T, D).
    """
    if cos.dim() == 2:
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
    return x * cos + rotate_half(x) * sin


class RotaryEmbedding(nn.Module):
    """Produces cos/sin tables for a given set of absolute positions.

    Args:
        dim: head dimension the rotation is applied to. Must be even.
        max_position: size of the precomputed cache.
        base: the `10000` in theta_i = base^{-2i/dim}.
        learned: if True, inverse frequencies become trainable parameters and
            the tables are recomputed each call (paper's literal wording).
        scaling: optional linear position scaling factor for context extension.
    """

    def __init__(
        self,
        dim: int,
        max_position: int = 4096,
        base: float = 10_000.0,
        learned: bool = False,
        scaling: float = 1.0,
    ) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE requires an even head dimension, got {dim}")
        self.dim = dim
        self.base = base
        self.learned = learned
        self.scaling = scaling
        self.max_position = max_position

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        if learned:
            self.inv_freq = nn.Parameter(inv_freq)
        else:
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self._build_cache(max_position)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, dtype=torch.float32) / self.scaling
        freqs = torch.outer(t, self.inv_freq)          # (T, dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)        # (T, dim)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
        self.max_position = seq_len

    def forward(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (cos, sin) for absolute positions [offset, offset + seq_len)."""
        end = offset + seq_len
        if self.learned:
            t = torch.arange(offset, end, device=device, dtype=torch.float32) / self.scaling
            freqs = torch.outer(t, self.inv_freq.to(device))
            emb = torch.cat((freqs, freqs), dim=-1)
            return emb.cos().to(dtype), emb.sin().to(dtype)

        if end > self.max_position:
            self._build_cache(max(end, self.max_position * 2))
        cos = self.cos_cached[offset:end].to(device=device, dtype=dtype)
        sin = self.sin_cached[offset:end].to(device=device, dtype=dtype)
        return cos, sin
