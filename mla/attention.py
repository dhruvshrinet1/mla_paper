"""Attention variants from the paper: MHA, MLA, and MLA+RoPE.

Paper references
----------------
Eq. 1  (Sec 2.1)  standard multi-head attention
Eq. 2-3 (Sec 2.1) MLA's low-rank factorisation of K and V
Eq. 4  (Sec 2.2)  rotary position embeddings

Implementation notes that go beyond what the paper states
---------------------------------------------------------
1. The down-projections are PER HEAD. The paper's notation carries the head
   subscript (W^{K-down}_h) and it states cache memory drops from O(n H d_k)
   to O(n H r) -- the H survives. This is *not* DeepSeek-V2 MLA, where one
   shared down-projection gives O(n r). See PAPER_ANALYSIS.md issue #3.

2. `r` is a fraction of the HEAD dimension d_k, not of the model dimension d.
   Sec 2.1 says r < d_k and Fig 1b's axis is r/d_k, even though the results
   tables write "r = d/2". With d=512, H=8 a literal r=d/2=256 would be a 4x
   expansion of a 64-dim head.

3. Absorption. The paper never mentions it, but

       Q (K_lat W_up)^T = (Q W_up^T) K_lat^T
       A (V_lat W_up)   = (A V_lat) W_up

   so keys and values need never be materialised: attention can run entirely
   in the r-dimensional latent space. This is what makes MLA fast and what
   explains Sec 5.4's observation about a smaller attention matrix fitting in
   SRAM. Enabled by `absorb=True` whenever RoPE is off.

4. RoPE destroys absorption, because the rotation depends on absolute position
   and is applied after the up-projection. Three resolutions exist; the paper
   gives MLA and MLA+RoPE *identical* cache sizes in Table 2 and reports
   MLA+RoPE as consistently slower, which identifies its choice as
   `kv_cache_mode="latent"`: cache the latents and re-expand + re-rotate every
   decode step. `"expanded"` (cache post-RoPE keys, no memory saving) is
   provided for A/B timing. `rope_mode="decoupled"` implements DeepSeek-V2's
   fix and is an EXTENSION BEYOND THE PAPER, flagged as such in all output.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import RotaryEmbedding, apply_rope


# --------------------------------------------------------------------------
# KV cache
# --------------------------------------------------------------------------


@dataclass
class LayerKVCache:
    """Preallocated per-layer KV cache.

    `k`/`v` hold either materialised heads (dim d_k) or latents (dim r)
    depending on the attention module's cache mode. `k_rope` is only used by
    the decoupled-RoPE extension.
    """

    k: torch.Tensor                      # (B, H, max_len, k_dim)
    v: torch.Tensor                      # (B, H, max_len, v_dim)
    k_rope: torch.Tensor | None = None   # (B, 1, max_len, rope_dim)
    length: int = 0

    @classmethod
    def allocate(
        cls,
        batch_size: int,
        n_head: int,
        max_len: int,
        k_dim: int,
        v_dim: int,
        device: torch.device,
        dtype: torch.dtype,
        rope_dim: int = 0,
    ) -> "LayerKVCache":
        k = torch.zeros(batch_size, n_head, max_len, k_dim, device=device, dtype=dtype)
        v = torch.zeros(batch_size, n_head, max_len, v_dim, device=device, dtype=dtype)
        k_rope = (
            torch.zeros(batch_size, 1, max_len, rope_dim, device=device, dtype=dtype)
            if rope_dim > 0
            else None
        )
        return cls(k=k, v=v, k_rope=k_rope)

    def append(
        self,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        k_rope_new: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Write new entries and return views over the whole valid prefix."""
        t = k_new.shape[2]
        end = self.length + t
        if end > self.k.shape[2]:
            raise RuntimeError(
                f"KV cache overflow: need {end} slots, have {self.k.shape[2]}"
            )
        self.k[:, :, self.length : end] = k_new
        self.v[:, :, self.length : end] = v_new
        if k_rope_new is not None and self.k_rope is not None:
            self.k_rope[:, :, self.length : end] = k_rope_new
        self.length = end
        rope_view = self.k_rope[:, :, :end] if self.k_rope is not None else None
        return self.k[:, :, :end], self.v[:, :, :end], rope_view

    def bytes_allocated(self) -> int:
        total = self.k.numel() * self.k.element_size()
        total += self.v.numel() * self.v.element_size()
        if self.k_rope is not None:
            total += self.k_rope.numel() * self.k_rope.element_size()
        return total

    def reset(self) -> None:
        self.length = 0


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _causal_mask(
    q_len: int, kv_len: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor | None:
    """Additive mask for queries at positions [kv_len - q_len, kv_len)."""
    if q_len == 1:
        return None  # a single query attends to the whole cached prefix
    offset = kv_len - q_len
    q_pos = torch.arange(offset, offset + q_len, device=device).unsqueeze(1)
    k_pos = torch.arange(kv_len, device=device).unsqueeze(0)
    blocked = k_pos > q_pos
    mask = torch.zeros(q_len, kv_len, device=device, dtype=dtype)
    return mask.masked_fill_(blocked, torch.finfo(dtype).min)


def _attend(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    dropout_p: float,
    is_causal: bool,
    attn_mask: torch.Tensor | None,
    extra_scores: torch.Tensor | None = None,
) -> torch.Tensor:
    """Scaled dot-product attention with an optional additive score term.

    `extra_scores` carries the decoupled-RoPE positional contribution, which
    cannot be folded into q/k because it lives in a different subspace.
    """
    if extra_scores is None:
        return F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
        )

    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    scores = scores + extra_scores * scale
    if is_causal:
        q_len, kv_len = scores.shape[-2], scores.shape[-1]
        mask = _causal_mask(q_len, kv_len, scores.device, scores.dtype)
        if mask is not None:
            scores = scores + mask
    elif attn_mask is not None:
        scores = scores + attn_mask
    probs = F.softmax(scores, dim=-1)
    if dropout_p > 0.0:
        probs = F.dropout(probs, p=dropout_p, training=True)
    return torch.matmul(probs, v)


# --------------------------------------------------------------------------
# Standard multi-head attention (Eq. 1)
# --------------------------------------------------------------------------


class MultiHeadAttention(nn.Module):
    """Baseline MHA, with optional GQA/MQA via `n_kv_heads` and optional RoPE.

    GQA/MQA are not in the paper; they are included because Section 3 claims
    MLA's benefits are "orthogonal" to them, and that claim is only checkable
    if both are available in the same codebase.
    """

    def __init__(
        self,
        d_model: int,
        n_head: int,
        n_kv_heads: int | None = None,
        dropout: float = 0.0,
        use_rope: bool = False,
        rope_base: float = 10_000.0,
        rope_learned: bool = False,
        max_seq_len: int = 512,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError(f"d_model {d_model} not divisible by n_head {n_head}")
        self.d_model = d_model
        self.n_head = n_head
        self.n_kv_heads = n_kv_heads or n_head
        if n_head % self.n_kv_heads != 0:
            raise ValueError("n_head must be divisible by n_kv_heads")
        self.n_rep = n_head // self.n_kv_heads
        self.head_dim = d_model // n_head
        self.dropout = dropout
        self.use_rope = use_rope

        self.q_proj = nn.Linear(d_model, n_head * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(n_head * self.head_dim, d_model, bias=bias)
        self.resid_dropout = nn.Dropout(dropout)

        self.rope = (
            RotaryEmbedding(
                self.head_dim,
                max_position=max_seq_len,
                base=rope_base,
                learned=rope_learned,
            )
            if use_rope
            else None
        )

    # -- cache geometry, used by the memory benchmark ----------------------

    @property
    def cache_k_dim(self) -> int:
        return self.head_dim

    @property
    def cache_v_dim(self) -> int:
        return self.head_dim

    @property
    def cache_heads(self) -> int:
        return self.n_kv_heads

    @property
    def cache_rope_dim(self) -> int:
        return 0

    def kv_elements_per_token(self) -> int:
        """K and V elements cached per token, per layer."""
        return self.n_kv_heads * self.head_dim * 2

    def allocate_cache(self, batch_size, max_len, device, dtype) -> LayerKVCache:
        return LayerKVCache.allocate(
            batch_size, self.n_kv_heads, max_len,
            self.head_dim, self.head_dim, device, dtype,
        )

    # -- forward -----------------------------------------------------------

    def forward(
        self, x: torch.Tensor, cache: LayerKVCache | None = None
    ) -> torch.Tensor:
        B, T, _ = x.shape
        offset = cache.length if cache is not None else 0

        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.rope is not None:
            cos, sin = self.rope(T, x.device, q.dtype, offset=offset)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        if cache is not None:
            k, v, _ = cache.append(k, v)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        kv_len = k.shape[2]
        is_causal = offset == 0 and T > 1
        attn_mask = None if is_causal else _causal_mask(T, kv_len, x.device, q.dtype)

        out = _attend(
            q, k, v,
            scale=1.0 / math.sqrt(self.head_dim),
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
            attn_mask=attn_mask,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.resid_dropout(self.o_proj(out))


# --------------------------------------------------------------------------
# Multi-head latent attention (Eq. 2-3), with and without RoPE
# --------------------------------------------------------------------------


class LatentMultiHeadAttention(nn.Module):
    """MLA: K and V are produced through a per-head rank-`r` bottleneck.

    Args:
        latent_dim: r. If None, derived from `latent_ratio * head_dim`.
        latent_ratio: r as a fraction of d_k. The paper's "r = d/2" means this
            is 0.5 (see PAPER_ANALYSIS.md issue #3).
        use_rope: adds rotary embeddings, giving the paper's MLA+RoPE variant.
        kv_cache_mode: "latent" caches the r-dim latents (the paper's setting);
            "expanded" caches materialised d_k-dim keys/values (no saving).
        rope_mode: "standard" follows the paper; "decoupled" is the DeepSeek-V2
            technique and is an extension beyond the paper.
        absorb: fold the up-projections into the query/output side. Only
            possible when RoPE is off or decoupled.
    """

    def __init__(
        self,
        d_model: int,
        n_head: int,
        latent_dim: int | None = None,
        latent_ratio: float = 0.5,
        dropout: float = 0.0,
        use_rope: bool = False,
        rope_base: float = 10_000.0,
        rope_learned: bool = False,
        rope_mode: str = "standard",
        rope_dim: int | None = None,
        kv_cache_mode: str = "latent",
        absorb: bool = True,
        max_seq_len: int = 512,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError(f"d_model {d_model} not divisible by n_head {n_head}")
        if rope_mode not in ("standard", "decoupled"):
            raise ValueError(f"unknown rope_mode {rope_mode!r}")
        if kv_cache_mode not in ("latent", "expanded"):
            raise ValueError(f"unknown kv_cache_mode {kv_cache_mode!r}")

        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.dropout = dropout
        self.use_rope = use_rope
        self.rope_mode = rope_mode if use_rope else "standard"
        self.kv_cache_mode = kv_cache_mode

        if latent_dim is None:
            latent_dim = max(1, int(round(self.head_dim * latent_ratio)))
        self.latent_dim = latent_dim
        self.latent_ratio = latent_dim / self.head_dim

        # Absorption is impossible with standard RoPE: the rotation sits
        # between the latent and the score, and depends on position.
        self.absorb = absorb and not (use_rope and self.rope_mode == "standard")
        if self.kv_cache_mode == "expanded":
            self.absorb = False

        self.q_proj = nn.Linear(d_model, n_head * self.head_dim, bias=bias)
        # Per-head down-projections, packed into one matmul: d -> H*r
        self.k_down = nn.Linear(d_model, n_head * latent_dim, bias=bias)
        self.v_down = nn.Linear(d_model, n_head * latent_dim, bias=bias)
        # Per-head up-projections: (H, r, d_k)
        self.k_up = nn.Parameter(torch.empty(n_head, latent_dim, self.head_dim))
        self.v_up = nn.Parameter(torch.empty(n_head, latent_dim, self.head_dim))
        nn.init.normal_(self.k_up, mean=0.0, std=0.02)
        nn.init.normal_(self.v_up, mean=0.0, std=0.02)

        self.o_proj = nn.Linear(n_head * self.head_dim, d_model, bias=bias)
        self.resid_dropout = nn.Dropout(dropout)

        self.rope = None
        self.rope_dim = 0
        if use_rope:
            if self.rope_mode == "standard":
                self.rope = RotaryEmbedding(
                    self.head_dim, max_position=max_seq_len,
                    base=rope_base, learned=rope_learned,
                )
            else:
                # Decoupled: a small dedicated positional subspace. The key
                # side is shared across heads, as in DeepSeek-V2.
                self.rope_dim = rope_dim or max(2, self.head_dim // 2)
                self.q_rope = nn.Linear(d_model, n_head * self.rope_dim, bias=bias)
                self.k_rope = nn.Linear(d_model, self.rope_dim, bias=bias)
                self.rope = RotaryEmbedding(
                    self.rope_dim, max_position=max_seq_len,
                    base=rope_base, learned=rope_learned,
                )

    # -- cache geometry ----------------------------------------------------

    @property
    def cache_k_dim(self) -> int:
        return self.latent_dim if self.kv_cache_mode == "latent" else self.head_dim

    @property
    def cache_v_dim(self) -> int:
        return self.cache_k_dim

    @property
    def cache_heads(self) -> int:
        return self.n_head

    @property
    def cache_rope_dim(self) -> int:
        return self.rope_dim if self.rope_mode == "decoupled" else 0

    def kv_elements_per_token(self) -> int:
        return self.n_head * self.cache_k_dim * 2 + self.cache_rope_dim

    def allocate_cache(self, batch_size, max_len, device, dtype) -> LayerKVCache:
        return LayerKVCache.allocate(
            batch_size, self.n_head, max_len,
            self.cache_k_dim, self.cache_v_dim, device, dtype,
            rope_dim=self.cache_rope_dim,
        )

    # -- forward -----------------------------------------------------------

    def forward(
        self, x: torch.Tensor, cache: LayerKVCache | None = None
    ) -> torch.Tensor:
        B, T, _ = x.shape
        offset = cache.length if cache is not None else 0

        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k_lat = self.k_down(x).view(B, T, self.n_head, self.latent_dim).transpose(1, 2)
        v_lat = self.v_down(x).view(B, T, self.n_head, self.latent_dim).transpose(1, 2)

        if self.kv_cache_mode == "expanded":
            return self._forward_expanded(x, q, k_lat, v_lat, cache, offset, B, T)
        if self.use_rope and self.rope_mode == "standard":
            return self._forward_latent_rope(q, k_lat, v_lat, cache, offset, B, T)
        return self._forward_absorbed(x, q, k_lat, v_lat, cache, offset, B, T)

    # Path A: latents cached, keys/values materialised and rotated every step.
    # This is the paper's MLA+RoPE. Memory stays at O(H r); decode compute
    # grows because the whole prefix is re-expanded and re-rotated each step.
    def _forward_latent_rope(self, q, k_lat, v_lat, cache, offset, B, T):
        if cache is not None:
            k_lat, v_lat, _ = cache.append(k_lat, v_lat)

        k = torch.einsum("bhsr,hrd->bhsd", k_lat, self.k_up)
        v = torch.einsum("bhsr,hrd->bhsd", v_lat, self.v_up)

        kv_len = k.shape[2]
        cos_k, sin_k = self.rope(kv_len, q.device, q.dtype, offset=0)
        k = apply_rope(k, cos_k, sin_k)
        cos_q, sin_q = self.rope(T, q.device, q.dtype, offset=offset)
        q = apply_rope(q, cos_q, sin_q)

        is_causal = offset == 0 and T > 1
        attn_mask = None if is_causal else _causal_mask(T, kv_len, q.device, q.dtype)
        out = _attend(
            q, k, v,
            scale=1.0 / math.sqrt(self.head_dim),
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
            attn_mask=attn_mask,
        )
        return self._finish(out, B, T)

    # Path B: fully absorbed. Attention runs in r dimensions; K and V are
    # never materialised. Used for MLA without RoPE and for decoupled RoPE.
    def _forward_absorbed(self, x, q, k_lat, v_lat, cache, offset, B, T):
        k_rope_new = None
        q_rope = None
        if self.rope_mode == "decoupled":
            k_rope_new = self.k_rope(x).view(B, T, 1, self.rope_dim).transpose(1, 2)
            q_rope = (
                self.q_rope(x)
                .view(B, T, self.n_head, self.rope_dim)
                .transpose(1, 2)
            )
            cos_q, sin_q = self.rope(T, x.device, q.dtype, offset=offset)
            q_rope = apply_rope(q_rope, cos_q, sin_q)
            cos_k, sin_k = self.rope(T, x.device, q.dtype, offset=offset)
            k_rope_new = apply_rope(k_rope_new, cos_k, sin_k)

        k_rope_all = None
        if cache is not None:
            k_lat, v_lat, k_rope_all = cache.append(k_lat, v_lat, k_rope_new)
        else:
            k_rope_all = k_rope_new

        if self.absorb:
            # (Q W_up^T) K_lat^T  instead of  Q (K_lat W_up)^T
            q_eff = torch.einsum("bhtd,hrd->bhtr", q, self.k_up)
            k_eff, v_eff = k_lat, v_lat
        else:
            q_eff = q
            k_eff = torch.einsum("bhsr,hrd->bhsd", k_lat, self.k_up)
            v_eff = torch.einsum("bhsr,hrd->bhsd", v_lat, self.v_up)

        extra = None
        scale = 1.0 / math.sqrt(self.head_dim)
        if self.rope_mode == "decoupled" and k_rope_all is not None:
            extra = torch.matmul(q_rope, k_rope_all.transpose(-2, -1))
            scale = 1.0 / math.sqrt(self.head_dim + self.rope_dim)

        kv_len = k_eff.shape[2]
        is_causal = offset == 0 and T > 1
        attn_mask = None if is_causal else _causal_mask(T, kv_len, q.device, q.dtype)

        ctx = _attend(
            q_eff, k_eff, v_eff,
            scale=scale,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
            attn_mask=attn_mask,
            extra_scores=extra,
        )
        if self.absorb:
            # (A V_lat) W_up  instead of  A (V_lat W_up)
            ctx = torch.einsum("bhtr,hrd->bhtd", ctx, self.v_up)
        return self._finish(ctx, B, T)

    # Path C: cache materialised post-RoPE keys. No memory saving; here as a
    # timing reference for what MLA+RoPE costs in cache re-expansion.
    def _forward_expanded(self, x, q, k_lat, v_lat, cache, offset, B, T):
        k = torch.einsum("bhtr,hrd->bhtd", k_lat, self.k_up)
        v = torch.einsum("bhtr,hrd->bhtd", v_lat, self.v_up)
        if self.use_rope and self.rope is not None and self.rope_mode == "standard":
            cos, sin = self.rope(T, x.device, q.dtype, offset=offset)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
        if cache is not None:
            k, v, _ = cache.append(k, v)

        kv_len = k.shape[2]
        is_causal = offset == 0 and T > 1
        attn_mask = None if is_causal else _causal_mask(T, kv_len, q.device, q.dtype)
        out = _attend(
            q, k, v,
            scale=1.0 / math.sqrt(self.head_dim),
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
            attn_mask=attn_mask,
        )
        return self._finish(out, B, T)

    def _finish(self, ctx: torch.Tensor, B: int, T: int) -> torch.Tensor:
        ctx = ctx.transpose(1, 2).contiguous().view(B, T, -1)
        return self.resid_dropout(self.o_proj(ctx))


# --------------------------------------------------------------------------
# factory
# --------------------------------------------------------------------------


def build_attention(cfg, max_seq_len: int) -> nn.Module:
    """Construct the attention module named by `cfg.attention`."""
    kind = cfg.attention.lower()
    if kind == "mha":
        return MultiHeadAttention(
            d_model=cfg.d_model,
            n_head=cfg.n_head,
            n_kv_heads=cfg.n_kv_heads,
            dropout=cfg.dropout,
            use_rope=cfg.use_rope,
            rope_base=cfg.rope_base,
            rope_learned=cfg.rope_learned,
            max_seq_len=max_seq_len,
            bias=cfg.bias,
        )
    if kind in ("mla", "mla_rope"):
        return LatentMultiHeadAttention(
            d_model=cfg.d_model,
            n_head=cfg.n_head,
            latent_dim=cfg.latent_dim,
            latent_ratio=cfg.latent_ratio,
            dropout=cfg.dropout,
            use_rope=(kind == "mla_rope"),
            rope_base=cfg.rope_base,
            rope_learned=cfg.rope_learned,
            rope_mode=cfg.rope_mode,
            rope_dim=cfg.rope_dim,
            kv_cache_mode=cfg.kv_cache_mode,
            absorb=cfg.absorb,
            max_seq_len=max_seq_len,
            bias=cfg.bias,
        )
    raise ValueError(f"unknown attention variant {cfg.attention!r}")
