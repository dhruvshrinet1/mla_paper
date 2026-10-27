"""Correctness tests for the attention variants.

Run with:  python -m pytest tests/ -v     (or: python tests/test_attention.py)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch

from mla.attention import LatentMultiHeadAttention, MultiHeadAttention
from mla.config import model_config
from mla.model import build_model
from mla.rope import RotaryEmbedding, apply_rope


# --------------------------------------------------------------------------
# RoPE (Eq. 4)
# --------------------------------------------------------------------------


def test_rope_preserves_norm():
    """Rotation is orthogonal, so vector norms are unchanged."""
    rope = RotaryEmbedding(dim=64)
    x = torch.randn(2, 4, 16, 64)
    cos, sin = rope(16, x.device, x.dtype)
    out = apply_rope(x, cos, sin)
    assert torch.allclose(x.norm(dim=-1), out.norm(dim=-1), atol=1e-5)


def test_rope_is_relative():
    """<RoPE(q,m), RoPE(k,n)> depends only on m-n, which is RoPE's whole point."""
    rope = RotaryEmbedding(dim=32)
    q, k = torch.randn(1, 1, 1, 32), torch.randn(1, 1, 1, 32)
    cos, sin = rope(64, q.device, q.dtype)

    def dot(m, n):
        qa = apply_rope(q, cos[m : m + 1], sin[m : m + 1])
        ka = apply_rope(k, cos[n : n + 1], sin[n : n + 1])
        return (qa * ka).sum().item()

    assert dot(5, 3) == pytest.approx(dot(20, 18), abs=1e-4)
    assert dot(10, 0) == pytest.approx(dot(31, 21), abs=1e-4)


def test_rope_position_zero_is_identity():
    rope = RotaryEmbedding(dim=16)
    x = torch.randn(1, 1, 1, 16)
    cos, sin = rope(1, x.device, x.dtype)
    assert torch.allclose(apply_rope(x, cos, sin), x, atol=1e-6)


def test_rope_learned_frequencies_are_parameters():
    """The paper says theta is learned; that option must actually train."""
    rope = RotaryEmbedding(dim=16, learned=True)
    assert rope.inv_freq.requires_grad
    cos, sin = rope(8, torch.device("cpu"))
    cos.sum().backward()
    assert rope.inv_freq.grad is not None


def test_rope_odd_dim_rejected():
    with pytest.raises(ValueError):
        RotaryEmbedding(dim=15)


# --------------------------------------------------------------------------
# MLA factorisation (Eq. 2-3)
# --------------------------------------------------------------------------


def test_latent_dim_derives_from_head_dim_not_model_dim():
    """Paper issue #3: r is a fraction of d_k, so r=d/2 on a 512-d model is 32."""
    attn = LatentMultiHeadAttention(d_model=512, n_head=8, latent_ratio=0.5)
    assert attn.head_dim == 64
    assert attn.latent_dim == 32


def test_cache_scales_exactly_linearly_in_r():
    """Section 2.1: cache is O(n H r), so halving r must halve the cache."""
    sizes = []
    for ratio in (1.0, 0.5, 0.25, 0.125):
        a = LatentMultiHeadAttention(512, 8, latent_ratio=ratio)
        sizes.append(a.kv_elements_per_token())
    for prev, cur in zip(sizes, sizes[1:]):
        assert prev == 2 * cur, f"expected exact 2x scaling, got {prev} -> {cur}"


def test_full_rank_mla_matches_mha_cache_size():
    """Section 2.1: r = d_k yields no compression."""
    mha = MultiHeadAttention(512, 8)
    mla = LatentMultiHeadAttention(512, 8, latent_ratio=1.0)
    assert mla.kv_elements_per_token() == mha.kv_elements_per_token()


def test_full_rank_mla_can_represent_mha():
    """At r = d_k the factorisation is unconstrained, so MLA can fit any MHA.

    W_down @ W_up must be able to equal an arbitrary d x d_k matrix.
    """
    torch.manual_seed(0)
    d_model, n_head = 64, 2
    mha = MultiHeadAttention(d_model, n_head, dropout=0.0).eval()
    mla = LatentMultiHeadAttention(
        d_model, n_head, latent_ratio=1.0, dropout=0.0, absorb=False
    ).eval()
    head_dim = d_model // n_head

    with torch.no_grad():
        mla.q_proj.weight.copy_(mha.q_proj.weight)
        mla.o_proj.weight.copy_(mha.o_proj.weight)
        # set up = identity so down must equal MHA's projection
        for h in range(n_head):
            mla.k_up[h].copy_(torch.eye(head_dim))
            mla.v_up[h].copy_(torch.eye(head_dim))
        mla.k_down.weight.copy_(mha.k_proj.weight)
        mla.v_down.weight.copy_(mha.v_proj.weight)

    x = torch.randn(2, 10, d_model)
    assert torch.allclose(mha(x), mla(x), atol=1e-5)


@pytest.mark.parametrize("use_rope,rope_mode", [(False, "standard"), (True, "decoupled")])
def test_absorption_is_mathematically_exact(use_rope, rope_mode):
    """(Q W_up^T) K_lat^T must equal Q (K_lat W_up)^T exactly."""
    torch.manual_seed(0)
    kw = dict(d_model=128, n_head=4, latent_ratio=0.5, dropout=0.0,
              use_rope=use_rope, rope_mode=rope_mode)
    a = LatentMultiHeadAttention(absorb=True, **kw).eval()
    b = LatentMultiHeadAttention(absorb=False, **kw).eval()
    b.load_state_dict(a.state_dict())
    assert a.absorb and not b.absorb
    x = torch.randn(2, 12, 128)
    assert torch.allclose(a(x), b(x), atol=1e-5)


def test_absorption_disabled_under_standard_rope():
    """RoPE sits between the latent and the score, so absorption is invalid."""
    a = LatentMultiHeadAttention(128, 4, use_rope=True, rope_mode="standard",
                                 absorb=True)
    assert a.absorb is False


def test_expanded_cache_mode_gives_no_memory_saving():
    a = LatentMultiHeadAttention(512, 8, latent_ratio=0.25,
                                 kv_cache_mode="expanded")
    mha = MultiHeadAttention(512, 8)
    assert a.kv_elements_per_token() == mha.kv_elements_per_token()


# --------------------------------------------------------------------------
# causality and caching
# --------------------------------------------------------------------------


ALL_VARIANTS = [
    ("mha", 1.0, "standard"),
    ("mla", 1.0, "standard"),
    ("mla", 0.5, "standard"),
    ("mla", 0.25, "standard"),
    ("mla_rope", 1.0, "standard"),
    ("mla_rope", 0.5, "standard"),
    ("mla_rope", 0.5, "decoupled"),
]


@pytest.mark.parametrize("attn,ratio,rope_mode", ALL_VARIANTS)
def test_incremental_decode_matches_full_forward(attn, ratio, rope_mode):
    """Token-by-token decode with a cache must equal one full forward pass."""
    torch.manual_seed(0)
    cfg = model_config(
        "6L-256d", attention=attn, latent_ratio=ratio, vocab_size=128,
        block_size=32, dropout=0.0, rope_mode=rope_mode,
    )
    model = build_model(cfg).eval()
    x = torch.randint(0, 128, (2, 16))

    full, _ = model(x)
    caches = model.allocate_cache(2, 16)
    for t in range(16):
        step, _ = model(x[:, t : t + 1], caches=caches)
    assert torch.allclose(full, step, atol=1e-4)


@pytest.mark.parametrize("attn,ratio,rope_mode", ALL_VARIANTS)
def test_prefill_then_decode_matches_full_forward(attn, ratio, rope_mode):
    """A chunked prefill followed by decode must also match."""
    torch.manual_seed(0)
    cfg = model_config(
        "6L-256d", attention=attn, latent_ratio=ratio, vocab_size=128,
        block_size=32, dropout=0.0, rope_mode=rope_mode,
    )
    model = build_model(cfg).eval()
    x = torch.randint(0, 128, (2, 16))

    full, _ = model(x)
    caches = model.allocate_cache(2, 16)
    model(x[:, :10], caches=caches)
    for t in range(10, 16):
        step, _ = model(x[:, t : t + 1], caches=caches)
    assert torch.allclose(full, step, atol=1e-4)


@pytest.mark.parametrize("attn,ratio,rope_mode", ALL_VARIANTS)
def test_causality(attn, ratio, rope_mode):
    """Changing a future token must not change an earlier position's output."""
    torch.manual_seed(0)
    cfg = model_config(
        "6L-256d", attention=attn, latent_ratio=ratio, vocab_size=128,
        block_size=32, dropout=0.0, rope_mode=rope_mode,
    )
    model = build_model(cfg).eval()
    x = torch.randint(0, 128, (1, 12))
    y = x.clone()
    y[0, -1] = (y[0, -1] + 7) % 128

    hx = model.wte(x)
    hy = model.wte(y)
    if model.wpe is not None:
        pos = torch.arange(12)
        hx, hy = hx + model.wpe(pos), hy + model.wpe(pos)
    for block in model.blocks:
        hx, hy = block(hx), block(hy)
    assert torch.allclose(hx[:, :-1], hy[:, :-1], atol=1e-5)


def test_cache_overflow_raises():
    cfg = model_config("6L-256d", vocab_size=64, block_size=16)
    model = build_model(cfg).eval()
    caches = model.allocate_cache(1, 4)
    model(torch.randint(0, 64, (1, 4)), caches=caches)
    with pytest.raises(RuntimeError, match="overflow"):
        model(torch.randint(0, 64, (1, 1)), caches=caches)


# --------------------------------------------------------------------------
# GQA / MQA (Section 3's "orthogonal benefits" claim)
# --------------------------------------------------------------------------


def test_mqa_reduces_cache_by_head_count():
    mha = MultiHeadAttention(512, 8)
    mqa = MultiHeadAttention(512, 8, n_kv_heads=1)
    assert mha.kv_elements_per_token() == 8 * mqa.kv_elements_per_token()


def test_gqa_forward_shape():
    gqa = MultiHeadAttention(512, 8, n_kv_heads=2, dropout=0.0).eval()
    x = torch.randn(2, 10, 512)
    assert gqa(x).shape == (2, 10, 512)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "--tb=short"]))
