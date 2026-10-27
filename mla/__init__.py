"""Reimplementation of "Latent Multi-Head Attention for Small Language Models".

arXiv:2506.09342v2 -- Mehta, Dandekar, Dandekar, Panat.

See PAPER_ANALYSIS.md for the section-by-section mapping from the paper to
this code, including the twelve internal inconsistencies found in the paper
and the choices made where it is under-specified.
"""

from .config import (
    ModelConfig,
    TrainConfig,
    TABLE1,
    PAPER_RATIOS,
    STATED_RATIOS,
    model_config,
    table2_variants,
    table4_variants,
)
from .attention import (
    MultiHeadAttention,
    LatentMultiHeadAttention,
    LayerKVCache,
    build_attention,
)
from .rope import RotaryEmbedding, apply_rope
from .model import GPT, build_model

__version__ = "0.1.0"

__all__ = [
    "ModelConfig",
    "TrainConfig",
    "TABLE1",
    "PAPER_RATIOS",
    "STATED_RATIOS",
    "model_config",
    "table2_variants",
    "table4_variants",
    "MultiHeadAttention",
    "LatentMultiHeadAttention",
    "LayerKVCache",
    "build_attention",
    "RotaryEmbedding",
    "apply_rope",
    "GPT",
    "build_model",
]
