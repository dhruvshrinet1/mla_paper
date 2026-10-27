"""Model and training configuration.

Table 1 of the paper gives eight depth/width configurations. Reproducing the
stated parameter counts pins down the architecture: standard GPT-2 blocks
(4x MLP, no biases) with a tied softmax output layer. See PAPER_ANALYSIS.md
Section 4 for the verification table -- all eight match to within 0.5%.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

# Head dimension is fixed at 64 across the paper's configs. The paper never
# states H, but d_k = 64 is the only choice that makes "r = d_k/2" and
# "r = d_k/4" land on sensible latent sizes (32 and 16) for every width.
DEFAULT_HEAD_DIM = 64


@dataclass
class ModelConfig:
    """Everything needed to instantiate a model."""

    # -- Table 1 axes ----------------------------------------------------
    n_layer: int = 6
    d_model: int = 512
    n_head: int = 8
    vocab_size: int = 50257          # GPT-2 tokenizer, "50K vocab" (Sec 4.1)
    block_size: int = 512            # context length (Sec 4.1)

    # -- attention variant ------------------------------------------------
    attention: str = "mha"           # "mha" | "mla" | "mla_rope"
    latent_dim: int | None = None    # r; None -> latent_ratio * head_dim
    latent_ratio: float = 1.0        # r as a fraction of d_k (NOT of d)
    n_kv_heads: int | None = None    # GQA/MQA for the MHA path only
    absorb: bool = True
    kv_cache_mode: str = "latent"    # "latent" (paper) | "expanded"

    # -- positional -------------------------------------------------------
    use_rope: bool = False           # only consulted for the MHA path
    rope_base: float = 10_000.0
    rope_learned: bool = False       # paper says "learned"; cited work says fixed
    rope_mode: str = "standard"      # "standard" (paper) | "decoupled" (extension)
    rope_dim: int | None = None
    use_learned_pos_emb: bool | None = None  # None -> auto: off when RoPE is on

    # -- regularisation and init -------------------------------------------
    dropout: float = 0.1             # Sec 4.1
    bias: bool = False
    tie_embeddings: bool = True      # Table 1 caption: "tied softmax output layer"
    init_std: float = 0.02

    def __post_init__(self) -> None:
        if self.d_model % self.n_head != 0:
            raise ValueError(
                f"d_model {self.d_model} not divisible by n_head {self.n_head}"
            )
        if self.use_learned_pos_emb is None:
            # Sec 2.2: RoPE encodes position "without explicit position
            # embeddings", so the learned table is dropped when RoPE is active.
            self.use_learned_pos_emb = not self.rope_active

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head

    @property
    def rope_active(self) -> bool:
        if self.attention == "mla_rope":
            return True
        if self.attention == "mha":
            return self.use_rope
        return False

    @property
    def effective_latent_dim(self) -> int:
        if self.attention == "mha":
            return self.head_dim
        if self.latent_dim is not None:
            return self.latent_dim
        return max(1, int(round(self.head_dim * self.latent_ratio)))

    @property
    def name(self) -> str:
        base = f"{self.n_layer}L-{self.d_model}d"
        if self.attention == "mha":
            return f"{base}_mha"
        ratio = self.effective_latent_dim / self.head_dim
        tag = _ratio_tag(ratio)
        return f"{base}_{self.attention}_r{tag}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _ratio_tag(ratio: float) -> str:
    """Render 0.5 as 'd2' (meaning d_k/2), 1.0 as 'd', etc."""
    if ratio >= 1.0:
        return "d"
    inv = round(1.0 / ratio)
    return f"d{inv}"


@dataclass
class TrainConfig:
    """Section 4.1 hyperparameters, plus the ones the paper leaves open."""

    # -- stated in the paper ----------------------------------------------
    learning_rate: float = 3e-4
    beta1: float = 0.9
    beta2: float = 0.95
    max_steps: int = 50_000
    warmup_ratio: float = 0.10
    batch_size: int = 128
    block_size: int = 512
    grad_clip: float = 1.0

    # -- not stated; documented defaults ----------------------------------
    weight_decay: float = 0.1
    min_lr_ratio: float = 0.10       # cosine floor as a fraction of peak LR
    micro_batch_size: int = 16       # grad accumulation to reach batch_size
    eval_interval: int = 1_000
    eval_iters: int = 100
    log_interval: int = 50
    checkpoint_interval: int = 5_000
    seed: int = 1337
    dtype: str = "auto"              # "auto" | "float32" | "bfloat16" | "float16"
    compile_model: bool = False
    device: str = "auto"
    out_dir: str = "runs"
    data_dir: str = "data/tinystories"

    @property
    def warmup_steps(self) -> int:
        return int(self.max_steps * self.warmup_ratio)

    @property
    def grad_accum_steps(self) -> int:
        if self.batch_size % self.micro_batch_size != 0:
            raise ValueError(
                f"batch_size {self.batch_size} not divisible by "
                f"micro_batch_size {self.micro_batch_size}"
            )
        return self.batch_size // self.micro_batch_size

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------
# Table 1: the eight configurations
# --------------------------------------------------------------------------

#: name -> (n_layer, d_model, paper's reported parameter count in millions)
TABLE1 = {
    "6L-256d":   (6, 256, 17.5),
    "6L-512d":   (6, 512, 44.5),
    "9L-256d":   (9, 256, 19.9),
    "9L-512d":   (9, 512, 53.9),
    "12L-256d":  (12, 256, 22.2),
    "12L-512d":  (12, 512, 63.3),
    "12L-768d":  (12, 768, 123.3),
    "12L-1024d": (12, 1024, 202.7),
}


def model_config(
    size: str = "9L-512d",
    attention: str = "mha",
    latent_ratio: float = 1.0,
    **overrides: Any,
) -> ModelConfig:
    """Build a ModelConfig for one of the Table 1 sizes."""
    if size not in TABLE1:
        raise KeyError(f"unknown size {size!r}; options: {sorted(TABLE1)}")
    n_layer, d_model, _ = TABLE1[size]
    cfg = dict(
        n_layer=n_layer,
        d_model=d_model,
        n_head=d_model // DEFAULT_HEAD_DIM,
        attention=attention,
        latent_ratio=latent_ratio,
    )
    cfg.update(overrides)
    return ModelConfig(**cfg)


#: The compression ratios actually reported in Tables 2 and 5. Note Section 6
#: claims only three were evaluated; six appear in the results.
PAPER_RATIOS = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125]

#: The subset Section 6 claims was evaluated.
STATED_RATIOS = [1.0, 0.5, 0.25]


def table2_variants(size: str = "6L-512d", ratios: list[float] | None = None):
    """The full variant list of Table 2: MHA plus MLA and MLA+RoPE at each r.

    Default size is 6L-512d, not the 9L-512d the table is labelled with --
    Table 2's loss values are the 6L-512d row of Table 4. See
    PAPER_ANALYSIS.md issue #5.
    """
    ratios = ratios or PAPER_RATIOS
    variants = [model_config(size, attention="mha")]
    for attn in ("mla", "mla_rope"):
        for r in ratios:
            variants.append(model_config(size, attention=attn, latent_ratio=r))
    return variants


def table4_variants(ratios: list[float] | None = None):
    """Appendix A sweep: all 8 sizes x {MHA, MLA, MLA+RoPE} x {r=d, r=d/2}."""
    ratios = ratios or [1.0, 0.5]
    out = []
    for size in TABLE1:
        out.append(model_config(size, attention="mha"))
        for attn in ("mla", "mla_rope"):
            for r in ratios:
                out.append(model_config(size, attention=attn, latent_ratio=r))
    return out
