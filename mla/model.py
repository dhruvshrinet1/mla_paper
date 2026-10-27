"""Decoder-only transformer backbone (paper Section 4.1, Table 1).

Standard GPT-2 architecture: pre-norm blocks, 4x GELU MLP, no biases, tied
softmax output layer. The only thing that varies across the paper's three
architectures is the attention module, which is swapped in by `build_attention`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import LayerKVCache, build_attention
from .config import ModelConfig


class MLP(nn.Module):
    def __init__(self, d_model: int, dropout: float, bias: bool) -> None:
        super().__init__()
        self.fc = nn.Linear(d_model, 4 * d_model, bias=bias)
        self.proj = nn.Linear(4 * d_model, d_model, bias=bias)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.proj(F.gelu(self.fc(x), approximate="tanh")))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.d_model, bias=cfg.bias)
        self.attn = build_attention(cfg, max_seq_len=cfg.block_size)
        self.ln_2 = nn.LayerNorm(cfg.d_model, bias=cfg.bias)
        self.mlp = MLP(cfg.d_model, cfg.dropout, cfg.bias)

    def forward(self, x: torch.Tensor, cache: LayerKVCache | None = None):
        x = x + self.attn(self.ln_1(x), cache=cache)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.wpe = (
            nn.Embedding(cfg.block_size, cfg.d_model)
            if cfg.use_learned_pos_emb
            else None
        )
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            self.lm_head.weight = self.wte.weight

        self.apply(self._init_weights)
        # GPT-2 scaled init on residual projections
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("o_proj.weight"):
                nn.init.normal_(
                    p, mean=0.0, std=cfg.init_std / math.sqrt(2 * cfg.n_layer)
                )

    def _init_weights(self, module: nn.Module) -> None:
        std = self.cfg.init_std
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    # -- parameter accounting ---------------------------------------------

    def num_parameters(self, non_embedding: bool = False) -> int:
        total = sum(p.numel() for p in self.parameters())
        if non_embedding:
            total -= self.wte.weight.numel()
            if self.wpe is not None:
                total -= self.wpe.weight.numel()
        return total

    # -- KV cache ----------------------------------------------------------

    def allocate_cache(
        self,
        batch_size: int,
        max_len: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> list[LayerKVCache]:
        max_len = max_len or self.cfg.block_size
        device = device or next(self.parameters()).device
        dtype = dtype or next(self.parameters()).dtype
        return [
            b.attn.allocate_cache(batch_size, max_len, device, dtype)
            for b in self.blocks
        ]

    def kv_elements_per_token(self) -> int:
        """Total cached K/V elements per token across all layers."""
        return sum(b.attn.kv_elements_per_token() for b in self.blocks)

    def kv_bytes_per_token(self, dtype: torch.dtype = torch.float16) -> int:
        itemsize = torch.empty(0, dtype=dtype).element_size()
        return self.kv_elements_per_token() * itemsize

    # -- forward -----------------------------------------------------------

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        caches: list[LayerKVCache] | None = None,
    ):
        B, T = idx.shape
        offset = caches[0].length if caches is not None else 0
        if offset + T > self.cfg.block_size:
            raise ValueError(
                f"sequence length {offset + T} exceeds block_size "
                f"{self.cfg.block_size}"
            )

        x = self.wte(idx)
        if self.wpe is not None:
            pos = torch.arange(offset, offset + T, device=idx.device)
            x = x + self.wpe(pos)
        x = self.drop(x)

        for i, block in enumerate(self.blocks):
            x = block(x, cache=caches[i] if caches is not None else None)
        x = self.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )
            return logits, loss

        # inference: only the final position is needed
        logits = self.lm_head(x[:, [-1], :])
        return logits, None

    # -- generation --------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int | None = 200,
        top_p: float | None = None,
        use_cache: bool = True,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        self.eval()
        B = idx.shape[0]
        caches = None
        if use_cache:
            max_len = min(self.cfg.block_size, idx.shape[1] + max_new_tokens)
            caches = self.allocate_cache(B, max_len)

        finished = torch.zeros(B, dtype=torch.bool, device=idx.device)
        cur = idx
        for _ in range(max_new_tokens):
            if use_cache:
                step_in = cur if caches[0].length == 0 else cur[:, -1:]
            else:
                step_in = cur[:, -self.cfg.block_size :]

            logits, _ = self(step_in, caches=caches)
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            if top_k is not None:
                k = min(top_k, logits.size(-1))
                thresh = torch.topk(logits, k, dim=-1).values[:, [-1]]
                logits = logits.masked_fill(logits < thresh, float("-inf"))
            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                drop = probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
                logits = torch.full_like(logits, float("-inf")).scatter_(
                    -1, sorted_idx, sorted_logits
                )

            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            if eos_token_id is not None:
                nxt = torch.where(
                    finished.unsqueeze(1), torch.full_like(nxt, eos_token_id), nxt
                )
                finished |= nxt.squeeze(1) == eos_token_id
            cur = torch.cat((cur, nxt), dim=1)
            if eos_token_id is not None and bool(finished.all()):
                break
            if cur.shape[1] >= self.cfg.block_size:
                break
        return cur

    # -- optimiser ---------------------------------------------------------

    def configure_optimizers(
        self, weight_decay: float, lr: float, betas: tuple[float, float]
    ) -> torch.optim.Optimizer:
        """AdamW with decay on 2D parameters only (Sec 4.1 states beta and lr)."""
        decay, no_decay = [], []
        for _, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=lr, betas=betas)


def build_model(cfg: ModelConfig) -> GPT:
    return GPT(cfg)
