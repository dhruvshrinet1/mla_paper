#!/usr/bin/env python3
"""Train one model variant (paper Section 4.1).

Hyperparameters stated in the paper and implemented as defaults:
    AdamW, beta = (0.9, 0.95), lr 3e-4
    cosine schedule, 10% warmup, 50K steps
    batch size 128, context length 512
    gradient clipping 1.0, dropout 0.1
    GPT-2 tokenizer, tied softmax output layer

Everything the paper leaves open (weight decay, cosine floor, precision,
head count, init) is defaulted in mla/config.py and recorded in the run's
config.json so results are reproducible.

Examples
--------
    python scripts/train.py --size 9L-512d --attention mha
    python scripts/train.py --size 9L-512d --attention mla_rope --latent-ratio 0.5
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from mla.config import TrainConfig, model_config  # noqa: E402
from mla.data import TokenDataset  # noqa: E402
from mla.model import build_model  # noqa: E402


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(requested: str, device: torch.device) -> torch.dtype:
    if requested != "auto":
        return getattr(torch, requested)
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def lr_at(step: int, tc: TrainConfig) -> float:
    """Cosine schedule with linear warmup over the first 10% of steps."""
    warmup = tc.warmup_steps
    if step < warmup:
        return tc.learning_rate * (step + 1) / max(warmup, 1)
    if step >= tc.max_steps:
        return tc.learning_rate * tc.min_lr_ratio
    progress = (step - warmup) / max(tc.max_steps - warmup, 1)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    floor = tc.learning_rate * tc.min_lr_ratio
    return floor + coeff * (tc.learning_rate - floor)


@torch.no_grad()
def estimate_loss(model, datasets, tc, device, autocast_ctx) -> dict[str, float]:
    model.eval()
    out = {}
    for split, ds in datasets.items():
        losses = torch.zeros(tc.eval_iters)
        for i in range(tc.eval_iters):
            x, y = ds.get_batch(tc.micro_batch_size, device)
            with autocast_ctx:
                _, loss = model(x, targets=y)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def train(args) -> dict:
    tc = TrainConfig(
        learning_rate=args.lr,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        micro_batch_size=args.micro_batch_size,
        block_size=args.block_size,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        log_interval=args.log_interval,
        checkpoint_interval=args.checkpoint_interval,
        seed=args.seed,
        dtype=args.dtype,
        device=args.device,
        out_dir=args.out_dir,
        data_dir=args.data_dir,
        weight_decay=args.weight_decay,
        compile_model=args.compile,
    )

    torch.manual_seed(tc.seed)
    device = pick_device(tc.device)
    dtype = pick_dtype(tc.dtype, device)
    autocast_ctx = (
        torch.autocast(device_type=device.type, dtype=dtype)
        if device.type == "cuda" and dtype != torch.float32
        else torch.autocast(device_type="cpu", enabled=False)
    )

    train_ds = TokenDataset(tc.data_dir, "train", tc.block_size)
    val_ds = TokenDataset(tc.data_dir, "val", tc.block_size)
    datasets = {"train": train_ds, "val": val_ds}

    mcfg = model_config(
        args.size,
        attention=args.attention,
        latent_ratio=args.latent_ratio,
        vocab_size=train_ds.vocab_size,
        block_size=tc.block_size,
        dropout=args.dropout,
        rope_mode=args.rope_mode,
        rope_learned=args.rope_learned,
        kv_cache_mode=args.kv_cache_mode,
    )
    model = build_model(mcfg).to(device)
    if tc.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)

    run_dir = Path(tc.out_dir) / (args.run_name or mcfg.name)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "model": mcfg.to_dict(),
                "train": asdict(tc),
                "data_meta": train_ds.meta,
                "device": str(device),
                "dtype": str(dtype),
                "n_params": model.num_parameters(),
                "n_params_non_embedding": model.num_parameters(non_embedding=True),
                "latent_dim": mcfg.effective_latent_dim,
                "kv_elements_per_token": model.kv_elements_per_token(),
            },
            indent=2,
        )
    )

    print(f"[train] {mcfg.name}  params={model.num_parameters()/1e6:.2f}M  "
          f"r={mcfg.effective_latent_dim}  device={device}  dtype={dtype}")
    print(f"[train] {tc.max_steps} steps, batch {tc.batch_size} "
          f"({tc.grad_accum_steps} x {tc.micro_batch_size}), ctx {tc.block_size}")

    optimizer = model.configure_optimizers(
        tc.weight_decay, tc.learning_rate, (tc.beta1, tc.beta2)
    )
    scaler = torch.amp.GradScaler(
        device.type, enabled=(dtype == torch.float16 and device.type == "cuda")
    )

    history: list[dict] = []
    best_val = float("inf")
    t0 = time.time()

    for step in range(tc.max_steps):
        lr = lr_at(step, tc)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(tc.grad_accum_steps):
            x, y = train_ds.get_batch(tc.micro_batch_size, device)
            with autocast_ctx:
                _, loss = model(x, targets=y)
                loss = loss / tc.grad_accum_steps
            scaler.scale(loss).backward()
            total += loss.item()

        if tc.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if step % tc.log_interval == 0:
            el = time.time() - t0
            print(f"  step {step:>6}  loss {total:.4f}  lr {lr:.2e}  {el:.0f}s")

        is_last = step == tc.max_steps - 1
        if (step + 1) % tc.eval_interval == 0 or is_last:
            metrics = estimate_loss(model, datasets, tc, device, autocast_ctx)
            metrics["step"] = step + 1
            metrics["ppl_val"] = math.exp(min(metrics["val"], 20))
            history.append(metrics)
            print(f"  [eval] step {step+1}  train {metrics['train']:.4f}  "
                  f"val {metrics['val']:.4f}  ppl {metrics['ppl_val']:.2f}")
            if metrics["val"] < best_val:
                best_val = metrics["val"]
                torch.save(
                    {
                        "model": model.state_dict(),
                        "model_config": mcfg.to_dict(),
                        "step": step + 1,
                        "val_loss": best_val,
                    },
                    run_dir / "best.pt",
                )
            (run_dir / "history.json").write_text(json.dumps(history, indent=2))

    result = {
        "name": mcfg.name,
        "size": args.size,
        "attention": args.attention,
        "latent_ratio": mcfg.effective_latent_dim / mcfg.head_dim,
        "latent_dim": mcfg.effective_latent_dim,
        "best_val_loss": best_val,
        "best_val_ppl": math.exp(min(best_val, 20)),
        "n_params": model.num_parameters(),
        "kv_elements_per_token": model.kv_elements_per_token(),
        "steps": tc.max_steps,
        "wall_seconds": time.time() - t0,
        "run_dir": str(run_dir),
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(f"[train] done. best val loss {best_val:.4f} "
          f"({time.time()-t0:.0f}s) -> {run_dir}")
    return result


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--size", default="9L-512d")
    p.add_argument("--attention", default="mha", choices=["mha", "mla", "mla_rope"])
    p.add_argument("--latent-ratio", type=float, default=1.0,
                   help="r as a fraction of d_k; the paper's 'r=d/2' is 0.5")
    p.add_argument("--rope-mode", default="standard",
                   choices=["standard", "decoupled"],
                   help="decoupled is an extension beyond the paper")
    p.add_argument("--rope-learned", action="store_true",
                   help="learnable RoPE frequencies, per the paper's wording")
    p.add_argument("--kv-cache-mode", default="latent",
                   choices=["latent", "expanded"])
    # Section 4.1 values as defaults
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-steps", type=int, default=50_000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--micro-batch-size", type=int, default=16)
    p.add_argument("--block-size", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.1)
    # infrastructure
    p.add_argument("--eval-interval", type=int, default=1_000)
    p.add_argument("--eval-iters", type=int, default=100)
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--checkpoint-interval", type=int, default=5_000)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--dtype", default="auto")
    p.add_argument("--device", default="auto")
    p.add_argument("--data-dir", default="data/tinystories")
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--run-name", default=None)
    p.add_argument("--compile", action="store_true")
    return p


if __name__ == "__main__":
    train(build_parser().parse_args())
