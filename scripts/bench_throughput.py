#!/usr/bin/env python3
"""Decode throughput benchmark (paper Section 5.4, Figure 1a).

The paper benchmarks on a single A100 80GB over batch sizes 1 to 32. This
script runs the same sweep on whatever device is available.

It deliberately reports speedup against TWO baselines, because the paper's
headline is ambiguous (PAPER_ANALYSIS.md issue #12):

  - vs MHA               the reading invited by the abstract and conclusion
                         ("45% memory reduction and 1.4x inference speedup")
  - vs full-rank MLA     what Section 5.4 actually measures

By the paper's own numbers, MLA+RoPE at r=d/2 reaches 97 tok/s where MHA
reaches 170, so against the MHA baseline the recommended configuration is
roughly 43% SLOWER, not 1.4x faster. The 1.4x is relative to MLA+RoPE at
full rank.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from mla.config import model_config  # noqa: E402
from mla.model import build_model  # noqa: E402

#: Figure 1a, tokens/sec at batch 32, as stated in Section 5.4.
PAPER_TOKS_AT_B32 = {
    ("mha", 1.0): 170,
    ("mla", 1.0): 110,
    ("mla", 0.5): 162,
    ("mla", 0.25): 186,
    ("mla_rope", 1.0): 69,   # implied by "97 tok/s is 1.4x faster than r=d"
    ("mla_rope", 0.5): 97,
}

RATIO_LABEL = {1.0: "r=d", 0.5: "r=d/2", 0.25: "r=d/4",
               0.125: "r=d/8", 0.0625: "r=d/16", 0.03125: "r=d/32"}


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def decode_throughput(
    model, batch_size: int, prompt_len: int, gen_len: int,
    device: torch.device, warmup: int, repeats: int,
) -> dict:
    """Tokens/sec during autoregressive decode, excluding prefill."""
    model.eval()
    vocab = model.cfg.vocab_size
    prompt = torch.randint(0, vocab, (batch_size, prompt_len), device=device)

    def one_pass() -> float:
        caches = model.allocate_cache(batch_size, prompt_len + gen_len)
        model(prompt, caches=caches)          # prefill, not timed
        nxt = torch.randint(0, vocab, (batch_size, 1), device=device)
        sync(device)
        t0 = time.perf_counter()
        for _ in range(gen_len):
            logits, _ = model(nxt, caches=caches)
            nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        sync(device)
        return time.perf_counter() - t0

    for _ in range(warmup):
        one_pass()
    times = [one_pass() for _ in range(repeats)]

    best = min(times)
    total_tokens = gen_len * batch_size
    return {
        "tokens_per_sec": total_tokens / best,
        "tokens_per_sec_median": total_tokens / statistics.median(times),
        "latency_per_step_ms": 1000 * best / gen_len,
        "seconds_best": best,
        "n_repeats": repeats,
    }


def peak_memory_mb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated() / 1e6


def run(args) -> dict:
    device = torch.device(
        args.device if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    dtype = getattr(torch, args.dtype) if args.dtype != "auto" else (
        torch.float16 if device.type == "cuda" else torch.float32
    )

    if device.type == "cpu":
        print("[bench] WARNING: running on CPU. Absolute tokens/sec are not "
              "comparable to the paper's A100 figures; relative ordering "
              "between variants is still informative.")

    variants = [("mha", 1.0)]
    for attn in ("mla", "mla_rope"):
        for r in args.ratios:
            variants.append((attn, r))

    batch_sizes = args.batch_sizes
    results = []

    for attn, ratio in variants:
        cfg = model_config(
            args.size, attention=attn, latent_ratio=ratio,
            block_size=args.prompt_len + args.gen_len + 8,
            vocab_size=args.vocab_size, dropout=0.0,
            rope_mode=args.rope_mode, kv_cache_mode=args.kv_cache_mode,
        )
        model = build_model(cfg).to(device=device, dtype=dtype)
        label = f"{attn}/{RATIO_LABEL.get(ratio, ratio)}"
        for bs in batch_sizes:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            m = decode_throughput(
                model, bs, args.prompt_len, args.gen_len,
                device, args.warmup, args.repeats,
            )
            m.update({
                "attention": attn,
                "latent_ratio": ratio,
                "ratio_label": RATIO_LABEL.get(ratio, str(ratio)),
                "label": label,
                "batch_size": bs,
                "latent_dim": cfg.effective_latent_dim,
                "kv_elements_per_token": model.kv_elements_per_token(),
                "peak_memory_mb": peak_memory_mb(device),
            })
            results.append(m)
            print(f"  {label:<18} batch {bs:>3}  "
                  f"{m['tokens_per_sec']:>9.1f} tok/s  "
                  f"{m['latency_per_step_ms']:>7.2f} ms/step")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # -- speedups against both baselines --------------------------------
    by_key = {(r["attention"], r["latent_ratio"], r["batch_size"]): r for r in results}
    for r in results:
        bs = r["batch_size"]
        mha = by_key.get(("mha", 1.0, bs))
        full = by_key.get((r["attention"], 1.0, bs))
        r["speedup_vs_mha"] = (
            r["tokens_per_sec"] / mha["tokens_per_sec"] if mha else None
        )
        r["speedup_vs_full_rank_same_variant"] = (
            r["tokens_per_sec"] / full["tokens_per_sec"] if full else None
        )

    # -- the headline disambiguation ------------------------------------
    top_bs = max(batch_sizes)
    print(f"\nSpeedup at batch {top_bs}, measured two ways")
    hdr = (f"{'variant':<18} {'tok/s':>9} {'vs MHA':>9} "
           f"{'vs full-rank':>13} {'paper@b32':>10}")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        if r["batch_size"] != top_bs:
            continue
        paper = PAPER_TOKS_AT_B32.get((r["attention"], r["latent_ratio"]))
        ps = f"{paper}" if paper else "-"
        vm = f"{r['speedup_vs_mha']:.2f}x" if r["speedup_vs_mha"] else "-"
        vf = (f"{r['speedup_vs_full_rank_same_variant']:.2f}x"
              if r["speedup_vs_full_rank_same_variant"] else "-")
        print(f"{r['label']:<18} {r['tokens_per_sec']:>9.1f} {vm:>9} "
              f"{vf:>13} {ps:>10}")

    print(f"\nThe paper's own figures at batch 32 (Section 5.4):")
    print(f"  MHA                170 tok/s")
    print(f"  MLA+RoPE r=d        69 tok/s  (implied)")
    print(f"  MLA+RoPE r=d/2      97 tok/s")
    print(f"  -> 97/69 = 1.41x   which is the abstract's '1.4x speedup'")
    print(f"  -> 97/170 = 0.57x  i.e. 43% SLOWER than the MHA baseline")
    print(f"  The headline speedup is against full-rank MLA, not against MHA.")

    out = {
        "device": str(device),
        "dtype": str(dtype),
        "size": args.size,
        "prompt_len": args.prompt_len,
        "gen_len": args.gen_len,
        "batch_sizes": batch_sizes,
        "results": results,
        "paper_reference_b32": {f"{k[0]}@{k[1]}": v
                                for k, v in PAPER_TOKS_AT_B32.items()},
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"\n[bench] written to {args.out}")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--size", default="9L-512d")
    p.add_argument("--batch-sizes", type=int, nargs="+",
                   default=[1, 2, 4, 8, 16, 32])
    p.add_argument("--ratios", type=float, nargs="+", default=[1.0, 0.5, 0.25])
    p.add_argument("--prompt-len", type=int, default=64)
    p.add_argument("--gen-len", type=int, default=128)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--vocab-size", type=int, default=50257)
    p.add_argument("--dtype", default="auto")
    p.add_argument("--device", default="auto")
    p.add_argument("--rope-mode", default="standard",
                   choices=["standard", "decoupled"])
    p.add_argument("--kv-cache-mode", default="latent",
                   choices=["latent", "expanded"])
    p.add_argument("--out", default="results/throughput.json")
    run(p.parse_args())
