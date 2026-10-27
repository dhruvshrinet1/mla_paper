#!/usr/bin/env python3
"""KV-cache memory benchmark (paper Table 2 "KV Mem" column, Figure 1b).

Reports both an analytical figure derived from the model's actual cache
geometry and a measured figure from real allocated tensors, and prints the
paper's numbers alongside.

Two findings this script makes visible (PAPER_ANALYSIS.md issues #6 and #7):

1. The paper's own column breaks O(r) scaling at exactly one point. Every
   step from r=d/2 downward is a clean 2x, but d -> d/2 is 1.81x. Halving r
   must halve the cache. The true reduction is 50%, not the headline 45%.

2. No combination of layer count and dtype reproduces the paper's absolute
   MB/tok values from the stated configuration.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from mla.config import PAPER_RATIOS, model_config  # noqa: E402
from mla.model import build_model  # noqa: E402

#: Table 2's KV Mem column, MB/token, keyed by (attention, ratio).
PAPER_KV_MEM = {
    ("mha", 1.0): 0.0288,
    ("mla", 1.0): 0.0288,
    ("mla", 0.5): 0.0159,
    ("mla", 0.25): 0.0080,
    ("mla", 0.125): 0.0040,
    ("mla", 0.0625): 0.0020,
    ("mla", 0.03125): 0.0010,
    ("mla_rope", 1.0): 0.0288,
    ("mla_rope", 0.5): 0.0159,
    ("mla_rope", 0.25): 0.0080,
    ("mla_rope", 0.125): 0.0040,
    ("mla_rope", 0.0625): 0.0020,
    ("mla_rope", 0.03125): 0.0010,
}

RATIO_LABEL = {
    1.0: "r = d", 0.5: "r = d/2", 0.25: "r = d/4",
    0.125: "r = d/8", 0.0625: "r = d/16", 0.03125: "r = d/32",
}


def measure_cache_bytes(model, batch_size: int, seq_len: int, dtype) -> int:
    caches = model.allocate_cache(
        batch_size, seq_len, device=torch.device("cpu"), dtype=dtype
    )
    return sum(c.bytes_allocated() for c in caches)


def run(args) -> dict:
    dtype = getattr(torch, args.dtype)
    itemsize = torch.empty(0, dtype=dtype).element_size()
    rows = []

    variants = [("mha", 1.0)]
    for attn in ("mla", "mla_rope"):
        for r in PAPER_RATIOS:
            variants.append((attn, r))

    for attn, ratio in variants:
        cfg = model_config(
            args.size, attention=attn, latent_ratio=ratio,
            block_size=args.seq_len, vocab_size=args.vocab_size,
            rope_mode=args.rope_mode,
        )
        model = build_model(cfg)
        elements = model.kv_elements_per_token()
        analytical_mb = elements * itemsize / 1e6
        measured = measure_cache_bytes(model, args.batch_size, args.seq_len, dtype)
        measured_mb = measured / args.batch_size / args.seq_len / 1e6

        rows.append({
            "attention": attn,
            "latent_ratio": ratio,
            "ratio_label": RATIO_LABEL.get(ratio, f"r={ratio}"),
            "latent_dim": cfg.effective_latent_dim,
            "kv_elements_per_token": elements,
            "analytical_mb_per_token": analytical_mb,
            "measured_mb_per_token": measured_mb,
            "paper_mb_per_token": PAPER_KV_MEM.get((attn, ratio)),
        })
        del model

    baseline = rows[0]["analytical_mb_per_token"]
    for r in rows:
        r["reduction_vs_mha_pct"] = 100.0 * (1 - r["analytical_mb_per_token"] / baseline)
        r["savings_factor_vs_mha"] = baseline / r["analytical_mb_per_token"]

    # -- report ---------------------------------------------------------
    print(f"\nKV-cache memory, {args.size}, dtype={args.dtype} "
          f"({itemsize} bytes/elem), L={model_config(args.size).n_layer}, "
          f"d_k={model_config(args.size).head_dim}\n")
    hdr = (f"{'variant':<12} {'r':>8} {'r_dim':>6} {'ours MB/tok':>12} "
           f"{'measured':>10} {'paper':>8} {'vs MHA':>9} {'factor':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        paper = f"{r['paper_mb_per_token']:.4f}" if r["paper_mb_per_token"] else "-"
        print(f"{r['attention']:<12} {r['ratio_label']:>8} {r['latent_dim']:>6} "
              f"{r['analytical_mb_per_token']:>12.5f} "
              f"{r['measured_mb_per_token']:>10.5f} {paper:>8} "
              f"{r['reduction_vs_mha_pct']:>8.1f}% "
              f"{r['savings_factor_vs_mha']:>7.2f}x")

    # -- the scaling audit ----------------------------------------------
    print("\nScaling audit: cache is O(r), so every halving of r must be exactly 2.00x")
    print(f"{'step':<22} {'paper ratio':>12} {'ours ratio':>12}   verdict")
    mla_rows = [r for r in rows if r["attention"] == "mla"]
    for prev, cur in zip(mla_rows, mla_rows[1:]):
        p_ratio = (prev["paper_mb_per_token"] / cur["paper_mb_per_token"]
                   if prev["paper_mb_per_token"] and cur["paper_mb_per_token"] else None)
        o_ratio = prev["analytical_mb_per_token"] / cur["analytical_mb_per_token"]
        step = f"{prev['ratio_label']} -> {cur['ratio_label']}"
        verdict = "OK" if p_ratio and abs(p_ratio - 2.0) < 0.05 else "BREAKS O(r)"
        ps = f"{p_ratio:.2f}x" if p_ratio else "-"
        print(f"{step:<22} {ps:>12} {o_ratio:>11.2f}x   {verdict}")

    half = next(r for r in rows if r["attention"] == "mla" and r["latent_ratio"] == 0.5)
    print(f"\nHeadline check at r = d/2:")
    print(f"  paper claims 45% reduction (0.0288 -> 0.0159 MB/tok)")
    print(f"  true value   {half['reduction_vs_mha_pct']:.1f}% reduction "
          f"(halving r halves the cache)")
    print(f"  the 45% figure comes from one table cell that breaks O(r) scaling")

    print("\nDerivability check: can the paper's 0.0288 MB/tok be reproduced?")
    for L in (6, 9, 12):
        for name, size in (("fp16", 2), ("fp32", 4)):
            d = model_config(args.size).d_model
            mb = L * 2 * d * size / 1e6
            print(f"  L={L:>2} d={d} {name}: {mb:.4f} MB/tok")
    print("  paper:                 0.0288 MB/tok  -> matches none of the above")

    out = {
        "size": args.size,
        "dtype": args.dtype,
        "bytes_per_element": itemsize,
        "rows": rows,
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
    p.add_argument("--dtype", default="float16",
                   choices=["float16", "bfloat16", "float32"])
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--vocab-size", type=int, default=50257)
    p.add_argument("--rope-mode", default="standard",
                   choices=["standard", "decoupled"])
    p.add_argument("--out", default="results/kv_memory.json")
    run(p.parse_args())
