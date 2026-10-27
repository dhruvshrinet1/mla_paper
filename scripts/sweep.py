#!/usr/bin/env python3
"""Train a whole variant matrix and assemble Table 2 / Table 4 (paper Section 5.1, Appendix A).

Examples
--------
    # Table 2: one size, MHA plus both MLA variants at six compression ratios
    python scripts/sweep.py --table 2 --size 6L-512d

    # Appendix A: all eight sizes
    python scripts/sweep.py --table 4

    # cheap sanity sweep
    python scripts/sweep.py --table 2 --size 6L-256d --max-steps 500 \
        --batch-size 16 --micro-batch-size 16 --block-size 128
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mla.config import PAPER_RATIOS, TABLE1  # noqa: E402
from scripts.train import build_parser, train  # noqa: E402

#: Table 2 validation losses. Labelled 9L-512d in the paper, but these are the
#: 6L-512d row of Table 4 -- see PAPER_ANALYSIS.md issue #5.
PAPER_TABLE2 = {
    ("mha", 1.0): 2.147,
    ("mla", 1.0): 2.216, ("mla", 0.5): 2.194, ("mla", 0.25): 2.298,
    ("mla", 0.125): 2.443, ("mla", 0.0625): 2.716, ("mla", 0.03125): 3.142,
    ("mla_rope", 1.0): 2.102, ("mla_rope", 0.5): 2.154, ("mla_rope", 0.25): 2.241,
    ("mla_rope", 0.125): 2.368, ("mla_rope", 0.0625): 2.547,
    ("mla_rope", 0.03125): 2.891,
}

#: Appendix A, Table 4: validation loss for every size at r=d and r=d/2.
PAPER_TABLE4 = {
    "6L-256d":   {"mha": 2.584, "mla@1.0": 2.642, "mla@0.5": 2.621,
                  "mla_rope@1.0": 2.537, "mla_rope@0.5": 2.569},
    "6L-512d":   {"mha": 2.147, "mla@1.0": 2.216, "mla@0.5": 2.194,
                  "mla_rope@1.0": 2.102, "mla_rope@0.5": 2.154},
    "9L-256d":   {"mha": 2.423, "mla@1.0": 2.487, "mla@0.5": 2.465,
                  "mla_rope@1.0": 2.381, "mla_rope@0.5": 2.409},
    "9L-512d":   {"mha": 1.983, "mla@1.0": 2.044, "mla@0.5": 2.026,
                  "mla_rope@1.0": 1.942, "mla_rope@0.5": 1.979},
    "12L-256d":  {"mha": 2.298, "mla@1.0": 2.359, "mla@0.5": 2.341,
                  "mla_rope@1.0": 2.257, "mla_rope@0.5": 2.284},
    "12L-512d":  {"mha": 1.847, "mla@1.0": 1.896, "mla@0.5": 1.881,
                  "mla_rope@1.0": 1.819, "mla_rope@0.5": 1.842},
    "12L-768d":  {"mha": 1.765, "mla@1.0": 1.808, "mla@0.5": 1.794,
                  "mla_rope@1.0": 1.742, "mla_rope@0.5": 1.761},
    "12L-1024d": {"mha": 1.623, "mla@1.0": 1.658, "mla@0.5": 1.647,
                  "mla_rope@1.0": 1.614, "mla_rope@0.5": 1.625},
}

RATIO_LABEL = {1.0: "r = d", 0.5: "r = d/2", 0.25: "r = d/4",
               0.125: "r = d/8", 0.0625: "r = d/16", 0.03125: "r = d/32"}


def variant_list(table: int, size: str, ratios: list[float]):
    if table == 2:
        out = [(size, "mha", 1.0)]
        for attn in ("mla", "mla_rope"):
            out += [(size, attn, r) for r in ratios]
        return out
    out = []
    for sz in TABLE1:
        out.append((sz, "mha", 1.0))
        for attn in ("mla", "mla_rope"):
            out += [(sz, attn, r) for r in (1.0, 0.5)]
    return out


def run(args) -> dict:
    variants = variant_list(args.table, args.size, args.ratios)
    print(f"[sweep] {len(variants)} variants, {args.max_steps} steps each\n")

    base = build_parser().parse_args([])
    results = []
    for i, (size, attn, ratio) in enumerate(variants, 1):
        print(f"\n[sweep] ({i}/{len(variants)}) {size} {attn} r={ratio}")
        cfg = argparse.Namespace(**vars(base))
        cfg.size, cfg.attention, cfg.latent_ratio = size, attn, ratio
        for key in ("max_steps", "batch_size", "micro_batch_size", "block_size",
                    "lr", "dropout", "seed", "data_dir", "out_dir", "device",
                    "dtype", "eval_interval", "eval_iters", "log_interval",
                    "rope_mode", "kv_cache_mode"):
            setattr(cfg, key, getattr(args, key))
        cfg.run_name = None
        try:
            results.append(train(cfg))
        except Exception as exc:  # noqa: BLE001
            print(f"[sweep] FAILED: {exc.__class__.__name__}: {exc}")
            results.append({"size": size, "attention": attn,
                            "latent_ratio": ratio, "error": str(exc)})

    report(results, args)

    out = {"table": args.table, "size": args.size, "results": results}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"\n[sweep] written to {args.out}")
    return out


def report(results: list[dict], args) -> None:
    ok = [r for r in results if "error" not in r]
    if not ok:
        print("\n[sweep] nothing completed")
        return

    baseline = next(
        (r["best_val_loss"] for r in ok if r["attention"] == "mha"), None
    )
    kv_base = next(
        (r["kv_elements_per_token"] for r in ok if r["attention"] == "mha"), None
    )

    print(f"\n\nTable 2 reproduction, {args.size}, {args.max_steps} steps")
    if args.max_steps < 50_000:
        print(f"NOTE: the paper trains 50,000 steps. At {args.max_steps} steps "
              f"absolute losses are higher; the comparison between variants is "
              f"what this measures.")
    hdr = (f"{'Model':<10} {'Latent':>9} {'ValLoss':>9} {'vs MHA':>9} "
           f"{'KV elem/tok':>12} {'KV vs MHA':>10} {'paper':>8} {'paper d%':>9}")
    print(hdr)
    print("-" * len(hdr))
    for r in ok:
        lbl = RATIO_LABEL.get(round(r["latent_ratio"], 5), f"{r['latent_ratio']:.3f}")
        lbl = "—" if r["attention"] == "mha" else lbl
        delta = (100 * (r["best_val_loss"] - baseline) / baseline) if baseline else 0
        kv = (100 * (1 - r["kv_elements_per_token"] / kv_base)) if kv_base else 0
        pk = PAPER_TABLE2.get((r["attention"], round(r["latent_ratio"], 5)))
        pdelta = (100 * (pk - PAPER_TABLE2[("mha", 1.0)]) / PAPER_TABLE2[("mha", 1.0)]
                  if pk else None)
        print(f"{r['attention']:<10} {lbl:>9} {r['best_val_loss']:>9.4f} "
              f"{delta:>+8.1f}% {r['kv_elements_per_token']:>12,} "
              f"{kv:>9.1f}% {(f'{pk:.3f}' if pk else '-'):>8} "
              f"{(f'{pdelta:+.1f}%' if pdelta is not None else '-'):>9}")

    # -- the paper's two headline claims, checked ------------------------
    def find(attn, ratio):
        return next((r for r in ok if r["attention"] == attn
                     and abs(r["latent_ratio"] - ratio) < 1e-6), None)

    print("\nHeadline claims from the abstract, checked against this run")
    mla_full, rope_full = find("mla", 1.0), find("mla_rope", 1.0)
    rope_half = find("mla_rope", 0.5)
    if baseline and mla_full:
        d = 100 * (mla_full["best_val_loss"] - baseline) / baseline
        print(f"  'without RoPE, MLA underperforms vanilla by 3-5%': "
              f"measured {d:+.1f}%")
    if baseline and rope_full:
        d = 100 * (rope_full["best_val_loss"] - baseline) / baseline
        print(f"  'with RoPE, it surpasses vanilla by 2%': measured {d:+.1f}%")
    if baseline and rope_half:
        d = 100 * (rope_half["best_val_loss"] - baseline) / baseline
        kv = 100 * (1 - rope_half["kv_elements_per_token"] / kv_base)
        print(f"  'MLA+RoPE at r=d/2: 45% memory cut for +0.3% loss': "
              f"measured {kv:.1f}% memory cut for {d:+.1f}% loss")
        print(f"    (the memory figure is 50% by construction; the paper's 45% "
              f"comes from a table cell that breaks O(r) scaling)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--table", type=int, default=2, choices=[2, 4])
    p.add_argument("--size", default="6L-512d",
                   help="Table 2 is labelled 9L-512d but its values are 6L-512d")
    p.add_argument("--ratios", type=float, nargs="+", default=PAPER_RATIOS)
    p.add_argument("--max-steps", type=int, default=50_000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--micro-batch-size", type=int, default=16)
    p.add_argument("--block-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--eval-interval", type=int, default=1000)
    p.add_argument("--eval-iters", type=int, default=100)
    p.add_argument("--log-interval", type=int, default=200)
    p.add_argument("--rope-mode", default="standard")
    p.add_argument("--kv-cache-mode", default="latent")
    p.add_argument("--data-dir", default="data/tinystories")
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto")
    p.add_argument("--out", default="results/sweep.json")
    run(p.parse_args())
