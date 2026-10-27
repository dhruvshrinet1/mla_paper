#!/usr/bin/env python3
"""Reproduce Figure 1 (paper Section 5.4) from benchmark JSON.

Figure 1a: throughput vs batch size for each latent dimension
Figure 1b: KV-cache memory vs latent dimension ratio r/d_k

Both panels overlay the paper's reported values so the comparison is direct.

Example
-------
    python scripts/bench_memory.py --out results/kv_memory.json
    python scripts/bench_throughput.py --out results/throughput.json
    python scripts/make_figures.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LABELS = {
    ("mha", 1.0): "MHA",
    ("mla", 1.0): "MLA (r=d)",
    ("mla", 0.5): "MLA (r=d/2)",
    ("mla", 0.25): "MLA (r=d/4)",
    ("mla", 0.125): "MLA (r=d/8)",
    ("mla_rope", 1.0): "MLA+RoPE (r=d)",
    ("mla_rope", 0.5): "MLA+RoPE (r=d/2)",
    ("mla_rope", 0.25): "MLA+RoPE (r=d/4)",
}
COLORS = {
    "mha": "#d95f02", "mla": "#e7298a", "mla_rope": "#1b9e77",
}
STYLES = {1.0: "-", 0.5: "--", 0.25: ":", 0.125: "-."}


def figure_1a(path: Path, out: Path) -> None:
    blob = json.loads(path.read_text())
    rows = blob["results"]
    series: dict[tuple[str, float], list[tuple[int, float]]] = {}
    for r in rows:
        series.setdefault((r["attention"], r["latent_ratio"]), []).append(
            (r["batch_size"], r["tokens_per_sec"])
        )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for key, pts in sorted(series.items()):
        pts.sort()
        xs, ys = zip(*pts)
        ax.plot(
            xs, ys, marker="o", markersize=4,
            color=COLORS.get(key[0], "#666"),
            linestyle=STYLES.get(key[1], "-"),
            label=LABELS.get(key, f"{key[0]} r={key[1]}"),
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(blob["batch_sizes"])
    ax.set_xticklabels([str(b) for b in blob["batch_sizes"]])
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Throughput (tokens/second)")
    ax.set_title(
        f"Throughput vs Batch Size ({blob['size']}, {blob['device']})\n"
        f"paper Figure 1a, measured on {blob['device']} not A100",
        fontsize=10,
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"[fig] {out}")


def figure_1b(path: Path, out: Path) -> None:
    blob = json.loads(path.read_text())
    rows = [r for r in blob["rows"] if r["attention"] == "mla"]
    rows.sort(key=lambda r: r["latent_ratio"])
    xs = [r["latent_ratio"] for r in rows]
    ours = [r["analytical_mb_per_token"] for r in rows]
    paper = [r["paper_mb_per_token"] for r in rows]
    mha = next(r for r in blob["rows"] if r["attention"] == "mha")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(xs, ours, marker="o", color="#e7298a",
            label="this implementation (exactly O(r))")
    ax.plot(xs, paper, marker="s", color="#7570b3", linestyle="--",
            label="paper Table 2")
    ax.axhline(mha["analytical_mb_per_token"], color="#1f78b4",
               linestyle=":", label="MHA baseline (ours)")
    ax.axhline(mha["paper_mb_per_token"], color="#999",
               linestyle=":", label="MHA baseline (paper)")

    # mark the point where the paper's own column stops scaling as O(r)
    half = next(r for r in rows if abs(r["latent_ratio"] - 0.5) < 1e-6)
    expected = mha["paper_mb_per_token"] / 2
    ax.annotate(
        f"paper Table 2 reads {half['paper_mb_per_token']:.4f} here;\n"
        f"O(r) scaling implies {expected:.4f}.\n"
        f"This one cell is the source of the\n'45%' headline (true value: 50%).",
        xy=(0.5, half["paper_mb_per_token"]),
        xytext=(0.042, mha["paper_mb_per_token"] * 0.52),
        fontsize=7.5,
        arrowprops=dict(arrowstyle="->", color="crimson", lw=1,
                        connectionstyle="arc3,rad=-0.2"),
        color="crimson",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="crimson", lw=0.6,
                  alpha=0.9),
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"d/{round(1/x)}" if x < 1 else "d" for x in xs])
    ax.set_xlabel("Latent Dimension Ratio (r / d_k)")
    ax.set_ylabel("KV-Cache Memory (MB/token)")
    ax.set_title(
        f"KV-cache memory vs latent dimension ({blob['size']}, {blob['dtype']})\n"
        f"paper Figure 1b",
        fontsize=10,
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_ylim(0, mha["paper_mb_per_token"] * 1.15)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"[fig] {out}")


def figure_quality(path: Path, out: Path) -> None:
    blob = json.loads(path.read_text())
    items = [(k, v) for k, v in blob["results"].items()
             if v.get("val_loss") is not None]
    if len(items) < 2:
        print("[fig] skipping quality plot: need val losses for 2+ runs")
        return
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, v in items:
        attn = v["config"]["attention"]
        ax.scatter(v["val_loss"], v["overall"]["mean"],
                   color=COLORS.get(attn, "#666"), s=45)
        ax.annotate(name, (v["val_loss"], v["overall"]["mean"]),
                    fontsize=6, xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("Validation loss")
    ax.set_ylabel(f"Overall quality score ({blob['scorer']})")
    ax.set_title("Section 5.2: quality against perplexity", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"[fig] {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--throughput", default="results/throughput.json")
    p.add_argument("--memory", default="results/kv_memory.json")
    p.add_argument("--quality", default="results/quality.json")
    p.add_argument("--out-dir", default="results")
    a = p.parse_args()
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for src, fn, name in (
        (a.throughput, figure_1a, "figure1a_throughput.png"),
        (a.memory, figure_1b, "figure1b_kv_memory.png"),
        (a.quality, figure_quality, "quality_vs_loss.png"),
    ):
        if Path(src).exists():
            fn(Path(src), out_dir / name)
        else:
            print(f"[fig] skipping {name}: {src} not found")
