#!/usr/bin/env python3
"""Quality evaluation (paper Section 5.2, Tables 3 and 5, Appendix B).

Scores generated completions on grammar, creativity and narrative consistency
and prints a table in the paper's format, with the paper's own numbers
alongside.

Two scorers:
    --scorer heuristic   deterministic, offline, no API key (default)
    --scorer judge       an LLM judge via any OpenAI-compatible endpoint,
                         which is the faithful reproduction of the paper

Examples
--------
    python scripts/generate.py --runs runs/*/best.pt --out results/samples.json
    python scripts/eval_judge.py --samples results/samples.json
    OPENAI_API_KEY=... python scripts/eval_judge.py --samples results/samples.json \
        --scorer judge --judge-model gpt-4o
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mla.scoring import aggregate, heuristic_scores, judge_scores  # noqa: E402

#: Table 5: GPT-4 scores for 12L-1024d at every compression ratio.
PAPER_TABLE5 = {
    "mha@1.0":        (7.1, 5.9, 5.6, 6.2),
    "mla@1.0":        (6.5, 5.2, 4.8, 5.5),
    "mla@0.5":        (6.2, 5.0, 4.5, 5.2),
    "mla@0.25":       (5.8, 4.6, 3.9, 4.8),
    "mla@0.125":      (5.0, 3.8, 3.2, 4.0),
    "mla@0.0625":     (3.5, 2.5, 2.0, 2.7),
    "mla@0.03125":    (2.0, 1.5, 1.2, 1.6),
    "mla_rope@1.0":   (7.8, 7.2, 7.3, 7.4),
    "mla_rope@0.5":   (7.6, 7.0, 7.1, 7.2),
    "mla_rope@0.25":  (7.2, 6.5, 6.6, 6.8),
    "mla_rope@0.125": (6.5, 5.8, 5.7, 6.0),
    "mla_rope@0.0625":(4.8, 4.0, 3.5, 4.1),
    "mla_rope@0.03125":(2.8, 2.2, 1.8, 2.3),
}


def paper_key(cfg: dict) -> str:
    attn = cfg["attention"]
    if attn == "mha":
        return "mha@1.0"
    head_dim = cfg["d_model"] // cfg["n_head"]
    r = cfg.get("latent_dim") or round(head_dim * cfg["latent_ratio"])
    return f"{attn}@{r / head_dim}"


def audit_paper_claims() -> None:
    """Check the paper's stated conclusions against its own Table 5."""
    print("\nAudit of the paper's claims against its own Table 5")

    rope_half = PAPER_TABLE5["mla_rope@0.5"][3]
    rope_full = PAPER_TABLE5["mla_rope@1.0"][3]
    print(f"  Sec 5.2: 'MLA+RoPE (r=d/2) achieves the highest scores in all "
          f"dimensions'")
    print(f"    r=d/2 overall {rope_half}, r=d overall {rope_full} "
          f"-> {'HOLDS' if rope_half > rope_full else 'CONTRADICTED: r=d is higher'}")

    gap_full = PAPER_TABLE5["mla_rope@1.0"][3] - PAPER_TABLE5["mla@1.0"][3]
    gap_16 = PAPER_TABLE5["mla_rope@0.0625"][3] - PAPER_TABLE5["mla@0.0625"][3]
    print(f"  App B: 'the gap expands' from ~2 points at r=d to 1.4 at r=d/16")
    print(f"    measured gap: {gap_full:.1f} at r=d, {gap_16:.1f} at r=d/16 "
          f"-> {'widens' if gap_16 > gap_full else 'NARROWS, contradicting the text'}")

    retain = PAPER_TABLE5["mla_rope@0.5"][3] / PAPER_TABLE5["mla_rope@1.0"][3]
    print(f"  App B: 'at r=d/2 the model retains 97% of its original score'")
    print(f"    {rope_half}/{rope_full} = {100*retain:.1f}% -> HOLDS")


def run(args) -> dict:
    blob = json.loads(Path(args.samples).read_text())
    results = {}

    for name, run_blob in blob["runs"].items():
        samples = run_blob["samples"][: args.limit] if args.limit else run_blob["samples"]
        scored = []
        for i, s in enumerate(samples):
            if args.scorer == "judge":
                try:
                    scored.append(
                        judge_scores(
                            s["prompt"], s["completion"],
                            model=args.judge_model,
                            base_url=args.base_url,
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"[eval] judge failed on sample {i} "
                          f"({exc.__class__.__name__}); using heuristic for it")
                    scored.append(heuristic_scores(s["completion"]))
            else:
                scored.append(heuristic_scores(s["completion"]))

        agg = aggregate(scored)
        agg["config"] = run_blob["config"]
        agg["val_loss"] = run_blob.get("val_loss")
        agg["latent_dim"] = run_blob.get("latent_dim")
        agg["paper_key"] = paper_key(run_blob["config"])
        agg["paper_scores"] = PAPER_TABLE5.get(agg["paper_key"])
        results[name] = agg
        print(f"[eval] {name}: overall {agg['overall']['mean']:.2f} "
              f"(n={agg['n']})")

    # -- Table 3 / 5 format ---------------------------------------------
    print(f"\nQuality scores, scorer={args.scorer}"
          f"{' (' + args.judge_model + ')' if args.scorer == 'judge' else ''}")
    if args.scorer == "heuristic":
        print("NOTE: heuristic proxy scores. Absolute values are not comparable "
              "to the paper's GPT-4 numbers; the ordering between variants is "
              "what this measures.")
    hdr = (f"{'Model':<26} {'Grammar':>9} {'Creativ':>9} {'Consist':>9} "
           f"{'Overall':>9} {'ValLoss':>9} {'paper ovr':>10}")
    print(hdr)
    print("-" * len(hdr))
    for name, a in sorted(results.items(), key=lambda kv: -kv[1]["overall"]["mean"]):
        pv = f"{a['paper_scores'][3]:.1f}" if a["paper_scores"] else "-"
        vl = f"{a['val_loss']:.4f}" if a.get("val_loss") is not None else "-"
        print(f"{name:<26} "
              f"{a['grammar']['mean']:>6.2f} ({a['grammar']['std']:>.1f}) "
              f"{a['creativity']['mean']:>6.2f} ({a['creativity']['std']:>.1f}) "
              f"{a['consistency']['mean']:>6.2f} ({a['consistency']['std']:>.1f}) "
              f"{a['overall']['mean']:>9.2f} {vl:>9} {pv:>10}")

    # -- Section 5.2's correlation claim ---------------------------------
    pairs = [(a["val_loss"], a["overall"]["mean"]) for a in results.values()
             if a.get("val_loss") is not None]
    if len(pairs) >= 3:
        n = len(pairs)
        mx = sum(p[0] for p in pairs) / n
        my = sum(p[1] for p in pairs) / n
        num = sum((x - mx) * (y - my) for x, y in pairs)
        den = (sum((x - mx) ** 2 for x, _ in pairs)
               * sum((y - my) ** 2 for _, y in pairs)) ** 0.5
        r = num / den if den else 0.0
        print(f"\nSec 5.2 claims quality scores correlate with validation loss.")
        print(f"  Pearson r between val loss and overall score: {r:+.3f} "
              f"(n={n}; negative is the expected direction)")

    audit_paper_claims()

    out = {"scorer": args.scorer, "judge_model": args.judge_model, "results": results}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"\n[eval] written to {args.out}")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--samples", default="results/samples.json")
    p.add_argument("--scorer", default="heuristic", choices=["heuristic", "judge"])
    p.add_argument("--judge-model", default="gpt-4o")
    p.add_argument("--base-url", default=None,
                   help="any OpenAI-compatible endpoint")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", default="results/quality.json")
    run(p.parse_args())
