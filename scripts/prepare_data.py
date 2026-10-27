#!/usr/bin/env python3
"""Build the training corpus (paper Section 4.1: 100K TinyStories-style stories).

Examples
--------
    # the real dataset (needs network + `pip install datasets`)
    python scripts/prepare_data.py --source hf --n-stories 100000

    # offline, deterministic stand-in
    python scripts/prepare_data.py --source synthetic --n-stories 100000

    # your own corpus
    python scripts/prepare_data.py --source local --local-path ./my_corpus
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mla.data import prepare  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default="data/tinystories")
    p.add_argument("--source", default="hf", choices=["hf", "local", "synthetic"],
                   help="hf falls back to synthetic if the dataset is unreachable")
    p.add_argument("--local-path", default=None)
    p.add_argument("--n-stories", type=int, default=100_000,
                   help="Section 4.1 uses 100,000")
    p.add_argument("--val-fraction", type=float, default=0.01)
    p.add_argument("--tokenizer", default="gpt2", choices=["gpt2", "word"],
                   help="gpt2 is the paper's choice; word is an offline fallback")
    p.add_argument("--vocab-size", type=int, default=50257)
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()

    meta = prepare(
        out_dir=args.out_dir,
        source=args.source,
        n_stories=args.n_stories,
        val_fraction=args.val_fraction,
        local_path=args.local_path,
        tokenizer_pref=args.tokenizer,
        vocab_size=args.vocab_size,
        seed=args.seed,
    )
    print(f"\n[data] written to {args.out_dir}")
    if meta["tokenizer_is_fallback"]:
        print("[data] NOTE: word-level fallback tokenizer in use. Losses are "
              "comparable across attention variants but not to the paper.")
    if meta["source"] == "synthetic":
        print("[data] NOTE: synthetic corpus in use. Same caveat applies.")


if __name__ == "__main__":
    main()
