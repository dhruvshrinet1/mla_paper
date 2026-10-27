#!/usr/bin/env python3
"""Sample story completions from trained checkpoints (paper Section 5.3, Appendix C).

Appendix C compares variants on one fixed prompt. This script does the same,
running every checkpoint on identical prompts with identical seeds so the
comparison is apples to apples.

Examples
--------
    python scripts/generate.py --runs runs/*/best.pt
    python scripts/generate.py --runs runs/*/best.pt --n-samples 100 --out results/samples.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from mla.config import ModelConfig  # noqa: E402
from mla.model import build_model  # noqa: E402
from mla.tokenizer import WordTokenizer, get_tokenizer  # noqa: E402

#: Appendix C's prompt, plus a few more so quality estimates are not
#: conditioned on a single opening.
DEFAULT_PROMPTS = [
    "Once upon a time, there was a little rabbit who lived in",
    "One morning, a curious fox found",
    "There was a brave little mouse named",
    "Long ago, a gentle bear lived near",
]


def load_checkpoint(path: str | Path, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ModelConfig(**ckpt["model_config"])
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg, ckpt


def resolve_tokenizer(data_dir: str | Path):
    tok, fallback = get_tokenizer("gpt2", data_dir=data_dir)
    if fallback:
        vocab_path = Path(data_dir) / "word_vocab.json"
        if vocab_path.exists():
            tok = WordTokenizer.load(vocab_path)
    return tok


def run(args) -> dict:
    device = torch.device(
        args.device if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    tok = resolve_tokenizer(args.data_dir)
    prompts = args.prompts or DEFAULT_PROMPTS

    out = {"device": str(device), "tokenizer": tok.name, "runs": {}}
    for ckpt_path in args.runs:
        model, cfg, ckpt = load_checkpoint(ckpt_path, device)
        samples = []
        for i in range(args.n_samples):
            prompt = prompts[i % len(prompts)]
            torch.manual_seed(args.seed + i)   # same seed per index across models
            ids = torch.tensor(
                [tok.encode(prompt)], dtype=torch.long, device=device
            )
            gen = model.generate(
                ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                use_cache=True,
                eos_token_id=tok.eos_token_id,
            )
            completion = tok.decode(gen[0, ids.shape[1] :].tolist())
            samples.append({"prompt": prompt, "completion": completion.strip()})

        key = cfg.name
        out["runs"][key] = {
            "checkpoint": str(ckpt_path),
            "config": cfg.to_dict(),
            "val_loss": ckpt.get("val_loss"),
            "latent_dim": cfg.effective_latent_dim,
            "samples": samples,
        }
        print(f"[gen] {key}: {len(samples)} samples")
        if args.show:
            print(f"      prompt: {samples[0]['prompt']}")
            print(f"      ---> {samples[0]['completion'][:220]}\n")
        del model

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"[gen] written to {args.out}")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True, help="paths to best.pt files")
    p.add_argument("--prompts", nargs="*", default=None)
    p.add_argument("--n-samples", type=int, default=100,
                   help="Section 5.2 uses 100 completions per model")
    p.add_argument("--max-new-tokens", type=int, default=120)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=200)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--data-dir", default="data/tinystories")
    p.add_argument("--device", default="auto")
    p.add_argument("--show", action="store_true")
    p.add_argument("--out", default="results/samples.json")
    run(p.parse_args())
