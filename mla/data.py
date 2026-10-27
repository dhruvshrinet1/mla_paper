"""Corpus preparation (paper Section 4.1).

The paper trains on "a TinyStories-style dataset of 100K synthetic children's
stories". Note Table 1's caption instead says "a corpus of 10,000 books" --
see PAPER_ANALYSIS.md issue #2. This module follows Section 4.1.

Three sources, tried in order:

1. `--source hf`      the real TinyStories dataset via `datasets`
2. `--source local`   a directory of .txt / .jsonl files
3. `--source synthetic` a deterministic template generator, so the pipeline
                        is runnable with no network access

The synthetic source is a stand-in, not a replica of TinyStories. Losses from
it are internally comparable across attention variants (which is what the
paper's claims are about) but are not comparable to the paper's absolute
numbers.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from .tokenizer import WordTokenizer, get_tokenizer

# --------------------------------------------------------------------------
# synthetic TinyStories-style generator
# --------------------------------------------------------------------------

NAMES = [
    "Lily", "Tom", "Sara", "Ben", "Mia", "Max", "Anna", "Leo", "Zoe", "Sam",
    "Ella", "Finn", "Nora", "Jack", "Ruby", "Oscar", "Ivy", "Theo", "Luna", "Milo",
]
ANIMALS = [
    "rabbit", "fox", "owl", "cat", "dog", "mouse", "bear", "duck", "frog",
    "squirrel", "turtle", "bird", "deer", "hedgehog", "kitten", "puppy",
]
PLACES = [
    "a cozy burrow under the old oak tree", "a small house near the river",
    "a bright meadow full of wildflowers", "a quiet garden behind the hill",
    "a warm nest high in a tall tree", "a little cave beside the pond",
    "a wooden cabin at the edge of the woods", "a sunny field of tall grass",
]
OBJECTS = [
    "a shiny blue stone", "a red ball", "an old brass key", "a silver feather",
    "a tiny glass bottle", "a wooden boat", "a golden ring", "a soft wool hat",
    "a glowing pebble", "a torn paper map",
]
ADJECTIVES = [
    "happy", "curious", "brave", "sleepy", "kind", "clever", "gentle",
    "cheerful", "quiet", "friendly", "careful", "proud",
]
FEELINGS = ["happy", "excited", "proud", "calm", "glad", "surprised", "thankful"]
WEATHER = [
    "The sun was shining brightly.", "A soft rain began to fall.",
    "The wind blew through the leaves.", "Mist covered the ground.",
    "The sky was clear and blue.", "Snow fell quietly all morning.",
]

OPENINGS = [
    "Once upon a time, there was a little {animal} who lived in {place}.",
    "There was a {adj} little {animal} named {name}.",
    "Long ago, a small {animal} lived in {place}.",
    "One morning, a {adj} {animal} named {name} woke up early.",
]
MIDDLES = [
    "{name} wanted to find {object}.",
    "One day, {name} found {object} near the path.",
    "{name} met a {adj} {animal2} who needed help.",
    "{name} decided to walk to the top of the hill.",
    "A {adj} {animal2} came to visit {name}.",
]
EVENTS = [
    "They looked everywhere, under rocks and behind trees.",
    "{name} was a little scared, but kept going anyway.",
    "The {animal2} showed {name} a secret path through the grass.",
    "They shared the food they had and talked for a long time.",
    "{name} carried {object} carefully all the way home.",
    "It took a long time, but {name} did not give up.",
]
ENDINGS = [
    "In the end, {name} felt very {feeling}.",
    "{name} went home and slept well that night.",
    "From that day on, they were good friends.",
    "{name} learned that helping others feels {feeling}.",
    "Everyone was {feeling}, and the day ended well.",
]


def generate_synthetic_stories(n: int, seed: int = 1234):
    """Yield `n` simple children's stories with TinyStories-like structure."""
    rng = random.Random(seed)
    for _ in range(n):
        slots = {
            "name": rng.choice(NAMES),
            "animal": rng.choice(ANIMALS),
            "animal2": rng.choice(ANIMALS),
            "place": rng.choice(PLACES),
            "object": rng.choice(OBJECTS),
            "adj": rng.choice(ADJECTIVES),
            "feeling": rng.choice(FEELINGS),
        }
        parts = [
            rng.choice(OPENINGS).format(**slots),
            rng.choice(WEATHER),
            rng.choice(MIDDLES).format(**slots),
        ]
        for _ in range(rng.randint(2, 4)):
            parts.append(rng.choice(EVENTS).format(**slots))
        parts.append(rng.choice(ENDINGS).format(**slots))
        yield " ".join(parts)


# --------------------------------------------------------------------------
# loaders
# --------------------------------------------------------------------------


def load_hf_tinystories(n_stories: int):
    from datasets import load_dataset

    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    out = []
    for i, row in enumerate(ds):
        if i >= n_stories:
            break
        out.append(row["text"].strip())
    return out


def load_local(path: str | Path, n_stories: int):
    path = Path(path)
    files = sorted([*path.glob("*.txt"), *path.glob("*.jsonl")])
    if not files:
        raise FileNotFoundError(f"no .txt or .jsonl files under {path}")
    out = []
    for f in files:
        if f.suffix == ".jsonl":
            for line in f.read_text(encoding="utf-8").splitlines():
                if len(out) >= n_stories:
                    return out
                row = json.loads(line)
                out.append(str(row.get("text", row)).strip())
        else:
            for chunk in f.read_text(encoding="utf-8").split("\n\n"):
                if len(out) >= n_stories:
                    return out
                if chunk.strip():
                    out.append(chunk.strip())
    return out


def get_stories(source: str, n_stories: int, local_path=None, seed: int = 1234):
    """Return (stories, source_actually_used)."""
    if source == "hf":
        try:
            return load_hf_tinystories(n_stories), "hf"
        except Exception as exc:  # noqa: BLE001
            print(f"[data] HuggingFace unavailable ({exc.__class__.__name__}); "
                  f"falling back to synthetic.")
            source = "synthetic"
    if source == "local":
        return load_local(local_path, n_stories), "local"
    return list(generate_synthetic_stories(n_stories, seed=seed)), "synthetic"


# --------------------------------------------------------------------------
# preparation and loading
# --------------------------------------------------------------------------


def prepare(
    out_dir: str | Path,
    source: str = "hf",
    n_stories: int = 100_000,
    val_fraction: float = 0.01,
    local_path: str | Path | None = None,
    tokenizer_pref: str = "gpt2",
    vocab_size: int = 50257,
    seed: int = 1234,
) -> dict:
    """Tokenise a corpus into flat uint16/uint32 binary shards."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stories, used_source = get_stories(source, n_stories, local_path, seed)
    print(f"[data] {len(stories):,} stories from source={used_source}")

    # vocab_size <= 0 means "fit tightly to the corpus", which keeps smoke
    # runs cheap. The paper's setting is the full 50257.
    tight = vocab_size <= 0
    tok, fallback = get_tokenizer(
        tokenizer_pref, vocab_size if not tight else 50257
    )
    if fallback and isinstance(tok, WordTokenizer):
        tok.fit(stories)
        if tight:
            tok.vocab_size = len(tok.itos)
        tok.save(out_dir / "word_vocab.json")
        print(f"[data] word-level vocab: {len(tok.itos):,} real tokens "
              f"(padded to {tok.vocab_size:,})")

    rng = random.Random(seed)
    rng.shuffle(stories)
    n_val = max(1, int(len(stories) * val_fraction))
    splits = {"val": stories[:n_val], "train": stories[n_val:]}

    dtype = np.uint16 if tok.vocab_size < 2**16 else np.uint32
    meta = {
        "source": used_source,
        "tokenizer": tok.name,
        "tokenizer_is_fallback": fallback,
        "vocab_size": tok.vocab_size,
        "eos_token_id": tok.eos_token_id,
        "n_stories": len(stories),
        "dtype": np.dtype(dtype).name,
        "splits": {},
    }

    for split, texts in splits.items():
        ids: list[int] = []
        for t in texts:
            ids.extend(tok.encode(t))
            ids.append(tok.eos_token_id)
        arr = np.asarray(ids, dtype=dtype)
        arr.tofile(out_dir / f"{split}.bin")
        meta["splits"][split] = {"n_tokens": int(arr.size), "n_stories": len(texts)}
        print(f"[data] {split}: {arr.size:,} tokens from {len(texts):,} stories")

    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


class TokenDataset:
    """Memory-mapped flat token stream with random contiguous sampling."""

    def __init__(self, data_dir: str | Path, split: str, block_size: int) -> None:
        self.data_dir = Path(data_dir)
        meta_path = self.data_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"{meta_path} missing. Run scripts/prepare_data.py first."
            )
        self.meta = json.loads(meta_path.read_text())
        self.dtype = np.dtype(self.meta["dtype"])
        self.block_size = block_size
        self.data = np.memmap(
            self.data_dir / f"{split}.bin", dtype=self.dtype, mode="r"
        )
        if len(self.data) <= block_size + 1:
            raise ValueError(
                f"split {split!r} has {len(self.data)} tokens, needs > {block_size + 1}"
            )

    @property
    def vocab_size(self) -> int:
        return int(self.meta["vocab_size"])

    def __len__(self) -> int:
        return len(self.data) - self.block_size - 1

    def get_batch(self, batch_size: int, device, generator=None):
        import torch

        hi = len(self.data) - self.block_size - 1
        ix = torch.randint(hi, (batch_size,), generator=generator)
        x = torch.stack([
            torch.from_numpy(
                self.data[i : i + self.block_size].astype(np.int64)
            ) for i in ix
        ])
        y = torch.stack([
            torch.from_numpy(
                self.data[i + 1 : i + 1 + self.block_size].astype(np.int64)
            ) for i in ix
        ])
        if str(device).startswith("cuda"):
            return x.pin_memory().to(device, non_blocking=True), \
                   y.pin_memory().to(device, non_blocking=True)
        return x.to(device), y.to(device)
