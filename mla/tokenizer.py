"""Tokenizers.

Section 4.1 specifies the GPT-2 tokenizer with a ~50K vocabulary, which is
what `GPT2Tokenizer` provides via tiktoken. That requires a one-time download
of the BPE merge table, so a self-contained word-level fallback is included
for air-gapped environments. The fallback is always labelled in run metadata:
results produced with it are NOT comparable to the paper's perplexities,
because perplexity depends on the tokenizer.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path


class GPT2Tokenizer:
    """GPT-2 BPE via tiktoken. Matches the paper's stated setup."""

    name = "gpt2"

    def __init__(self) -> None:
        import tiktoken

        self._enc = tiktoken.get_encoding("gpt2")
        self.vocab_size = 50257
        self.eos_token_id = 50256

    def encode(self, text: str) -> list[int]:
        return self._enc.encode_ordinary(text)

    def decode(self, ids: list[int]) -> str:
        return self._enc.decode([int(i) for i in ids])


_WORD_RE = re.compile(r"\w+|[^\w\s]")


class WordTokenizer:
    """Offline fallback: whitespace/punctuation word-level vocabulary.

    The vocabulary is padded out to `vocab_size` with unused slots so that the
    model's embedding shapes are unchanged, keeping parameter counts and
    memory benchmarks comparable to a GPT-2-tokenizer run.
    """

    name = "word"

    def __init__(self, vocab_size: int = 50257) -> None:
        self.vocab_size = vocab_size
        self.eos_token_id = 0
        self.itos: list[str] = ["<|endoftext|>", "<|unk|>"]
        self.stoi: dict[str, int] = {s: i for i, s in enumerate(self.itos)}
        self.unk_token_id = 1

    def fit(self, texts, max_vocab: int | None = None) -> "WordTokenizer":
        counts: Counter[str] = Counter()
        for t in texts:
            counts.update(_WORD_RE.findall(t.lower()))
        limit = (max_vocab or self.vocab_size) - len(self.itos)
        for word, _ in counts.most_common(limit):
            self.stoi[word] = len(self.itos)
            self.itos.append(word)
        return self

    def encode(self, text: str) -> list[int]:
        return [
            self.stoi.get(w, self.unk_token_id) for w in _WORD_RE.findall(text.lower())
        ]

    def decode(self, ids: list[int]) -> str:
        words = [self.itos[i] if 0 <= i < len(self.itos) else "<|unk|>" for i in ids]
        out = " ".join(words)
        out = re.sub(r" ([.,!?;:'])", r"\1", out)
        return out.replace("<|endoftext|>", "\n")

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"vocab_size": self.vocab_size, "itos": self.itos})
        )

    @classmethod
    def load(cls, path: str | Path) -> "WordTokenizer":
        blob = json.loads(Path(path).read_text())
        tok = cls(vocab_size=blob["vocab_size"])
        tok.itos = blob["itos"]
        tok.stoi = {s: i for i, s in enumerate(tok.itos)}
        return tok


def get_tokenizer(prefer: str = "gpt2", vocab_size: int = 50257, data_dir=None):
    """Return (tokenizer, used_fallback).

    Tries the paper's GPT-2 tokenizer first; falls back to the word-level
    tokenizer if tiktoken is unavailable or cannot fetch its BPE table.
    """
    if prefer == "gpt2":
        try:
            return GPT2Tokenizer(), False
        except Exception as exc:  # noqa: BLE001 - any failure means fall back
            print(f"[tokenizer] GPT-2 BPE unavailable ({exc.__class__.__name__}); "
                  f"falling back to word-level. Perplexities will NOT be "
                  f"comparable to the paper.")
    if data_dir is not None:
        vocab_path = Path(data_dir) / "word_vocab.json"
        if vocab_path.exists():
            return WordTokenizer.load(vocab_path), True
    return WordTokenizer(vocab_size=vocab_size), True
