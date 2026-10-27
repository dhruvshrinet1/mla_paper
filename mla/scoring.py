"""Quality scoring for generated stories (paper Section 5.2, Tables 3 and 5).

The paper scores 100 completions per model with GPT-4 on grammar, creativity
and narrative consistency, 1-10. It does not publish the judge prompt, the
rubric anchors, the sampling settings or the model snapshot
(PAPER_ANALYSIS.md issue #13), so the prompt here is written from the
rubric dimensions the paper names.

Two scorers are provided:

`judge_scores`     an LLM judge through any OpenAI-compatible endpoint. This
                   is the faithful reproduction path.

`heuristic_scores` a deterministic offline scorer. It is NOT a substitute for
                   a judge model and its absolute values are not comparable to
                   the paper's. It is calibrated against the failure modes the
                   paper describes qualitatively in Section 5.3 and Appendix C
                   -- repetition at r=d/16, degenerate token sequences at
                   r=d/32 -- so it separates those regimes reliably and lets
                   the pipeline run with no API key.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, asdict

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[A-Za-z']+")


@dataclass
class StoryScores:
    grammar: float
    creativity: float
    consistency: float

    @property
    def overall(self) -> float:
        return (self.grammar + self.creativity + self.consistency) / 3.0

    def as_dict(self) -> dict:
        d = asdict(self)
        d["overall"] = self.overall
        return d


# --------------------------------------------------------------------------
# offline heuristic scorer
# --------------------------------------------------------------------------


def _distinct_n(tokens: list[str], n: int) -> float:
    if len(tokens) < n:
        return 0.0
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return len(set(grams)) / len(grams)


def _adjacent_repeat_rate(tokens: list[str]) -> float:
    """Fraction of positions where a token equals its predecessor.

    This is the signature of the paper's r=d/32 failure mode ("the the the").
    """
    if len(tokens) < 2:
        return 0.0
    return sum(a == b for a, b in zip(tokens, tokens[1:])) / (len(tokens) - 1)


_FUNCTION_WORDS = frozenset(
    "a an the and or but if then so of to in on at for with from by as is was "
    "are were be been being it its he she they them his her their there that "
    "this these those i you we not no yes do did does had has have will would "
    "can could up out down over all very into when who what which".split()
)


def _near_repeat_rate(tokens: list[str], window: int = 3) -> float:
    """Fraction of content words that recur within the next `window` tokens.

    This is the signature of the paper's MLA-without-RoPE failure mode:
    "liked carrots and carrots", "A fox and fox came". Strict adjacency misses
    it because an intervening function word breaks the pair.
    """
    content = [(i, t) for i, t in enumerate(tokens) if t not in _FUNCTION_WORDS]
    if len(content) < 2:
        return 0.0
    hits = 0
    for pos, (i, tok) in enumerate(content):
        for j, other in content[pos + 1 :]:
            if j - i > window:
                break
            if other == tok:
                hits += 1
                break
    return hits / len(content)


def _content_ratio(tokens: list[str]) -> float:
    """Share of tokens carrying lexical content.

    The paper's r=d/32 outputs ("the the the the. And and and.") are almost
    entirely function words. Repetition metrics alone miss this, because a
    text with no content words has no content-word repetition.
    """
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if t not in _FUNCTION_WORDS) / len(tokens)


def _hapax_ratio(tokens: list[str]) -> float:
    """Share of content words used exactly once. Less saturating than TTR."""
    content = [t for t in tokens if t not in _FUNCTION_WORDS]
    if not content:
        return 0.0
    counts = Counter(content)
    return sum(1 for t in counts.values() if t == 1) / len(counts)


def _max_ngram_repeat(tokens: list[str], n: int = 4) -> float:
    """How dominant the most-repeated n-gram is: the r=d/16 loop signature."""
    if len(tokens) < n + 1:
        return 0.0
    grams = Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))
    return grams.most_common(1)[0][1] / max(len(grams), 1)


def _scale(value: float, lo: float, hi: float) -> float:
    """Map a raw metric onto 1-10, clamped."""
    if hi == lo:
        return 5.5
    x = (value - lo) / (hi - lo)
    return max(1.0, min(10.0, 1.0 + 9.0 * x))


def heuristic_scores(text: str) -> StoryScores:
    """Deterministic 1-10 proxy scores. See the module docstring for caveats."""
    tokens = _WORD.findall(text.lower())
    if not tokens:
        return StoryScores(1.0, 1.0, 1.0)

    sentences = [s.strip() for s in _SENT_SPLIT.split(text.strip()) if s.strip()]
    adj_rep = _adjacent_repeat_rate(tokens)
    near_rep = _near_repeat_rate(tokens)
    d1, d2, d3 = (_distinct_n(tokens, i) for i in (1, 2, 3))
    loop = _max_ngram_repeat(tokens, 4)
    hapax = _hapax_ratio(tokens)
    content = _content_ratio(tokens)

    # Grammar: sentences that start with a capital and end with terminal
    # punctuation, have a plausible length, and avoid repeats.
    #
    # Some detokenizers (the offline word-level fallback among them) emit
    # lowercase-only text. Scoring capitalisation then measures the tokenizer,
    # not the model, so the criterion is dropped when the whole text is
    # lowercase.
    has_case = any(c.isupper() for c in text)
    well_formed = 0
    for s in sentences:
        words = [w.lower() for w in _WORD.findall(s)]
        ok = (
            (not has_case or bool(s[:1].isupper()))
            and s[-1:] in ".!?"
            and 4 <= len(words) <= 40
            and _adjacent_repeat_rate(words) < 0.08
            and _near_repeat_rate(words) < 0.20
        )
        well_formed += int(ok)
    form_ratio = well_formed / max(len(sentences), 1)
    grammar = (
        0.55 * _scale(form_ratio, 0.0, 1.0)
        + 0.25 * _scale(1 - adj_rep, 0.70, 1.0)
        + 0.20 * _scale(1 - near_rep, 0.70, 1.0)
    )

    # Creativity: lexical variety. Distinct-n and the hapax ratio collapse
    # under compression, which is what Section 5.3 reports qualitatively.
    creativity = (
        0.25 * _scale(d2, 0.65, 1.0)
        + 0.20 * _scale(d3, 0.80, 1.0)
        + 0.35 * _scale(hapax, 0.35, 0.92)
        + 0.20 * _scale(content, 0.25, 0.60)
    )

    # Consistency: absence of loops and of near repetition, sustained variety
    # across the whole span, and enough distinct tokens to carry a narrative.
    half = max(len(tokens) // 2, 1)
    drift = abs(_distinct_n(tokens[:half], 2) - _distinct_n(tokens[half:], 2))
    consistency = (
        0.25 * _scale(1 - loop, 0.70, 1.0)
        + 0.25 * _scale(1 - near_rep, 0.72, 1.0)
        + 0.15 * _scale(d1, 0.45, 0.85)
        + 0.15 * _scale(1 - drift, 0.75, 1.0)
        + 0.20 * _scale(content, 0.25, 0.60)
    )

    return StoryScores(
        grammar=round(grammar, 3),
        creativity=round(creativity, 3),
        consistency=round(consistency, 3),
    )


# --------------------------------------------------------------------------
# LLM judge
# --------------------------------------------------------------------------

JUDGE_SYSTEM = (
    "You are evaluating short children's stories produced by small language "
    "models. Score strictly and use the full 1-10 range."
)

JUDGE_TEMPLATE = """Rate the story continuation below on three dimensions, each 1-10.

grammar: syntactic correctness, well-formed sentences, correct agreement and punctuation.
creativity: originality of imagery, events and word choice, as opposed to formulaic filler.
consistency: narrative coherence. Are characters, objects and timeline tracked without contradiction, abrupt shifts or repetition loops?

Anchors: 1 means degenerate or unreadable, 4 means noticeably flawed but followable, 7 means competent, 10 means excellent.

PROMPT:
{prompt}

CONTINUATION:
{completion}

Reply with JSON only, exactly this shape:
{{"grammar": <number>, "creativity": <number>, "consistency": <number>}}"""


def judge_scores(
    prompt: str,
    completion: str,
    model: str = "gpt-4o",
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: float = 60.0,
) -> StoryScores:
    """Score one completion with an OpenAI-compatible chat endpoint."""
    from openai import OpenAI

    client = OpenAI(
        api_key=api_key or os.environ.get("OPENAI_API_KEY"),
        base_url=base_url or os.environ.get("OPENAI_BASE_URL") or None,
        timeout=timeout,
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {
                "role": "user",
                "content": JUDGE_TEMPLATE.format(
                    prompt=prompt, completion=completion
                ),
            },
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    blob = json.loads(resp.choices[0].message.content)
    return StoryScores(
        grammar=float(blob["grammar"]),
        creativity=float(blob["creativity"]),
        consistency=float(blob["consistency"]),
    )


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------


def aggregate(scores: list[StoryScores]) -> dict:
    """Mean and standard deviation, as Table 3 reports them."""
    if not scores:
        return {}

    def stats(values: list[float]) -> dict:
        mean = sum(values) / len(values)
        if len(values) < 2:
            return {"mean": round(mean, 2), "std": 0.0}
        var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        return {"mean": round(mean, 2), "std": round(math.sqrt(var), 2)}

    return {
        "n": len(scores),
        "grammar": stats([s.grammar for s in scores]),
        "creativity": stats([s.creativity for s in scores]),
        "consistency": stats([s.consistency for s in scores]),
        "overall": stats([s.overall for s in scores]),
    }
