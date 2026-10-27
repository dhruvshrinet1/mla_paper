"""Tests for the model configurations and an audit of the paper's own arithmetic.

The second half of this file does not test our code. It tests the paper's
numbers against each other, and is the executable form of PAPER_ANALYSIS.md.
Each failing-by-design assertion is written to PASS, asserting the
inconsistency that is actually present.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch

from mla.config import TABLE1, model_config
from mla.model import build_model
from mla.scoring import heuristic_scores
from scripts.bench_memory import PAPER_KV_MEM
from scripts.sweep import PAPER_TABLE2, PAPER_TABLE4
from scripts.eval_judge import PAPER_TABLE5


# --------------------------------------------------------------------------
# Table 1
# --------------------------------------------------------------------------


@pytest.mark.parametrize("size", list(TABLE1))
def test_parameter_counts_match_table1(size):
    """All eight configs must land within 1% of the paper's reported counts."""
    _, _, paper_millions = TABLE1[size]
    model = build_model(model_config(size, attention="mha"))
    total = model.num_parameters()
    without_pos = total - (model.wpe.weight.numel() if model.wpe is not None else 0)
    delta = abs(without_pos / 1e6 - paper_millions) / paper_millions
    assert delta < 0.01, (
        f"{size}: ours {without_pos/1e6:.2f}M vs paper {paper_millions}M "
        f"({100*delta:.2f}% off)"
    )


@pytest.mark.parametrize("size", list(TABLE1))
def test_head_dim_is_64_everywhere(size):
    """d_k = 64 for every config, which is what makes the r ratios uniform."""
    assert model_config(size).head_dim == 64


def test_rope_disables_learned_position_embeddings():
    """Sec 2.2: RoPE works 'without explicit position embeddings'."""
    assert build_model(model_config("6L-256d", attention="mla_rope")).wpe is None
    assert build_model(model_config("6L-256d", attention="mla")).wpe is not None


def test_embeddings_are_tied():
    """Table 1 caption specifies a tied softmax output layer."""
    model = build_model(model_config("6L-256d"))
    assert model.lm_head.weight is model.wte.weight


def test_training_step_reduces_loss():
    torch.manual_seed(0)
    cfg = model_config("6L-256d", attention="mla_rope", latent_ratio=0.5,
                       vocab_size=96, block_size=32, dropout=0.0)
    model = build_model(cfg)
    opt = model.configure_optimizers(0.1, 1e-3, (0.9, 0.95))
    x = torch.randint(0, 96, (4, 32))
    first = last = None
    for _ in range(25):
        _, loss = model(x, targets=x)
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = first if first is not None else loss.item()
        last = loss.item()
    assert last < first * 0.9, f"loss did not fall: {first:.3f} -> {last:.3f}"


def test_generate_runs_and_respects_length():
    cfg = model_config("6L-256d", attention="mla_rope", latent_ratio=0.5,
                       vocab_size=64, block_size=48, dropout=0.0)
    model = build_model(cfg).eval()
    out = model.generate(torch.randint(0, 64, (2, 5)), max_new_tokens=10,
                         top_k=10)
    assert out.shape == (2, 15)


# --------------------------------------------------------------------------
# Audit of the paper's internal consistency
# --------------------------------------------------------------------------


def test_issue5_table2_is_actually_the_6L512d_row():
    """Table 2 is labelled 9L-512d but matches Table 4's 6L-512d row exactly."""
    t2 = {
        "mha": PAPER_TABLE2[("mha", 1.0)],
        "mla@1.0": PAPER_TABLE2[("mla", 1.0)],
        "mla@0.5": PAPER_TABLE2[("mla", 0.5)],
        "mla_rope@1.0": PAPER_TABLE2[("mla_rope", 1.0)],
        "mla_rope@0.5": PAPER_TABLE2[("mla_rope", 0.5)],
    }
    assert t2 == PAPER_TABLE4["6L-512d"], "expected Table 2 to equal the 6L-512d row"
    assert t2 != PAPER_TABLE4["9L-512d"], "Table 2 should NOT match its own label"


def test_issue6_paper_kv_column_breaks_O_r_scaling_at_exactly_one_point():
    """Memory is O(r). Every halving of r must be 2.00x. One step is not."""
    ratios = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125]
    steps = []
    for prev, cur in zip(ratios, ratios[1:]):
        steps.append(PAPER_KV_MEM[("mla", prev)] / PAPER_KV_MEM[("mla", cur)])
    broken = [s for s in steps if abs(s - 2.0) > 0.05]
    assert len(broken) == 1, f"expected exactly one broken step, got {steps}"
    assert steps[0] == pytest.approx(1.81, abs=0.01), "the d -> d/2 step is the broken one"


def test_issue6_true_reduction_at_half_rank_is_50_not_45_percent():
    paper_reduction = 100 * (1 - PAPER_KV_MEM[("mla", 0.5)] / PAPER_KV_MEM[("mha", 1.0)])
    assert paper_reduction == pytest.approx(44.8, abs=0.5), "the paper's headline 45%"

    full = build_model(model_config("9L-512d", attention="mla", latent_ratio=1.0))
    half = build_model(model_config("9L-512d", attention="mla", latent_ratio=0.5))
    true_reduction = 100 * (
        1 - half.kv_elements_per_token() / full.kv_elements_per_token()
    )
    assert true_reduction == pytest.approx(50.0, abs=1e-6)


def test_issue7_paper_absolute_memory_not_derivable_from_stated_config():
    """No layer count x dtype reproduces 0.0288 MB/tok for a 512-wide model."""
    target = PAPER_KV_MEM[("mha", 1.0)]
    candidates = [
        L * 2 * 512 * size / 1e6 for L in (6, 9, 12) for size in (2, 4)
    ]
    assert all(abs(c - target) > 1e-3 for c in candidates), (
        f"expected none of {candidates} to match {target}"
    )


def test_issue8_and_9_savings_factors_disagree_with_the_tables():
    mha = PAPER_KV_MEM[("mha", 1.0)]
    at_eighth = mha / PAPER_KV_MEM[("mla", 0.125)]
    at_quarter = mha / PAPER_KV_MEM[("mla", 0.25)]
    assert at_eighth == pytest.approx(7.2, abs=0.05), "Sec 5.4 claims 7.5x"
    assert at_quarter == pytest.approx(3.6, abs=0.05), "Takeaways claim up to 6x"


def test_issue10_table5_best_score_is_full_rank_not_half_rank():
    """Sec 5.2 credits r=d/2 with the top scores; Table 5 shows r=d is higher."""
    full = PAPER_TABLE5["mla_rope@1.0"]
    half = PAPER_TABLE5["mla_rope@0.5"]
    assert full[3] > half[3], "r=d should score higher, contradicting Sec 5.2"
    assert all(f >= h for f, h in zip(full[:3], half[:3]))


def test_issue11_rope_gap_narrows_although_appendix_b_says_it_widens():
    gap_full = PAPER_TABLE5["mla_rope@1.0"][3] - PAPER_TABLE5["mla@1.0"][3]
    gap_16 = PAPER_TABLE5["mla_rope@0.0625"][3] - PAPER_TABLE5["mla@0.0625"][3]
    assert gap_full == pytest.approx(1.9, abs=0.01)
    assert gap_16 == pytest.approx(1.4, abs=0.01)
    assert gap_16 < gap_full, "App B claims the gap 'expands'; it contracts"


def test_issue12_headline_speedup_is_against_mla_not_mha():
    """97 tok/s is 1.4x full-rank MLA+RoPE but 0.57x the MHA baseline."""
    mha, rope_full, rope_half = 170, 69, 97
    assert rope_half / rope_full == pytest.approx(1.41, abs=0.02)
    assert rope_half / mha == pytest.approx(0.57, abs=0.02)


def test_table2_internal_deltas_are_self_consistent():
    """Table 2's percentage column does check out against its own losses."""
    baseline = PAPER_TABLE2[("mha", 1.0)]
    expected = {
        ("mla", 1.0): 3.2, ("mla", 0.5): 2.2, ("mla", 0.25): 7.0,
        ("mla", 0.125): 13.8, ("mla", 0.0625): 26.5, ("mla", 0.03125): 46.4,
        ("mla_rope", 1.0): -2.1, ("mla_rope", 0.5): 0.3,
        ("mla_rope", 0.25): 4.4, ("mla_rope", 0.125): 10.3,
        ("mla_rope", 0.0625): 18.6, ("mla_rope", 0.03125): 34.7,
    }
    for key, claimed in expected.items():
        actual = 100 * (PAPER_TABLE2[key] - baseline) / baseline
        assert actual == pytest.approx(claimed, abs=0.15), f"{key}: {actual:.2f}"


def test_table5_derived_claims_that_do_hold():
    """Not everything is wrong; these Appendix B figures check out."""
    t5 = PAPER_TABLE5
    assert t5["mla_rope@1.0"][2] - t5["mha@1.0"][2] == pytest.approx(1.7, abs=0.01)
    assert t5["mla_rope@1.0"][1] - t5["mha@1.0"][1] == pytest.approx(1.3, abs=0.01)
    assert t5["mla@1.0"][2] - t5["mha@1.0"][2] == pytest.approx(-0.8, abs=0.01)
    retained = t5["mla_rope@0.5"][3] / t5["mla_rope@1.0"][3]
    assert retained == pytest.approx(0.973, abs=0.005)       # "retains 97%"
    at_quarter = t5["mla_rope@0.25"][3] / t5["mla_rope@1.0"][3]
    assert at_quarter == pytest.approx(0.919, abs=0.005)     # "92% of baseline"


# --------------------------------------------------------------------------
# offline scorer calibration against Appendix C
# --------------------------------------------------------------------------

APPENDIX_C = {
    "mha": (
        "a cozy burrow under the old oak tree. Every morning, the rabbit would "
        "hop out to find fresh clover. One day, she discovered a mysterious blue "
        "stone that sparkled in the sunlight. She picked it up carefully and "
        "noticed it felt warm in her paws. When she showed it to her friend the "
        "wise owl, he told her it was a wishing stone from ancient times."
    ),
    "mla_no_rope": (
        "a burrow tree place. The rabbit liked carrots and carrots. A fox and fox "
        "came to visit. The rabbit rabbit was happy and then the stone. They "
        "played all day with together. The end came and rabbit went to home back. "
        "The fox was there and there again."
    ),
    "mla_rope": (
        "a beautiful meadow filled with wildflowers. The rabbit loved to explore "
        "beyond the hills where ancient stones marked forgotten paths. One misty "
        "morning, she found a glowing pebble that hummed with magic. As she "
        "touched it, the world shimmered and she could suddenly understand the "
        "language of all the forest creatures."
    ),
    "rope_d16": (
        "a place place. Rabbit was rabbit and happy happy. The the the day came "
        "and rabbit rabbit. Friend friend came too. They they went to to place. "
        "Happy end end."
    ),
    "rope_d32": (
        "the the the the. And and and. Rabbit the the. The the the the. And and. "
        "The the. End the the."
    ),
}


def test_scorer_separates_the_collapse_regimes():
    s = {k: heuristic_scores(v).overall for k, v in APPENDIX_C.items()}
    assert s["mla_rope"] > s["mla_no_rope"], "RoPE sample should outscore no-RoPE"
    assert s["mla_no_rope"] > s["rope_d16"], "mild failure beats repetition loops"
    assert s["rope_d16"] > s["rope_d32"], "repetition beats degenerate tokens"
    assert s["rope_d32"] < 5.0, "degenerate output must score poorly"


def test_scorer_ordering_matches_gpt4_ordering_on_appendix_c():
    """Rank agreement against the paper's own GPT-4 scores for these samples."""
    import itertools

    gpt4 = {"mha": 6.2, "mla_no_rope": 5.5, "mla_rope": 7.4,
            "rope_d16": 4.1, "rope_d32": 2.3}
    pairs = [(heuristic_scores(APPENDIX_C[k]).overall, gpt4[k]) for k in gpt4]
    combos = list(itertools.combinations(pairs, 2))
    concordant = sum(1 for (a1, b1), (a2, b2) in combos if (a1 - a2) * (b1 - b2) > 0)
    assert concordant == len(combos), f"{concordant}/{len(combos)} concordant"


def test_scorer_is_deterministic():
    a = heuristic_scores(APPENDIX_C["mha"])
    b = heuristic_scores(APPENDIX_C["mha"])
    assert a.as_dict() == b.as_dict()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "--tb=short"]))
