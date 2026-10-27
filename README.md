# Latent Multi-Head Attention for Small Language Models

A from-scratch PyTorch implementation of **arXiv:2506.09342v2**, Mehta, Dandekar,
Dandekar and Panat (SciSoc LLM Workshop '25), covering the three attention
architectures it compares, the training setup, the memory and throughput
benchmarks, and the quality evaluation.

Two things ship here. The first is working code. The second is
[`PAPER_ANALYSIS.md`](PAPER_ANALYSIS.md), a section-by-section implementation
spec that records where the paper is under-specified and where its own numbers
contradict each other. **Twelve internal inconsistencies were found, including
one that changes how the headline result should be read.** Each is asserted as
an executable test in `tests/test_config_and_claims.py`.

---

## What the paper claims, and what the numbers actually say

| Claim | Status after reimplementation |
|---|---|
| 45% KV-cache reduction at `r = d/2` | The true figure is **50%**. Memory is `O(r)`, so halving `r` halves the cache. The paper's 45% comes from a single table cell that breaks the `O(r)` scaling every other cell in its column follows. |
| 1.4x inference speedup | Correct, but **against full-rank MLA, not against MHA**. By the paper's own Figure 1a numbers the recommended config runs at 97 tok/s where MHA runs at 170, so it is roughly **43% slower** than the baseline. |
| +0.3% validation loss at `r = d/2` | Internally consistent, but Table 2 is **labelled 9L-512d while its values are the 6L-512d row** of Table 4. |
| MLA+RoPE `r=d/2` scores highest on quality | Table 5 shows **`r = d` scores higher** (7.4 vs 7.2). The central efficiency claim is attached to the wrong row. |
| RoPE's advantage widens under compression | The paper's own figures show it **narrowing**, from 1.9 points to 1.4. |

Run `python -m pytest tests/test_config_and_claims.py -v` to see all of these
checked.

What does hold up: the relative quality ordering (MLA+RoPE > MHA > MLA), the
internal arithmetic of Table 2's percentage column, every derived figure in
Appendix B about score retention, and the architecture itself, whose parameter
counts reproduce Table 1 across all eight configurations to within 0.5%.

---

## Install

```bash
pip install torch numpy tiktoken matplotlib
pip install datasets      # optional, for the real TinyStories corpus
pip install openai        # optional, for the LLM judge
pip install pytest        # optional, for the test suite
```

## Quick start

```bash
# 1. corpus. --source hf pulls real TinyStories and falls back to a
#    deterministic synthetic corpus if the network is unavailable.
python scripts/prepare_data.py --source hf --n-stories 100000

# 2. train one variant (Section 4.1 hyperparameters are the defaults)
python scripts/train.py --size 9L-512d --attention mla_rope --latent-ratio 0.5

# 3. benchmarks. These need no trained checkpoint.
python scripts/bench_memory.py --size 9L-512d
python scripts/bench_throughput.py --size 9L-512d

# 4. the full Table 2 sweep: MHA plus both MLA variants at six ratios
python scripts/sweep.py --table 2 --size 6L-512d

# 5. quality evaluation
python scripts/generate.py --runs runs/*/best.pt --n-samples 100
python scripts/eval_judge.py --samples results/samples.json
OPENAI_API_KEY=... python scripts/eval_judge.py --scorer judge --judge-model gpt-4o

# 6. Figure 1
python scripts/make_figures.py
```

A cheap end-to-end check on a laptop CPU, a few minutes:

```bash
python scripts/prepare_data.py --source synthetic --n-stories 20000 \
    --vocab-size 0 --out-dir data/smoke
python scripts/sweep.py --table 2 --size 6L-256d --ratios 1.0 0.5 0.25 \
    --max-steps 300 --batch-size 16 --micro-batch-size 16 --block-size 128 \
    --data-dir data/smoke --out-dir runs/smoke
```

---

## Layout

```
mla/
  rope.py         Rotary embeddings (Eq. 4), fixed and learned frequencies
  attention.py    MHA, MLA, MLA+RoPE, latent KV cache, absorption, GQA/MQA
  model.py        GPT backbone, tied softmax, cached generation
  config.py       Table 1's eight configs and the variant matrix
  data.py         TinyStories pipeline with HF, local and synthetic sources
  tokenizer.py    GPT-2 BPE with an offline word-level fallback
  scoring.py      LLM judge and the deterministic offline scorer
scripts/
  prepare_data.py     build the corpus
  train.py            train one variant
  sweep.py            train the matrix, assemble Tables 2 and 4
  bench_memory.py     KV cache MB/token, analytical and measured
  bench_throughput.py decode throughput vs batch size
  generate.py         sample completions
  eval_judge.py       score completions, assemble Tables 3 and 5
  make_figures.py     Figure 1a and 1b
tests/
  test_attention.py          37 correctness tests
  test_config_and_claims.py  33 tests, including the paper audit
PAPER_ANALYSIS.md   the implementation spec
```

---

## Three implementation decisions worth knowing about

### `r` is a fraction of the head dimension, not the model dimension

Section 2.1 says `r < d_k` and Figure 1b's axis is `r/d_k`, but every results
table writes `r = d/2`. For 9L-512d with eight heads, `d_k = 64`, so a literal
`r = d/2 = 256` would be a 4x *expansion*. `--latent-ratio 0.5` here means
`r = d_k/2 = 32`.

### The down-projection is per head

The paper's notation carries the head subscript and it states memory drops from
`O(n H d_k)` to `O(n H r)` with the `H` surviving. This is **not** DeepSeek-V2
MLA, where one shared down-projection gives `O(n r)`. Per-head latents are a
substantially weaker compression scheme, and that distinction matters when
comparing these results to the DeepSeek line of work.

### RoPE and the absorption trick are in tension, and the paper does not say so

Without RoPE, the up-projections can be folded into the query and output sides:

```
Q (K_lat W_up)^T = (Q W_up^T) K_lat^T
A (V_lat W_up)   = (A V_lat) W_up
```

so keys and values are never materialised and attention runs entirely in `r`
dimensions. This is what makes MLA fast and explains Section 5.4's remark about
a smaller attention matrix fitting in SRAM. RoPE breaks it, because the rotation
depends on absolute position and sits after the up-projection.

The paper gives MLA and MLA+RoPE identical cache sizes and reports MLA+RoPE as
consistently slower, which identifies its choice: cache the latents and
re-expand and re-rotate the whole prefix every decode step. That is the default
here (`--kv-cache-mode latent`). Two alternatives are available:

- `--kv-cache-mode expanded` caches post-RoPE keys. Fast, no memory saving.
- `--rope-mode decoupled` implements DeepSeek-V2's decoupled positional head,
  which keeps both the memory saving and absorption. This is an **extension
  beyond the paper**, labelled as such everywhere it appears.

---

## Reproducing the paper properly

The paper trains 50,000 steps at batch 128 and context 512 on 100K stories, and
benchmarks on an A100 80GB. The defaults here match that. Thirteen variants at
that budget is a multi-GPU-day job.

```bash
python scripts/prepare_data.py --source hf --n-stories 100000
python scripts/sweep.py --table 2 --size 6L-512d --out results/table2.json
python scripts/sweep.py --table 4 --out results/table4.json
python scripts/bench_memory.py --size 9L-512d
python scripts/bench_throughput.py --size 9L-512d --batch-sizes 1 2 4 8 16 32
python scripts/generate.py --runs runs/*/best.pt --n-samples 100
OPENAI_API_KEY=... python scripts/eval_judge.py --scorer judge
python scripts/make_figures.py
```

Note that the paper gives no seeds, no error bars on validation loss, and no
indication of how many runs each number represents, so differences under about
1% should not be read as meaningful without repeating runs across seeds.

## Offline fallbacks

If TinyStories or the GPT-2 BPE table cannot be downloaded, the pipeline falls
back to a synthetic corpus and a word-level tokenizer, announcing both. Those
runs are internally comparable across attention variants, which is what the
paper's claims are about, but their absolute perplexities are not comparable to
the paper's, since perplexity depends on the tokenizer.

The offline quality scorer is likewise a proxy, not a judge model. It is
calibrated against the failure modes the paper describes in Section 5.3 and
Appendix C, and on the paper's own example outputs it reproduces the GPT-4
ordering on all 15 pairwise comparisons. Its absolute values are compressed
toward the top of the range and should not be compared to the paper's numbers.


