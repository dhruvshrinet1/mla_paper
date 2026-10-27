# Paper Analysis: "Latent Multi-Head Attention for Small Language Models"

arXiv:2506.09342v2 — Mehta, Dandekar, Dandekar, Panat (SciSoc LLM Workshop '25)

This document is the implementation spec. It walks the paper section by section,
extracts every statement that implies code, and records (a) what must be built,
(b) where it lives in this repo, and (c) where the paper is internally
inconsistent or under-specified so that a reimplementation has to make a choice.

Legend:
- **[IMPL]** something that must be implemented
- **[GAP]** under-specified; implementation makes an explicit, documented choice
- **[INCONSISTENT]** the paper contradicts itself or its own numbers

---

## 1. Abstract and Introduction

### Claims made

| # | Claim | Where tested |
|---|---|---|
| C1 | MLA+RoPE at `r = d/2` gives 45% KV-cache reduction for +0.3% val loss | `bench_memory.py`, `train.py` |
| C2 | Without RoPE, MLA is 3–5% worse than MHA; with RoPE it beats MHA by 2% | `train.py` sweep |
| C3 | MLA `r = d/2` is 1.4× faster than full-rank MLA on A100 | `bench_throughput.py` |
| C4 | GPT-4 eval corroborates perplexity; MLA+RoPE scores 7.4/10 | `eval_judge.py` |

### [INCONSISTENT] 1.1 — model size

The abstract says "**30M-parameter** GPT models". Table 1 lists eight configs
spanning 17.5M to 202.7M, none of which is 30M. The closest are 9L-256d (19.9M)
and 12L-256d (22.2M). The headline GPT-4 results (Tables 3 and 5) are from
**12L-1024d, i.e. 202.7M**, which is not a small model by the paper's own framing.

*Choice:* implement all eight configs; make the config the user's explicit choice
rather than hard-coding a "30M" default.

### [INCONSISTENT] 1.2 — corpus description

Abstract and §4.1 say "100,000 synthetic stories" / TinyStories-style. The
caption of Table 1 says the models "were trained on a corpus of **10,000 books**".
These are different corpora.

*Choice:* follow §4.1 (100K TinyStories-style stories). `data.py` makes the
corpus size a flag.

---

## 2. Background — the parts that become code

### 2.1 Standard MHA (Eq. 1)

```
Q_h = X W_Q_h,  K_h = X W_K_h,  V_h = X W_V_h
W ∈ R^{d × d_k},  d_k = d / H
```

**[IMPL]** Baseline path in `mla/attention.py::MultiHeadAttention`. Nothing unusual;
fused QKV projection, causal mask, scaled dot product.

### 2.2 MLA factorisation (Eq. 2–3)

```
K_h = X W_K_down_h W_K_up_h = K_latent_h W_K_up_h
V_h = X W_V_down_h W_V_up_h = V_latent_h W_V_up_h

W_down_h ∈ R^{d × r},  W_up_h ∈ R^{r × d_k},  r < d_k
```

**[IMPL]** `mla/attention.py::LatentMultiHeadAttention`.

Three points the paper states explicitly and which constrain the implementation:

1. **The down-projection is per-head** (`W_K_down_h` carries subscript *h*). The
   paper confirms this by stating cache memory goes from `O(n H d_k)` to
   `O(n H r)` — the `H` survives. This is **not** DeepSeek-V2 MLA, where a
   *single* shared down-projection gives `O(n r)`. Per-head latents are a much
   weaker compression scheme.
2. **`r` is a fraction of `d_k`, not of `d`.** §2.1 says `r < d_k` and that the
   sweep is `r ∈ {d_k, d_k/2, d_k/4}`. Figure 1(b)'s x-axis is labelled `r/d_k`.
   But every results table writes `r = d`, `r = d/2` … This is a notation
   collision: for 9L-512d with H=8, `d_k = 64`, so a literal `r = d/2 = 256`
   would be a 4× *expansion*, not compression.

   **[INCONSISTENT] 2.1** — tables say `d`, text and figure say `d_k`. *Choice:*
   `r` is always specified as a ratio of `d_k`. `configs.py` exposes
   `latent_ratio`, and `r = max(1, round(d_k * latent_ratio))`.
3. **`r = d_k` is a no-op compression.** The paper says so. It is therefore a
   useful correctness test: MLA at `r = d_k` has the same cache size as MHA and
   the same *capacity* (a rank-`d_k` factorisation of a `d × d_k` matrix is
   unconstrained), differing only by reparameterisation.

   **[IMPL]** `tests/test_attention.py::test_full_rank_mla_matches_mha_capacity`.

#### [GAP] 2.2 — the absorption trick

The paper never mentions it, but it is what makes MLA fast and it changes the
inference code materially. Because

```
scores = Q_h K_h^T = Q_h (K_latent_h W_K_up_h)^T = (Q_h W_K_up_h^T) K_latent_h^T
out    = A V_h     = A (V_latent_h W_V_up_h)     = (A V_latent_h) W_V_up_h
```

the up-projections can be *absorbed* into the query side and the output side.
Keys and values are then **never materialised** at decode time; attention runs
entirely in the `r`-dimensional latent space. This is the mechanism behind the
paper's observation that MLA throughput improves with batch size "thanks to the
smaller `r × r` attention matrix that fits in on-chip SRAM".

**[IMPL]** `attention.py` implements `absorb=True` for the no-RoPE path.

#### [GAP] 2.3 — RoPE breaks absorption, and the paper does not say how it handles that

RoPE is position-dependent and is applied *after* the up-projection. So
`RoPE(K_latent W_up, m)` cannot be rewritten as something acting on `K_latent`
alone. Absorption is unavailable. That leaves three options:

| Option | KV cache size | Decode cost | Matches paper? |
|---|---|---|---|
| (a) cache latents, re-expand + re-RoPE every step | `H·r` ✅ | `O(n·r·d_k)` per step ❌ | yes: memory table shows MLA and MLA+RoPE **identical**, and §5.4 says RoPE costs throughput |
| (b) cache post-RoPE keys | `H·d_k` ❌ | cheap | no: would show no memory saving |
| (c) DeepSeek-V2 decoupled RoPE: small separate position head cached alongside | `H·r + d_r` | cheap | no: paper never mentions it, and cache would differ from the no-RoPE row |

The paper's Table 2 gives MLA and MLA+RoPE **the same** `KV Mem` at every `r`,
and §5.4 reports MLA+RoPE is consistently slower. Both facts point to **(a)**.

*Choice:* (a) is the default and is what reproduces the paper. (b) is available
as `kv_cache_mode="expanded"` for A/B timing. (c) is implemented as a clearly
labelled **extension beyond the paper** (`rope_mode="decoupled"`) because it is
the technique that actually makes MLA+RoPE fast, and a reader of this repo
should be able to see the difference.

### 2.4 RoPE (Eq. 4)

```
RoPE(x_m, m) = x_m e^{i m θ}
```

**[INCONSISTENT] 2.4** — the paper says "`θ` are **learned** frequencies". In
RoFormer (their own citation [23]) and in every standard implementation, `θ` is
*fixed*: `θ_i = base^{-2i/d_k}` with `base = 10000`. Learned frequencies are a
different method and would need to be stated as a contribution.

*Choice:* fixed frequencies by default (`rope_learned=False`), with a
`rope_learned=True` option implementing per-frequency learnable `θ` so the
paper's literal wording can also be tested.

**[GAP] 2.5** — RoPE base, whether RoPE applies to all head dims or a fraction,
and whether learned position embeddings are *removed* when RoPE is on, are all
unspecified. The paper says RoPE works "without explicit position embeddings",
so *choice:* `use_learned_pos_emb` is automatically disabled when RoPE is on
(overridable). This changes parameter counts slightly and is reported.

---

## 3. Related Work

No implementation requirements. Useful framing only: the paper positions MLA as
orthogonal to GQA/MQA/FlashAttention.

**[IMPL, optional]** Since the comparison is one line of code away and makes the
repo more useful, `attention.py` also provides GQA/MQA via `n_kv_heads`, so the
"orthogonal benefits" claim can actually be measured.

---

## 4. Method — Section 4.1 Training Setup

Every hyperparameter stated, and its status:

| Setting | Value | Status |
|---|---|---|
| Optimiser | AdamW, β = (0.9, 0.95) | specified |
| Learning rate | 3e-4 | specified |
| Schedule | cosine, 10% warmup, 50K steps | specified |
| Batch size | 128 | specified |
| Context length | 512 | specified |
| Gradient clipping | 1.0 | specified |
| Dropout | 0.1 | specified |
| Tokenizer | GPT-2, 50K vocab | specified |
| Output layer | tied softmax | specified (Table 1 caption) |
| Weight decay | — | **[GAP]** default 0.1, decay on 2D params only |
| Cosine floor LR | — | **[GAP]** default 10% of peak (`3e-5`) |
| Precision | — | **[GAP]** default bf16 autocast on GPU, fp32 on CPU |
| Attention head count `H` | — | **[GAP]** default `d/64`, which is what makes `d_k=64` and the `r` ratios meaningful |
| Init scheme | — | **[GAP]** GPT-2 style: N(0, 0.02), residual projections scaled by `1/sqrt(2L)` |
| Seed / number of runs | — | **[GAP]** no seeds or error bars given for loss; single run assumed |

**[IMPL]** All of the above in `mla/config.py` and `scripts/train.py`.

### Table 1 — the eight configurations

**[IMPL]** `mla/config.py::MODEL_CONFIGS`.

Verified parameter counts (vocab 50257, ctx 512, 4× MLP, tied embeddings, no biases):

| Config | L | d | Paper | Recomputed (with pos emb) | Recomputed (no pos emb) |
|---|---|---|---|---|---|
| 6L-256d | 6 | 256 | 17.5M | 17.72M | 17.58M |
| 6L-512d | 6 | 512 | 44.5M | 44.87M | 44.61M |
| 9L-256d | 9 | 256 | 19.9M | 20.07M | 19.94M |
| 9L-512d | 9 | 512 | 53.9M | 54.31M | 54.04M |
| 12L-256d | 12 | 256 | 22.2M | 22.43M | 22.30M |
| 12L-512d | 12 | 512 | 63.3M | 63.74M | 63.48M |
| 12L-768d | 12 | 768 | 123.3M | 123.93M | 123.53M |
| 12L-1024d | 12 | 1024 | 202.7M | 202.98M | 202.46M |

The "no pos emb" column matches the paper to within 0.5% across all eight, so the
architecture is standard GPT-2 with tied embeddings. Good — no ambiguity here.
`tests/test_config.py` asserts this.

---

## 5. Results — what has to be reproducible

### 5.1 Table 2 — quality/efficiency trade-off

Labelled "9L-512d". **[INCONSISTENT] 5.1:** the MHA value in Table 2 is **2.147**,
but Appendix Table 4 gives 9L-512d MHA = **1.983** and 6L-512d MHA = **2.147**.
Every other entry matches too:

| | Table 2 (claims 9L-512d) | Table 4 row 6L-512d | Table 4 row 9L-512d |
|---|---|---|---|
| MHA | 2.147 | **2.147** ✓ | 1.983 |
| MLA r=d | 2.216 | **2.216** ✓ | 2.044 |
| MLA r=d/2 | 2.194 | **2.194** ✓ | 2.026 |
| MLA+RoPE r=d | 2.102 | **2.102** ✓ | 1.942 |
| MLA+RoPE r=d/2 | 2.154 | **2.154** ✓ | 1.979 |

**Table 2 is the 6L-512d model, mislabelled as 9L-512d.** The percentage deltas
are unaffected (they are internally consistent within Table 2), but the main
results table names the wrong model. The reproduction script therefore runs
**both** and reports which one the numbers land on.

Delta arithmetic in Table 2 checks out everywhere (verified all 12 rows), so the
relative story is self-consistent.

### [INCONSISTENT] 5.2 — the "45%" memory figure

Table 2's `KV Mem` column:

| r | MB/tok | ratio to previous |
|---|---|---|
| d | 0.0288 | — |
| d/2 | 0.0159 | 1.81× |
| d/4 | 0.0080 | 1.99× |
| d/8 | 0.0040 | 2.00× |
| d/16 | 0.0020 | 2.00× |
| d/32 | 0.0010 | 2.00× |

Memory is `O(r)`, so **every** step should be exactly 2×. The rows from `d/2`
downward are a clean geometric series. Only the `d → d/2` step is off, at 1.81×.
Halving `r` must halve the cache: the true value at `r = d/2` is **0.0144**, not
0.0159, and the true reduction is **50%**, not 45%.

The headline "45% KV-cache reduction" in the title, abstract and conclusion is
therefore an artefact of one inconsistent table cell. The honest number is 50%,
which is *better* than claimed — but it means the number in the abstract is not
reproducible from the paper's own model of memory.

Separately, **none** of the absolute MB/tok values are derivable from the stated
config. For 6L-512d, `L × 2 × d × bytes` gives 0.0176 MB/tok at fp16 and 0.0352
at fp32; for 9 layers, 0.0264 / 0.0527. No layer count × dtype combination
produces 0.0288.

*Choice:* `bench_memory.py` reports the **analytical** value from the actual
config and a **measured** value from real allocated cache tensors, and prints the
paper's figures alongside for comparison rather than trying to match them.

### [INCONSISTENT] 5.3 — "7.5×" and "6×" cache savings

§5.4 says `r = d/8` gives "a reduction 7.5× relative to MHA". From Table 2:
`0.0288 / 0.0040 = 7.2×`. The Takeaways paragraph says `r = d/4` gives "up to 6×
KV cache savings"; Table 2 gives `0.0288 / 0.0080 = 3.6×`. Neither matches.

### 5.2 Table 3 / 5 — GPT-4 evaluation

**[IMPL]** `scripts/eval_judge.py`. Rubric: grammar, creativity, narrative
consistency, 1–10, 100 completions per model, mean and standard deviation.

Arithmetic checks that **pass**: consistency +1.7 over MHA (7.3−5.6), creativity
+1.3 (7.2−5.9), pure MLA consistency −0.8 (4.8−5.6), §5.3's "+30% consistency"
(7.3/5.6 = 1.30), Appendix B's "retains 97%" (7.2/7.4) and "92% of baseline"
(6.8/7.4).

**[INCONSISTENT] 5.4** — §5.2 states "MLA+RoPE (`r = d/2`) achieves the highest
scores in all dimensions", and Table 3 attributes 7.4 overall to "MLA+RoPE".
Table 5 shows `r = d/2` scores **7.2** and `r = d` scores **7.4**. So the
best-scoring model is full-rank MLA+RoPE, not the half-rank one, and Table 3's
unlabelled "MLA+RoPE" row is the `r = d` variant. The paper's central
efficiency/quality claim is attached to the wrong row.

**[INCONSISTENT] 5.5** — Appendix B, "RoPE's protective effect": the text says
the MLA vs MLA+RoPE gap "**widens** dramatically at higher compression ratios …
both start with a 2 point difference at `r = d`, this gap expands to 1.4 points
at `r = d/16`". From Table 5 the gap is 7.4 − 5.5 = **1.9** at `r = d` and
4.1 − 2.7 = **1.4** at `r = d/16`. The gap **narrows**. The stated numbers
contradict the stated conclusion.

**[GAP] 5.6** — the actual judge prompt, the rubric anchors, the sampling
settings used for the 100 completions, and the judge model snapshot are all
absent. *Choice:* write an explicit rubric prompt, pin temperature/top-p for
generation, record the judge model id in the output, and provide a deterministic
offline heuristic scorer (repetition rate, distinct-n, type-token ratio,
sentence-structure proxies) so the pipeline is runnable with no API key.

### 5.3 Qualitative analysis

**[IMPL]** `scripts/generate.py` — fixed prompt ("Once upon a time, there was a
little rabbit who lived in…"), same prompt across variants, side-by-side output,
deterministic seeding. The documented failure modes (repetition at `r = d/16`,
degenerate tokens at `r = d/32`) become the offline scorer's metrics.

### 5.4 Figure 1 — inference benchmarks

Hardware: single NVIDIA A100 80GB, batch sizes 1 → 32, decode throughput in
tokens/sec, plus KV-cache MB/tok vs `r/d_k`.

**[IMPL]** `scripts/bench_throughput.py` with warmup, CUDA event timing, repeat
counts, and graceful CPU fallback.

Reported numbers and their consistency:

| Statement | Check |
|---|---|
| MHA 210 → 170 tok/s (batch 1 → 32) | stated |
| MLA `r=d/4` 145 → 186 tok/s; beats MHA by 9% at batch 32 | 186/170 = 1.094 ✓ |
| MLA `r=d` scales to 110 tok/s | stated |
| MLA+RoPE `r=d/2` = 97 tok/s at batch 32, 1.4× faster than MLA+RoPE `r=d` | implies `r=d` ≈ 69 tok/s ✓ internally |

**[INCONSISTENT] 5.7** — "`r = d/2` delivers a strong 162 tok/s at batch 32
(**4% behind** `r = d/4`)". 162 vs 186 is **12.9%** behind, not 4%.

**[INCONSISTENT] 5.8** — "MLA + RoPE trails its counterpart without RoPE by
**≈15 tok/s** throughout the board". At batch 32, MLA `r=d/2` is 162 and
MLA+RoPE `r=d/2` is 97: a gap of **65 tok/s**, over 4× the stated figure.

**[INCONSISTENT] 5.9 — the most consequential one.** The abstract and conclusion
read as "45% memory reduction **and 1.4× inference speedup**", which invites the
reading that MLA+RoPE is 1.4× faster than the MHA baseline. The body makes clear
the 1.4× is MLA+RoPE `r=d/2` **versus MLA+RoPE `r=d`** — i.e. versus another MLA
variant, not versus MHA. Against the MHA baseline, MLA+RoPE `r=d/2` runs at
97 tok/s vs 170 tok/s: it is **0.57×, roughly 43% slower**.

So the paper's own numbers say the recommended configuration (MLA+RoPE, `r=d/2`)
is a memory win and a **throughput loss** against MHA. The "Pareto improvement"
framing in the abstract holds only on the memory/quality plane, not on
memory/quality/latency. `bench_throughput.py` reports speedup against *both*
baselines explicitly so this cannot be misread.

**[INCONSISTENT] 5.10** — §5.4 opens by saying it uses "larger batch sizes of 32
sequences", then the takeaway recommends `r = d/4` while the abstract, title and
conclusion all recommend `r = d/2`. The paper's two recommendations differ.

---

## 6. Limitations — implementation consequences

The paper lists: domain specificity (TinyStories only), 50K-step training,
only three compression ratios evaluated (contradicted by Tables 2 and 5, which
report six), A100-only benchmarks, GPT-4-only evaluation.

**[INCONSISTENT] 6.1** — "We evaluated only three compression levels
(`r ∈ {d, d/2, d/4}`)" while Table 2 and Table 5 both report six levels, down to
`r = d/32`.

**[IMPL]** Make all of these dimensions flags rather than constants: arbitrary
`latent_ratio`, arbitrary step budget, device-agnostic benchmarks, pluggable
judge.

---

## 7. Conclusion

Restates C1–C4. Covered above. The reproduction target is Tables 2–5 and Figure 1.

---

## Summary of what gets built

| Component | File | Covers |
|---|---|---|
| RoPE (fixed + learned, decoupled variant) | `mla/rope.py` | §2.2, Eq. 4 |
| MHA / MLA / MLA+RoPE, absorption, latent KV cache, GQA/MQA | `mla/attention.py` | §2.1, Eq. 1–3 |
| GPT backbone, tied softmax | `mla/model.py` | Table 1 |
| 8 configs + variant matrix | `mla/config.py` | Table 1 |
| TinyStories data pipeline | `mla/data.py` | §4.1 |
| Training loop | `scripts/train.py` | §4.1 |
| Sweep driver for Tables 2 and 4 | `scripts/sweep.py` | §5.1, App. A |
| KV-cache memory, analytical + measured | `scripts/bench_memory.py` | Table 2, Fig 1b |
| Throughput vs batch size | `scripts/bench_throughput.py` | §5.4, Fig 1a |
| Sampling | `scripts/generate.py` | §5.3, App. C |
| LLM judge + offline heuristic scorer | `scripts/eval_judge.py` | §5.2, Tables 3 & 5, App. B |
| Figure 1 reproduction | `scripts/make_figures.py` | Fig 1 |
| Correctness tests | `tests/` | all of the above |

### The twelve issues found, in one list

1. Abstract says 30M params; no config is 30M; headline results use 202.7M.
2. Corpus is "100K stories" in text, "10,000 books" in Table 1 caption.
3. `r` is a fraction of `d_k` in the text and figure, of `d` in every table.
4. RoPE `θ` described as learned; the cited method uses fixed frequencies.
5. Table 2 is labelled 9L-512d but its values are the 6L-512d row of Table 4.
6. KV-cache at `r=d/2` breaks the `O(r)` scaling the rest of the column follows; the true reduction is 50%, not the headline 45%.
7. Absolute MB/tok values are not derivable from the stated config under any dtype.
8. "7.5× reduction at `r=d/8`" — the table gives 7.2×.
9. "up to 6× savings at `r=d/4`" — the table gives 3.6×.
10. §5.2 credits `r=d/2` with the top scores; Table 5 shows `r=d` is higher.
11. Appendix B says the RoPE gap widens with compression; its own numbers show it narrowing (1.9 → 1.4).
12. "1.4× speedup" is versus full-rank MLA, not versus MHA; versus MHA the recommended config is ~43% *slower*. Also §5.4 recommends `r=d/4` while the abstract recommends `r=d/2`, and §6 says three ratios were tested when six are reported.
