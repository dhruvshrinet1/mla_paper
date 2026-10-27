#!/usr/bin/env bash
# Full reproduction of the paper's experiments.
#
#   bash scripts/reproduce.sh full    paper settings: 50K steps, 100K stories
#   bash scripts/reproduce.sh quick   a few minutes on CPU, for sanity only
#
# "full" is a multi-GPU-day job. "quick" verifies the pipeline end to end and
# produces every output artifact at a toy scale.
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-quick}"

if [[ "$MODE" == "full" ]]; then
  STORIES=100000; STEPS=50000; BATCH=128; MICRO=16; BLOCK=512
  SIZE=6L-512d; BENCH_SIZE=9L-512d; DATA=data/tinystories; RUNS=runs/full
  SOURCE=hf; VOCAB=50257; SAMPLES=100
  BATCHES="1 2 4 8 16 32"
else
  STORIES=20000;  STEPS=300;   BATCH=16;  MICRO=16; BLOCK=128
  SIZE=6L-256d; BENCH_SIZE=6L-256d; DATA=data/smoke; RUNS=runs/quick
  SOURCE=synthetic; VOCAB=0; SAMPLES=12
  BATCHES="1 4 8"
fi

echo "=== [1/6] corpus (Section 4.1) ==="
python scripts/prepare_data.py --source "$SOURCE" --n-stories "$STORIES" \
    --vocab-size "$VOCAB" --out-dir "$DATA"

echo "=== [2/6] Table 2 sweep (Section 5.1) ==="
python scripts/sweep.py --table 2 --size "$SIZE" \
    --max-steps "$STEPS" --batch-size "$BATCH" --micro-batch-size "$MICRO" \
    --block-size "$BLOCK" --data-dir "$DATA" --out-dir "$RUNS" \
    --out results/table2.json

echo "=== [3/6] KV-cache memory (Table 2, Figure 1b) ==="
python scripts/bench_memory.py --size "$BENCH_SIZE" --out results/kv_memory.json

echo "=== [4/6] throughput (Section 5.4, Figure 1a) ==="
python scripts/bench_throughput.py --size "$BENCH_SIZE" \
    --batch-sizes $BATCHES --out results/throughput.json

echo "=== [5/6] generation and quality (Section 5.2, Tables 3 and 5) ==="
python scripts/generate.py --runs "$RUNS"/*/best.pt --n-samples "$SAMPLES" \
    --data-dir "$DATA" --out results/samples.json
python scripts/eval_judge.py --samples results/samples.json \
    --out results/quality.json

echo "=== [6/6] figures ==="
python scripts/make_figures.py

echo
echo "Done. Artifacts in results/:"
ls -1 results/
