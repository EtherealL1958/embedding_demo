#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export TOKENIZERS_PARALLELISM=false

mkdir -p results

python -m src.exp1_retriever.evaluate \
  --eval_path data/dpr/biencoder-nq-dev.json.gz \
  --output_json results/bert_raw.json \
  --output_csv results/bert_raw.csv \
  --base_model models/bert-base-uncased \
  --max_eval_samples "${MAX_EVAL_SAMPLES:-3000}" \
  --max_query_length 64 \
  --max_passage_length 256 \
  --query_batch_size 64 \
  --passage_batch_size 128 \
  --eval_batch_size "${EVAL_BATCH_SIZE:-128}"
