#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export TOKENIZERS_PARALLELISM=false

mkdir -p results

python -m src.exp1_retriever.evaluate_bge \
  --eval_path data/dpr/biencoder-nq-dev.json.gz \
  --output_json results/bge_base_en_v15.json \
  --output_csv results/bge_base_en_v15.csv \
  --model_name models/bge-base-en-v1.5 \
  --max_eval_samples "${MAX_EVAL_SAMPLES:-3000}" \
  --max_query_length 64 \
  --max_passage_length 256 \
  --batch_size 128 \
  --eval_batch_size "${EVAL_BATCH_SIZE:-128}"
