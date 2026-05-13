#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export TOKENIZERS_PARALLELISM=false

mkdir -p results

if [ ! -f outputs/stage1/adapter_config.json ]; then
  echo "没有找到 outputs/stage1/adapter_config.json，请先运行 bash scripts/train_stage1.sh"
  exit 1
fi

python -m src.exp1_retriever.evaluate \
  --eval_path data/dpr/biencoder-nq-dev.json.gz \
  --output_json results/lora_stage1.json \
  --output_csv results/lora_stage1.csv \
  --base_model models/bert-base-uncased \
  --adapter_path outputs/stage1 \
  --max_eval_samples "${MAX_EVAL_SAMPLES:-3000}" \
  --max_query_length 64 \
  --max_passage_length 256 \
  --query_batch_size 64 \
  --passage_batch_size 128 \
  --eval_batch_size "${EVAL_BATCH_SIZE:-128}"
