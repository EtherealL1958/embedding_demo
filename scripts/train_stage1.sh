#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export TOKENIZERS_PARALLELISM=false

mkdir -p outputs/stage1 results

python -m src.exp1_retriever.train \
  --train_path data/dpr/biencoder-nq-train.json.gz \
  --output_dir outputs/stage1 \
  --base_model models/bert-base-uncased \
  --stage 1 \
  --max_train_samples "${MAX_TRAIN_SAMPLES:-50000}" \
  --max_query_length 64 \
  --max_passage_length 256 \
  --epochs "${EPOCHS:-1}" \
  --batch_size "${BATCH_SIZE:-32}" \
  --grad_accum_steps "${GRAD_ACCUM_STEPS:-1}" \
  --lr "${LR:-2e-5}" \
  --temperature 0.05 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --precision "${PRECISION:-fp16}" \
  --num_workers 2 \
  --log_steps 20
