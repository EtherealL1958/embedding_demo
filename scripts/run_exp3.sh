#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:src"

python -m exp3_generator.run_experiment \
  --train_questions_path data/clapnq/question_train_answerable.tsv \
  --dev_questions_path data/clapnq/question_dev_answerable.tsv \
  --passages_path data/clapnq/passages.tsv \
  --index_path indexes/clapnq_stage2.faiss \
  --exp2_summary_path results/exp2/summary.json \
  --retriever_base_model models/bert-base-uncased \
  --retriever_adapter_path outputs/stage2 \
  --reranker_model models/bge-reranker-base \
  --generator_model models/Qwen2.5-1.5B-Instruct \
  --adapter_output_dir outputs/generator_lora \
  --output_dir results/exp3 \
  --top_n 30 \
  --num_train_epochs 2 \
  --learning_rate 2e-5 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --generation_batch_size 4
