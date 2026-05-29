#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:src"

QUESTIONS_PATH="${QUESTIONS_PATH:-data/clapnq/question_dev_answerable.tsv}"
PASSAGES_PATH="${PASSAGES_PATH:-data/clapnq/passages.tsv}"
INDEX_PATH="${INDEX_PATH:-indexes/clapnq_stage2.faiss}"
OUTPUT_DIR="${OUTPUT_DIR:-results/exp2}"
RETRIEVER_BASE_MODEL="${RETRIEVER_BASE_MODEL:-models/bert-base-uncased}"
RETRIEVER_ADAPTER_PATH="${RETRIEVER_ADAPTER_PATH:-outputs/stage2}"
RERANKER_MODEL="${RERANKER_MODEL:-models/bge-reranker-base}"
RERANKER_MODE="${RERANKER_MODE:-cross_encoder}"
GENERATOR_MODEL="${GENERATOR_MODEL:-models/Qwen2.5-1.5B-Instruct}"
TOP_N="${TOP_N:-30}"
K_VALUES="${K_VALUES:-1,3,5,10}"
BEST_K="${BEST_K:-0}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-0}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
SKIP_GENERATION="${SKIP_GENERATION:-0}"

extra_args=()
if [[ "$SKIP_GENERATION" == "1" ]]; then
  extra_args+=(--skip_generation)
fi
if [[ "$BEST_K" != "0" ]]; then
  extra_args+=(--best_k "$BEST_K")
fi

python -m exp2_rag.run_experiments \
  --questions_path "$QUESTIONS_PATH" \
  --passages_path "$PASSAGES_PATH" \
  --index_path "$INDEX_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --retriever_base_model "$RETRIEVER_BASE_MODEL" \
  --retriever_adapter_path "$RETRIEVER_ADAPTER_PATH" \
  --reranker_model "$RERANKER_MODEL" \
  --reranker_mode "$RERANKER_MODE" \
  --generator_model "$GENERATOR_MODEL" \
  --top_n "$TOP_N" \
  --k_values "$K_VALUES" \
  --max_eval_samples "$MAX_EVAL_SAMPLES" \
  --generation_batch_size "$GENERATION_BATCH_SIZE" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  "${extra_args[@]}"
