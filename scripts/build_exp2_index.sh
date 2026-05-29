#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:src"

PASSAGES_PATH="${PASSAGES_PATH:-data/clapnq/passages.tsv}"
BASE_MODEL="${BASE_MODEL:-models/bert-base-uncased}"
ADAPTER_PATH="${ADAPTER_PATH:-outputs/stage2}"
INDEX_PATH="${INDEX_PATH:-indexes/clapnq_stage2.faiss}"
META_PATH="${META_PATH:-indexes/clapnq_passages.jsonl}"
PASSAGE_BATCH_SIZE="${PASSAGE_BATCH_SIZE:-256}"

python -m exp2_rag.build_index \
  --passages_path "$PASSAGES_PATH" \
  --base_model "$BASE_MODEL" \
  --adapter_path "$ADAPTER_PATH" \
  --index_path "$INDEX_PATH" \
  --meta_path "$META_PATH" \
  --batch_size "$PASSAGE_BATCH_SIZE"

