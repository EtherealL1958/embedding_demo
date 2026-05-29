#!/usr/bin/env bash
set -euo pipefail

mkdir -p models

hf download BAAI/bge-reranker-base \
  --local-dir models/bge-reranker-base \
  --include ".gitattributes" \
  --include "README.md" \
  --include "config.json" \
  --include "model.safetensors" \
  --include "sentencepiece.bpe.model" \
  --include "special_tokens_map.json" \
  --include "tokenizer.json" \
  --include "tokenizer_config.json"

hf download Qwen/Qwen2.5-1.5B-Instruct \
  --local-dir models/Qwen2.5-1.5B-Instruct \
  --include ".gitattributes" \
  --include "LICENSE" \
  --include "README.md" \
  --include "config.json" \
  --include "generation_config.json" \
  --include "merges.txt" \
  --include "model.safetensors" \
  --include "tokenizer.json" \
  --include "tokenizer_config.json" \
  --include "vocab.json"
