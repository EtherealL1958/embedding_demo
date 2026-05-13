#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

LOG_FILE="${LOG_FILE:-run_all.log}"

# 终端显示的同时，也写入 run_all.log
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========== RAG Exp1 started at $(date) =========="
echo "working directory: $(pwd)"
echo "log file: $LOG_FILE"

echo "========== 1. eval raw BERT =========="
bash scripts/eval_bert_raw.sh

echo "========== 2. train stage1 =========="
bash scripts/train_stage1.sh

echo "========== 3. eval stage1 =========="
bash scripts/eval_lora_stage1.sh

echo "========== 4. train stage2 =========="
bash scripts/train_stage2.sh

echo "========== 5. eval stage2 =========="
bash scripts/eval_lora_stage2.sh

echo "========== 6. eval BGE =========="
bash scripts/eval_bge.sh

echo "========== RAG Exp1 finished at $(date) =========="
echo "全部完成，结果在 results/ 目录。"
