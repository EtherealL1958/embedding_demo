#!/usr/bin/env bash
set -euo pipefail

mkdir -p data/clapnq

wget -nc -O data/clapnq/question_dev_answerable.tsv \
  https://raw.githubusercontent.com/primeqa/clapnq/main/retrieval/dev/question_dev_answerable.tsv

wget -nc -O data/clapnq/question_train_answerable.tsv \
  https://raw.githubusercontent.com/primeqa/clapnq/main/retrieval/train/question_train_answerable.tsv

wget -nc -O data/clapnq/passages.tsv.zip \
  https://github.com/primeqa/clapnq/raw/main/retrieval/passages.tsv.zip

unzip -n data/clapnq/passages.tsv.zip -d data/clapnq

wc -l data/clapnq/question_dev_answerable.tsv \
  data/clapnq/question_train_answerable.tsv \
  data/clapnq/passages.tsv

