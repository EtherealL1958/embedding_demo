# 模型下载说明

本目录用于存放实验一所需的 HuggingFace 模型文件。

由于模型文件较大，仓库中不直接提供模型权重。请在本目录下手动下载以下模型：

| 模型 | 用途 |
|---|---|
| `google-bert/bert-base-uncased` | BERT retriever 的基础模型，用于训练 BERT + LoRA embedding 模型 |
| `BAAI/bge-base-en-v1.5` | 开源 embedding 对比模型，用于和训练后的 BERT retriever 做性能对比 |

## 1. 安装 HuggingFace 下载工具下载模型

如果环境中没有 `hf` 命令，可以先安装：

```bash
pip install -U huggingface_hub


hf download google-bert/bert-base-uncased \
  --local-dir models/bert-base-uncased

hf download BAAI/bge-base-en-v1.5 \
  --local-dir models/bge-base-en-v1.5