# 实验一：BERT Retriever 训练与评估

# 实验一：BERT Retriever 训练与评估

本目录实现 RAG 作业中的实验一：BERT Retriever 训练与评估。

本实验的目标是将原始 `bert-base-uncased` 训练成一个 dense retriever，并观察普通负例训练、hard negative 训练对检索效果的影响。同时使用 `BAAI/bge-base-en-v1.5` 作为成熟 embedding 模型进行对比。

实验包含以下内容：

1. 原始 `bert-base-uncased` + mean pooling 评估；
2. 使用 DPR-NQ train 训练 BERT + LoRA retriever；
3. 使用 hard negatives 继续训练；
4. 与 `BAAI/bge-base-en-v1.5` 做同候选集排序评估对比。

---

## 1. 目录结构

```text
rag_exp1/
├── configs/
│   └── exp1.yaml
├── data/
│   └── dpr/
│       └── README.md
├── models/
│   └── README.md
├── outputs/
│   ├── stage1/
│   └── stage2/
├── results/
├── scripts/
│   ├── eval_bert_raw.sh
│   ├── eval_bge.sh
│   ├── eval_lora_stage1.sh
│   ├── eval_lora_stage2.sh
│   ├── run_all.sh
│   ├── train_stage1.sh
│   └── train_stage2.sh
├── src/
│   └── exp1_retriever/
│       ├── __init__.py
│       ├── data.py
│       ├── evaluate.py
│       ├── evaluate_bge.py
│       ├── loss.py
│       ├── metrics.py
│       ├── model.py
│       ├── train.py
│       └── utils.py
├── requirements.txt
└── README.md
```

其中：

| 路径 | 说明 |
|---|---|
| `configs/exp1.yaml` | 实验参数记录文件，当前代码不会自动读取该文件 |
| `data/dpr/` | 存放 DPR-NQ 数据集 |
| `models/` | 存放本地 HuggingFace 模型 |
| `outputs/stage1/` | 第一阶段训练后的 LoRA adapter |
| `outputs/stage2/` | 第二阶段 hard negative 训练后的 LoRA adapter |
| `results/` | 存放评估结果 |
| `scripts/` | 训练和评估脚本 |
| `src/exp1_retriever/` | 主要 Python 代码 |

---

## 2. 环境安装

建议使用新的 conda 环境：

```bash
conda create -n zh_rag python=3.10 -y
conda activate zh_rag
```

安装依赖：

```bash
pip install -r requirements.txt --timeout 120 --retries 10
```

注意：`requirements.txt` 中默认使用 CUDA 12.8 对应的 PyTorch 版本。如果服务器 CUDA 版本不同，需要根据实际情况修改 PyTorch 的安装源和版本。

安装完成后检查 GPU 是否可用：

```bash
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
PY
```

如果输出中有：

```text
cuda available: True
```

说明 PyTorch 可以正常使用 GPU。

---

## 3. 数据集准备

本实验使用 DPR-NQ 数据。数据文件较大，不直接放在代码目录中，需要手动下载。

进入数据目录：

```bash
cd data/dpr
```

下载三个数据文件：

```bash
wget -O biencoder-nq-dev.json.gz \
https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-dev.json.gz

wget -O biencoder-nq-train.json.gz \
https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-train.json.gz

wget -O biencoder-nq-adv-hn-train.json.gz \
https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-adv-hn-train.json.gz
```

下载完成后回到项目根目录：

```bash
cd ../..
```

检查数据文件：

```bash
ls -lh data/dpr
```

应包含：

```text
biencoder-nq-train.json.gz
biencoder-nq-adv-hn-train.json.gz
biencoder-nq-dev.json.gz
```

三个文件用途如下：

| 文件 | 用途 |
|---|---|
| `biencoder-nq-train.json.gz` | 第一阶段训练数据，主要使用普通负例 `negative_ctxs` |
| `biencoder-nq-adv-hn-train.json.gz` | 第二阶段训练数据，主要使用困难负例 `hard_negative_ctxs` |
| `biencoder-nq-dev.json.gz` | Retriever 评估数据 |

---

## 4. 模型准备

本实验默认从本地路径读取模型，因此需要先下载 HuggingFace 模型到 `models/` 目录。

安装 HuggingFace 下载工具：

```bash
pip install -U huggingface_hub
```

如果服务器访问 HuggingFace 较慢，可以临时使用镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

下载模型：

```bash
mkdir -p models

hf download google-bert/bert-base-uncased \
  --local-dir models/bert-base-uncased

hf download BAAI/bge-base-en-v1.5 \
  --local-dir models/bge-base-en-v1.5
```

下载完成后检查：

```bash
ls models/bert-base-uncased
ls models/bge-base-en-v1.5
```

当前代码默认模型路径为：

```text
models/bert-base-uncased
models/bge-base-en-v1.5
```

不要随意修改文件夹名称。如果修改了模型路径，需要同步修改 `scripts/*.sh` 中的参数。

---

## 5. 实验方法说明

### 5.1 Retriever 结构

本实验使用 bi-encoder dense retriever：

```text
question → BERT encoder → mean pooling → query embedding

passage  → BERT encoder → mean pooling → passage embedding
```

然后使用向量相似度计算 question 和 passage 的匹配分数：

```text
score(q, p) = q · p
```

模型设置：

| 项目 | 设置 |
|---|---|
| Backbone | `bert-base-uncased` |
| Pooling | mean pooling |
| Padding 处理 | 使用 attention mask 排除 padding token |
| 向量归一化 | L2 normalization |
| 微调方式 | LoRA |
| 相似度 | dot product / cosine similarity |

---

### 5.2 两阶段训练

本实验采用两阶段训练。

第一阶段：

```text
使用 biencoder-nq-train.json.gz
主要使用 negative_ctxs 普通负例
```

目的：让 BERT 学会基本的 question-passage 匹配。

第二阶段：

```text
使用 biencoder-nq-adv-hn-train.json.gz
主要使用 hard_negative_ctxs 困难负例
```

目的：让模型进一步学会区分“表面相关但不能回答问题”的 passage。

---

### 5.3 Loss 设计

训练使用 InfoNCE / Multiple Negatives Ranking Loss。

对于一个 batch：

```text
B 个 question
B 个 positive passage
B 个 negative passage
```

模型计算：

```text
score matrix: B × 2B
```

第 `i` 个 question 的正确 passage 是第 `i` 个 positive passage，其余 passage 都作为负例。因此训练中同时包含：

1. 当前样本的显式 negative；
2. batch 内其他样本的 positive 和 negative，即 in-batch negatives。

---

## 6. 运行实验

### 6.1 一键运行完整实验

在项目根目录下执行：

```bash
bash scripts/run_all.sh
```

该脚本会依次执行：

```text
1. 评估原始 BERT mean pooling
2. 训练 stage1
3. 评估 stage1
4. 训练 stage2
5. 评估 stage2
6. 评估 BGE-base-en-v1.5
```

---

### 6.2 指定 GPU 运行

例如使用第 0 张 GPU：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_all.sh
```

注意：设置 `CUDA_VISIBLE_DEVICES=0` 后，程序内部看到的仍然是 `cuda:0`，这是正常的。

---

### 6.3 推荐运行配置

为了避免显存不足，推荐使用以下配置：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 \
BATCH_SIZE=64 \
GRAD_ACCUM_STEPS=2 \
EVAL_BATCH_SIZE=64 \
QUERY_BATCH_SIZE=64 \
PASSAGE_BATCH_SIZE=128 \
BGE_BATCH_SIZE=128 \
EPOCHS=3 \
MAX_TRAIN_SAMPLES=0 \
MAX_EVAL_SAMPLES=500 \
bash scripts/run_all.sh
```

其中：

| 参数 | 含义 |
|---|---|
| `BATCH_SIZE` | 训练 batch size |
| `GRAD_ACCUM_STEPS` | 梯度累积步数 |
| `EPOCHS` | 每个阶段的训练 epoch 数 |
| `MAX_TRAIN_SAMPLES` | 最大训练样本数，设为 0 表示使用全量训练集 |
| `MAX_EVAL_SAMPLES` | 最大评估样本数 |
| `EVAL_BATCH_SIZE` | 评估时 query batch size |
| `QUERY_BATCH_SIZE` | 评估时 query 编码 batch size |
| `PASSAGE_BATCH_SIZE` | 评估时 passage 编码 batch size |
| `BGE_BATCH_SIZE` | BGE 评估时 batch size |

---

### 6.4 快速测试

如果只是想确认环境和代码能否跑通，可以使用小规模数据：

```bash
CUDA_VISIBLE_DEVICES=0 \
BATCH_SIZE=16 \
MAX_TRAIN_SAMPLES=1000 \
MAX_EVAL_SAMPLES=100 \
bash scripts/run_all.sh
```

该设置只用于测试流程，不代表正式实验结果。

---

### 6.5 分步运行

评估原始 BERT：

```bash
bash scripts/eval_bert_raw.sh
```

训练 stage1：

```bash
bash scripts/train_stage1.sh
```

评估 stage1：

```bash
bash scripts/eval_lora_stage1.sh
```

训练 stage2：

```bash
bash scripts/train_stage2.sh
```

评估 stage2：

```bash
bash scripts/eval_lora_stage2.sh
```

评估 BGE：

```bash
bash scripts/eval_bge.sh
```

---

## 7. 输出文件说明

训练完成后，模型输出在：

```text
outputs/stage1/
outputs/stage2/
```

其中：

| 目录 | 说明 |
|---|---|
| `outputs/stage1/` | stage1 普通负例训练后的 LoRA adapter |
| `outputs/stage2/` | stage2 hard negative 训练后的 LoRA adapter |

评估结果保存在：

```text
results/
```

主要包括：

| 文件 | 说明 |
|---|---|
| `results/bert_raw.json` | 原始 BERT mean pooling 的评估结果 |
| `results/lora_stage1.json` | stage1 模型评估结果 |
| `results/lora_stage2.json` | stage2 模型评估结果 |
| `results/bge_base_en_v15.json` | BGE 对比模型评估结果 |

查看结果：

```bash
cat results/bert_raw.json
cat results/lora_stage1.json
cat results/lora_stage2.json
cat results/bge_base_en_v15.json
```

---

## 8. 评估指标

本实验在 DPR-NQ dev 上采用候选集排序评估。

对于每个 question，将以下 passages 合并为候选集合：

```text
positive_ctxs + negative_ctxs + hard_negative_ctxs
```

模型对候选 passages 进行排序，然后计算：

| 指标 | 含义 |
|---|---|
| `MRR@10` | 正样本出现在 Top-10 内时，按照排名倒数计分 |
| `Recall@1` | Top-1 中是否召回正样本 |
| `Recall@5` | Top-5 中召回的正样本比例 |
| `Recall@10` | Top-10 中召回的正样本比例 |
| `Recall@20` | Top-20 中召回的正样本比例 |
| `Recall@30` | Top-30 中召回的正样本比例 |
| `Recall@50` | Top-50 中召回的正样本比例 |

---

## 9. 结果表填写

实验报告中可以整理如下表格：

| 方法 | MRR@10 | Recall@1 | Recall@5 | Recall@10 | Recall@20 | Recall@30 | Recall@50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| BERT-base + mean pooling |  |  |  |  |  |  |  |
| BERT-base + LoRA |  |  |  |  |  |  |  |
| BERT-base + LoRA + hard negatives |  |  |  |  |  |  |  |
| BGE-base-en-v1.5 |  |  |  |  |  |  |  |

需要重点分析：

1. 原始 BERT mean pooling 是否弱于训练后的 BERT retriever；
2. stage1 普通负例训练是否提升检索效果；
3. stage2 hard negative 训练是否进一步提升排序质量；
4. 自训练 BERT retriever 与 BGE-base-en-v1.5 的差距。

---

## 10. 配置文件说明

`configs/exp1.yaml` 用于记录实验参数。当前 demo 代码不会自动读取该配置文件，实际运行参数由 `scripts/*.sh` 和命令行环境变量控制。

例如：

```bash
BATCH_SIZE=64 EPOCHS=3 MAX_TRAIN_SAMPLES=0 bash scripts/train_stage1.sh
```

会覆盖脚本中的默认参数。

---

## 11. 常见问题

### 11.1 CUDA 不可用

如果检查 PyTorch 时输出：

```text
cuda available: False
```

可能原因包括：

1. PyTorch CUDA 版本与服务器驱动不匹配；
2. 当前机器没有可用 GPU；
3. 没有正确安装 CUDA 版本的 PyTorch。

可以通过以下命令查看驱动支持的 CUDA 版本：

```bash
nvidia-smi
```

然后根据服务器 CUDA 版本修改 `requirements.txt` 中的 PyTorch 安装版本。

---

### 11.2 HuggingFace 模型下载失败

可以设置镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

然后重新执行：

```bash
hf download google-bert/bert-base-uncased \
  --local-dir models/bert-base-uncased
```

---

### 11.3 代理报错

如果出现类似：

```text
ProxyError: Cannot connect to proxy
```

可以先清除代理环境变量：

```bash
unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
unset all_proxy
unset ALL_PROXY
```

---

### 11.4 显存不足

如果出现：

```text
CUDA out of memory
```

可以减小 batch size：

```bash
BATCH_SIZE=32 GRAD_ACCUM_STEPS=2 bash scripts/run_all.sh
```

也可以指定：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

减少显存碎片问题。

---

### 11.5 磁盘空间不足

如果出现：

```text
No space left on device
```

说明磁盘空间不足。可以检查：

```bash
df -h
du -h --max-depth=1 ~ | sort -hr | head
```

必要时清理缓存：

```bash
rm -rf ~/.cache/pip
rm -rf ~/.cache/huggingface/hub
conda clean -a -y
```

---

## 12. 备注

本实验重点不是训练出超过成熟开源 embedding 模型的 retriever，而是通过一个完整流程理解 dense retriever 的训练和评估方法。

通常可以观察到如下趋势：

```text
原始 BERT mean pooling < BERT + LoRA < BERT + LoRA + hard negatives < BGE-base-en-v1.5
```

如果结果符合这一趋势，说明实验基本成功。