# RAG 系统训练与评估作业设计

## 1. 作业目标

本作业要求学生完成一个完整的 RAG 系统实验，包含三个部分：

1. **训练并评估一个基于 BERT 的 embedding 模型**
2. **构建完整 RAG 系统，并完成多组对比实验**
3. **微调 generator，并在测试集上比较微调前后的 RAG 效果**

整体实验重点是：

- 让学生理解 BERT 如何通过对比学习训练成 dense retriever；
- 让学生观察检索质量、reranker、Top-k 设置对 RAG 效果的影响；
- 让学生评估 generator 微调是否能提升端到端问答效果。

## 2. 数据集、模型与工具总览

本作业使用 **DPR-NQ** 训练和评估 BERT retriever，使用 **CLAPNQ** 构建完整 RAG 系统并训练 generator。整体资源如下表所示。

| 块 | 名称 | 类型 | 是否必做 | 作用 | 地址 |
|---|---|---|---|---|---|
| Embedding 训练数据集（两个，加了一个硬负例） | DPR Natural Questions train   | 英文开放域检索训练数据集 | 必做     | 训练 dense retriever，使 BERT 学习 question-passage 匹配；使用 `data.retriever.nq-train`，包含 positive_ctxs、negative_ctxs 和 hard_negative_ctxs | [biencoder-nq-train.json.gz](https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-train.json.gz)     [biencoder-nq-adv-hn-train.json.gz](https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-adv-hn-train.json.gz) |
| Embedding 单独评估集                         | DPR Natural Questions dev     | 英文开放域检索评估集     | 必做     | 单独评估 BERT retriever 的 MRR@10、Recall@5、Recall@20、Recall@50；使用 `data.retriever.nq-dev` | [biencoder-nq-dev.json.gz](https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-dev.json.gz) |
| Embedding 模型基座 | google-bert/bert-base-uncased | 英文 BERT 编码器基座 | 必做 | 原始 BERT 本身不是专门的 embedding 模型，通过 mean pooling、LoRA 和对比学习训练后作为 dense retriever | [google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) |
| Embedding 对比模型 | BAAI/bge-base-en-v1.5 | 英文开源 embedding 模型 | 必做 | 与训练后的 BERT retriever 做同参数量级对比，观察 BERT 训练后与成熟 embedding 模型的差距 | [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) |
| 完整 RAG 测试集 | CLAPNQ dev | 专门 RAG 系统评估数据集 | 必做 | 在测试集上评估 retriever、reranker、generator 组合后的完整 RAG 效果 | https://raw.githubusercontent.com/primeqa/clapnq/main/retrieval/dev/question_dev_answerable.tsv |
| Generator 训练数据集 | CLAPNQ train | 英文长答案生成训练数据集 | 必做 | 使用 question + gold/retrieved passage → long answer 的形式微调 generator | https://raw.githubusercontent.com/primeqa/clapnq/main/retrieval/train/question_train_answerable.tsv |
| Generator 单独评估集 | CLAPNQ dev | 英文长答案问答评估集 | 必做 | 给定 question 和 passage，单独评估 generator 微调前后的回答能力 | https://raw.githubusercontent.com/primeqa/clapnq/main/retrieval/dev/question_dev_answerable.tsv |
| RAG 检索数据库 | CLAPNQ passages corpus | 官方检索语料库 | 必做 | 作为完整 RAG 测试时的检索数据库；只放 passages，不放测试集 question-answer pair | https://github.com/primeqa/clapnq/raw/main/retrieval/passages.tsv.zip |
| Generator 模型 | Qwen2.5-1.5B-Instruct | 英文/多语言生成模型 | 必做 | 作为 RAG 系统中的答案生成模型，并使用 CLAPNQ train 进行 LoRA 微调 | [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) |
| Reranker 模型 | BAAI/bge-reranker-base | 英文 reranker 模型 | 必做 | 对 retriever 召回的 Top-N passages 进行重排，用于 RAG Top-k + Reranker 对比实验 | [BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base) |
| 向量检索库 | FAISS | 向量检索库 | 必做 | 存储 CLAPNQ corpus 的 dense embedding，支持 Top-k 向量检索 | [facebookresearch/faiss](https://github.com/facebookresearch/faiss) |
| BM25 工具 | Pyserini | 稀疏检索工具 | 推荐必做 | 在 CLAPNQ corpus 上构建 BM25 baseline，也可用于 hard negative 挖掘 | [castorini/pyserini](https://github.com/castorini/pyserini) |
| 微调工具 | PEFT / LoRA | 参数高效微调工具 | 必做 | 用于 BERT retriever 和 Qwen generator 的 LoRA 微调，降低显存成本 | [huggingface/peft](https://github.com/huggingface/peft) |



## 3. 数据集规模

### 3.1 DPR-NQ

DPR 官方 retriever 训练数据格式包含：

```json
{
  "question": "...",
  "answers": ["..."],
  "positive_ctxs": [
    {
      "title": "...",
      "text": "..."
    }
  ],
  "negative_ctxs": [
    {
      "title": "...",
      "text": "..."
    }
  ],
  "hard_negative_ctxs": [
    {
      "title": "...",
      "text": "..."
    }
  ]
}
```

### 3.2 CLAPNQ

CLAPNQ 用于任务二和任务三：完整 RAG 系统构建、RAG 测试、generator 训练和评估。

CLAPNQ 的特点：

```text
1. 提供现成 passage corpus，不需要学生自己从 Wikipedia 构建语料库。
2. 每个 answerable question 有 gold passage，可以评估 retriever 是否检索到证据。
3. 提供 long-form answer，可以评估 generator 的生成质量。
4. corpus 规模只有 178,891 passages。
```

## 4. 模型选择

### 4.1 Embedding / Retriever 模型

主模型：

```text
google-bert/bert-base-uncased
```

地址：

```text
https://huggingface.co/google-bert/bert-base-uncased
```

BERT-base-uncased 参数规模：

```text
12 layers, 768 hidden size, 12 attention heads, 110M parameters
```

本作业要求学生将 BERT 训练成 bi-encoder dense retriever，而不是直接裸用 BERT。

对比模型：

```text
BAAI/bge-base-en-v1.5
```

地址：

```text
https://huggingface.co/BAAI/bge-base-en-v1.5
```

BGE-base-en-v1.5 参数量约 109M，与 BERT-base 的 110M 接近，适合作为参数量相近的开源 embedding 模型对比。

### 4.2 Reranker 模型

推荐使用：

```text
BAAI/bge-reranker-base
```

地址：

```text
https://huggingface.co/BAAI/bge-reranker-base
```

reranker 与 embedding 模型不同。embedding 模型分别编码 query 和 passage，再计算向量相似度；reranker 直接把 query 和 passage 拼接输入模型，输出相关性分数，因此排序通常更准确，但计算更慢。

### 4.3 Generator 模型

推荐使用：

```text
Qwen/Qwen2.5-1.5B-Instruct
```

地址：

```text
https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
```

训练方式：

```text
LoRA supervised fine-tuning
```

## 5. 实验一：BERT Embedding 模型训练与评估

### 5.1 实验目标

将 `bert-base-uncased` 训练成一个 dense retriever，用于后续 RAG 系统中的向量检索。

需要比较：

1. 原始 BERT mean pooling；
2. BERT + LoRA 检索训练；
3. BERT + LoRA + hard negatives；
4. 参数量相近的开源 embedding 模型 BGE-base-en-v1.5。

### 5.2 Retriever 结构

采用 bi-encoder 结构：

```text
question → BERT encoder → pooling → query embedding

passage  → BERT encoder → pooling → passage embedding
```

推荐设置：

| 项目 | 设置 |
|---|---|
| Backbone | `google-bert/bert-base-uncased` |
| Pooling | mean pooling |
| Padding 处理 | 使用 attention mask 排除 padding token |
| 向量归一化 | L2 normalization |
| 相似度 | cosine similarity 或 dot product |
| 微调方式 | LoRA |
| 输出维度 | 768 |

mean pooling：

```text
embedding = sum(last_hidden_state * attention_mask) / sum(attention_mask)
```

### 5.3 训练数据构造

使用 DPR-NQ train。

每个样本构造成：

```text
query = question
positive passage = positive_ctxs
normal negatives = negative_ctxs
hard negatives = hard_negative_ctxs
```

示例：

```text
Query:
who wrote the song don't let me down

Positive passage:
"Don't Let Me Down" is a song by the Beatles, written by John Lennon and credited to Lennon–McCartney.

Negative passage:
"Let Me Down Easy" is a song recorded by ...

Hard negative passage:
"Don't Let Me Down" is a song title used by multiple artists ...
```

### 5.4 训练策略

建议两阶段训练。

#### 第一阶段：基础对比学习

使用：

```text
positive_ctxs + negative_ctxs + in-batch negatives
```

目的：

```text
让 BERT 初步学会区分相关 passage 和明显不相关 passage。
```

#### 第二阶段：hard negative 训练

使用：

```text
positive_ctxs + hard_negative_ctxs + in-batch negatives
```

目的：

```text
提升模型区分“表面相关但不能回答问题”的 passage 的能力。
```

hard negative 很重要，因为真实 RAG 检索中，错误 passage 往往不是完全无关，而是主题接近但不能支持答案。

### 5.5 损失函数

使用 InfoNCE / Multiple Negatives Ranking Loss。

![image-20260511123033031](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20260511123033031.png)

其中：

| 符号 | 含义 |
|---|---|
| q | question embedding |
| p+ | positive passage embedding |
| pi- | negative passage embedding |
| sim(q, p) | 向量相似度 |
| τ | temperature |

### 5.6 推荐训练配置

| 项目 | 推荐值 |
|---|---|
| max query length | 64 |
| max passage length | 256 |
| batch size | 32–128，视显存而定 |
| LoRA rank | 8 或 16 |
| LoRA alpha | 16 或 32 |
| LoRA dropout | 0.05 |
| learning rate | 1e-5 到 5e-5 |
| epoch | 1–3 |
| precision | fp16 / bf16 |

训练时间预估：

#### 5.7 测试集评估方式

在 DPR-NQ dev 上评估训练效果。

课程基础版不要求索引 DPR 全量 2100 万 Wikipedia passages，而是采用候选集排序评估。

对每个 dev query：

```text
candidate passages =
positive_ctxs + negative_ctxs + hard_negative_ctxs
```

模型对 candidate passages 排序，然后计算：

| 指标 | 含义 |
|---|---|
| MRR@10 | 第一个 positive passage 排名越靠前越好 |
| Recall@10 | Top-10 中召回的 positive passage 比例 |
| Recall@20 | Top-20 中召回的 positive passage 比例 |
| Recall@30 | Top-30 中召回的 positive passage 比例 |

Recall@k：

```text
Recall@k = Top-k 中召回到的 positive passage 数量 / 该问题全部 positive passage 数量
```

MRR@10：

```text
如果第一个 positive passage 出现在第 r 名，则 RR@10 = 1/r；
如果 Top-10 没有 positive passage，则 RR@10 = 0；
MRR@10 是所有 query 的 RR@10 平均值。
```

### 5.8 实验一结果表

在 DPR-NQ dev 上采用候选集排序评估。对于每个 query，将 `positive_ctxs`、`negative_ctxs` 和 `hard_negative_ctxs` 合并为初始候选集合。

| 方法 | 参数量 | MRR@10 | Recall@10 | Recall@20 | Recall@30 |
|---|---:|---:|---:|---:|---:|
| BERT-base + mean pooling | 110M |  |  |  |  |
| BERT-base + LoRA | 110M |  |  |  |  |
| BERT-base + LoRA + hard negatives | 110M |  |  |  |  |
| BGE-base-en-v1.5 | 109M |  |  |  |  |

需要分析：

1. 原始 BERT mean pooling 是否明显弱于训练后的 BERT retriever；
2. 加入 hard negatives 后，MRR@10 和 Recall@k 是否提升；
3. 训练后的 BERT retriever 与同等参数量开源 embedding 模型 BGE-base-en-v1.5 差距有多大。

## 6. 实验二：完整 RAG 系统构建与评估

### 6.1 实验目标

使用实验一训练得到的 BERT retriever，在 CLAPNQ corpus 上构建完整 RAG 系统，并比较不同检索设置对最终问答效果的影响。

### 6.2 RAG 语料库

语料库地址：

```text
https://github.com/primeqa/clapnq/raw/main/retrieval/passages.tsv.zip
```

测试集中的字段用途：

| 字段           | 来源                          | 用途                                                         |
| -------------- | ----------------------------- | ------------------------------------------------------------ |
| `question`     | `question_dev_answerable.tsv` | 作为检索 query，输入 retriever 从 `passages.tsv` 中检索相关 passage |
| `doc-id-list`  | `question_dev_answerable.tsv` | gold passage id，只用于评估 retriever / reranker 是否召回并排好证据；不直接输入模型 |
| `answers`      | `question_dev_answerable.tsv` | 作为 reference answer，用于评估 generator 输出的 F1 和 ROUGE-L |
| `id`           | `question_dev_answerable.tsv` | 样本 id，用于结果记录、错误分析和对齐样本                    |
| `passage id`   | `passages.tsv`                | 语料库中每个 passage 的唯一编号，用于和 `doc-id-list` 对齐   |
| `passage text` | `passages.tsv`                | 检索数据库中的 passage 内容，retriever / reranker 检索后输入 generator |

### 6.3 完整 RAG 系统流程

完整 RAG 系统采用：

```text
CLAPNQ dev question
  ↓
BERT retriever 从 CLAPNQ corpus 检索 Top-N passages
  ↓
BGE reranker 对 Top-N passages 重新排序
  ↓
选择重排后的 Top-k passages
  ↓
Qwen2.5-1.5B-Instruct 生成答案
  ↓
与 CLAPNQ reference answer 对比
```

推荐参数：

```text
Top-N = 30
k = 1, 3, 5, 10
```

其中：

| 参数 | 含义 |
|---|---|
| Top-N | retriever 第一阶段召回的候选 passage 数量 |
| k | reranker 重排后最终输入 generator 的 passage 数量 |

### 6.4 第一步：不同 k 的影响实验

这一步只在完整 RAG 设置下进行，即：

```text
BERT retriever + BGE reranker + Generator
```

分别设置：

```text
k = 1, 3, 5, 10
```

实验目的：

```text
观察最终输入 generator 的 passage 数量对 RAG 效果的影响，并选择一个最佳 k 供后续对比实验使用。
```

需要观察：

1. k 太小时，可能没有包含足够证据；
2. k 增大时，gold passage 被输入 generator 的概率可能提高；
3. k 过大时，噪声 passage 增加，generator 可能被干扰；
4. 最终选择 F1 / ROUGE-L 综合效果最好的 k 作为后续实验固定值。

#### 表 1：RAG + Reranker 下不同 k 的生成效果

从根据测试集query从语料库里面检索得到top-k,然后生成模型根据top-k回答问题

| k | EM（该指标不做） | F1 | ROUGE-L |
|---:|---:|---:|---:|
| 1 |  |  |  |
| 3 |  |  |  |
| 5 |  |  |  |
| 10 |  |  |  |

#### 表 2：RAG + Reranker 下不同 k 的检索效果

检索到得top-k的质量

| k | Recall@k | MRR@10 | （选做）nDCG@10 |
|---:|---:|---:|---:|
| 1 |  |  |  |
| 3 |  |  |  |
| 5 |  |  |  |
| 10 |  |  |  |

最终选定：

```text
k* = 在 CLAPNQ dev 上表现最好的 k
```

如果多个 k 的效果接近，优先选择较小的 k，以减少输入长度和噪声。

### 6.5 第二步：固定 k* 的四组对比实验

在第一步确定最佳 k 后，固定：

```text
k = k*
Top-N = 30
```

然后进行四组对比实验。

| 实验组 | 输入给 Generator 的内容 | 作用 |
|---|---|---|
| No RAG | 只输入 question | 测试 generator 不使用外部知识时的表现 |
| Random-k | question + 从 CLAPNQ corpus 随机选取 k 个 passages | 排除“输入变长自然变好”的影响 |
| RAG Top-k without Reranker | question + BERT retriever 直接检索到的 Top-k passages | 无 reranker 的 RAG baseline |
| RAG Top-k with Reranker | question + BERT retriever Top-N 后由 BGE reranker 重排出的 Top-k passages | 完整 RAG 主系统 |

这里要特别说明：

```text
RAG Top-k with Reranker 是本实验的完整 RAG 系统；
RAG Top-k without Reranker 只是为了对比 reranker 的贡献。
```

### 6.6 RAG 检索指标

在 CLAPNQ dev 上计算：

| 指标 | 含义 |
|---|---|
| Recall@k | Top-k 中是否包含 gold passage |
| MRR@10 | gold passage 排名越靠前越好 |
| nDCG@10 | 考虑相关 passage 排名位置的排序指标 |

CLAPNQ 通常一个 answerable question 有一个 gold passage，因此 Recall@k 可以理解为：

```text
Top-k 中包含 gold passage，则该样本记为 1；
否则记为 0；
最终对所有样本取平均。
```

对于 reranker 实验，检索指标应在 reranker 重排后的结果上计算。

### 6.7 RAG 生成指标

| 指标 | 含义 |
|---|---|
| EM | 生成答案是否与标准答案完全一致 |
| F1 | 生成答案与标准答案的 token 重合程度 |
| ROUGE-L | 生成答案与标准答案的最长公共子序列相似度 |

CLAPNQ 是 long-form answer 数据集，EM 通常较低，因此重点看：

```text
F1 和 ROUGE-L
```

### 6.8 Prompt 模板

RAG 组统一使用：

```text
You are given a question and several passages.
Answer the question using only the information in the passages.
If the passages do not contain enough information, answer "I don't know".

Question:
{question}

Passages:
[1] {title_1}: {passage_1}
[2] {title_2}: {passage_2}
...
[k] {title_k}: {passage_k}

Answer:
```

No RAG 组使用：

```text
Answer the following question.
If you do not know the answer, answer "I don't know".

Question:
{question}

Answer:
```

### 6.9 实验二最终结果表

#### 表 3：固定 k* 后四组方法的生成效果

从根据测试集query从语料库里面检索得到top-k,然后生成模型根据top-k回答问题，评估生成质量

| 方法 | k | EM（不用做） | F1 | ROUGE-L |
|---|---:|---:|---:|---:|
| No RAG | - |  |  |  |
| Random-k | k* |  |  |  |
| RAG Top-k without Reranker | k* |  |  |  |
| RAG Top-k with Reranker（表二） | k* |  |  |  |

#### 表 4：固定 k* 后 RAG 方法的检索效果

检索加重排得到的top-k的质量

| 方法 | k | Recall@k | MRR@10 | （选做）nDCG@10 |
|---|---:|---:|---:|---:|
| RAG Top-k without Reranker | k* |  |  |  |
| RAG Top-k with Reranker（表二） | k* |  |  |  |

需要分析：

1. RAG Top-k with Reranker 是否优于 No RAG；
2. RAG Top-k with Reranker 是否优于 Random-k；
3. RAG Top-k with Reranker 是否优于 RAG Top-k without Reranker；
4. reranker 是否提升检索排序和最终生成效果；
5. 最佳 k 是多少，为什么这个 k 的效果最好。

## 7. 实验三：Generator 微调与端到端 RAG 效果评估

### 7.1 实验目标

使用 CLAPNQ train 微调 generator，然后在 CLAPNQ 测试集上比较微调前后的端到端 RAG 效果。

实验三延续实验二的设置：

```text
k = k*
Top-N = 30
完整 RAG = BERT retriever + BGE reranker + Generator
```

其中，`k*` 是实验二中通过不同 k 对比实验选出的最佳 k。

本实验重点比较：

2. generator 微调是否提升完整 RAG 条件下的回答质量；
4. 微调后的 generator 是否更会利用给定 passages，而不是凭自身知识回答。

### 7.2 训练数据

使用：

```text
CLAPNQ train: 1954条  questions         #该数据集包含可回答的问题和不可回答的问题，本实验给出的数据集仅包含可回答的问题。
```

训练样本构造为：

```text
Input:
question + passage(s)

Output:
reference long answer
```

这里的passage的来源用question从构建的RAG数据库里面检索出的passage,而非训练集中已有的gold passage.

### 7.3 Generator 训练配置

| 项目 | 推荐值 |
|---|---|
| 模型 | `Qwen/Qwen2.5-1.5B-Instruct` |
| 微调方式 | LoRA |
| max input length | 1024 或 1536 |
| max output length | 128 或 256 |
| epoch | 2–5 |
| learning rate | 1e-5 到 5e-5 |
| precision | fp16 / bf16 |
| GPU | 无 |

### 7.4 测试集对比设置

在 CLAPNQ 测试集 上评估，并固定实验二选出的最佳 k：

```text
k = k*
Top-N = 30
```

对比方法如下：

| 方法 | Retriever | Reranker | Generator | 作用 |
|---|---|---|---|---|
| 原始 Generator，无 RAG，gold passage | 无 | 无 | 原始 Qwen2.5-1.5B-Instruct | 测试原始模型不使用外部知识时的表现 |
| 微调 Generator，无 RAG, gold passage | 无 | 无 | LoRA 微调后的 Qwen2.5-1.5B-Instruct | 测试微调是否提升模型自身回答能力 |
| 原始 Generator，RAG with Reranker | BERT retriever | BGE reranker | 原始 Qwen2.5-1.5B-Instruct | 微调前的完整 RAG 系统 |
| 微调 Generator，RAG with Reranker | BERT retriever | BGE reranker | LoRA 微调后的 Qwen2.5-1.5B-Instruct | 微调后的完整 RAG 系统 |

### 7.5 实验三结果表

在无RAG条件下对比原始Generator和微调Generator下，可以直接用测试集的query+gold passage输入模型来生成答案，看是否训练有效。在有RAG条件下，就不能引入gold passage,而是要从RAG里面检索了。

| 方法 | k | EM（不做） | F1 | ROUGE-L |
|---|---:|---:|---:|---:|
| 原始 Generator，无 RAG，gold passage | - |  |  |  |
| 微调 Generator，无 RAG,   gold passage | - |  |  |  |
| 原始 Generator，RAG with Reranker | k* |  |  |  |
| 微调 Generator，RAG with Reranker | k* |  |  |  |

需要分析：

1. 微调 generator 是否提升生成模型的 F1 和 ROUGE-L；
2. 在完整 RAG 设置下，微调 generator 是否进一步提升生成效果；

## 8. 最终提交内容

学生需要提交：

1. 实验报告；
2. BERT retriever 训练代码；
3. DPR-NQ 测试集评估结果；
4. 与 BGE-base-en-v1.5 的对比结果；
5. CLAPNQ corpus 建库和 FAISS 检索代码；
6. RAG 四组对比实验结果；
7. 不同 k 的影响分析；
8. generator LoRA 微调代码；
9. generator 微调前后测试集评估结果；
10. 错误样例分析。

## 9. 基础要求与加分项

### 9.1 要求

1. 使用 BERT-base 构建 bi-encoder retriever；
2. 使用 DPR-NQ train 训练 BERT retriever；
3. 在 DPR-NQ dev 上评估 MRR@10 和 Recall@k；
4. 与 BGE-base-en-v1.5 做同参数量级对比；
5. 使用 CLAPNQ passages 构建 FAISS 检索库；
6. 在 CLAPNQ dev 上完成 No RAG、Random-k、RAG Top-k 三组实验；
7. 设置不同 k，观察 RAG 效果变化；
8. 使用 CLAPNQ train 微调 generator；
9. 在 CLAPNQ dev 上比较 generator 微调前后的效果。
