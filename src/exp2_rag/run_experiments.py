import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Sequence

import faiss
import numpy as np
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from exp1_retriever.model import build_retriever_model, encode_texts
from exp1_retriever.utils import get_device, print_gpu_info, save_json, set_seed
from exp2_rag.data import Passage, load_jsonl, read_passages, read_questions, sample_random_passages, save_jsonl
from exp2_rag.metrics import format_metrics_table, max_generation_scores, mean_dicts, retrieval_metrics
from exp2_rag.prompts import build_no_rag_prompt, build_rag_prompt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions_path", type=str, default="data/clapnq/question_dev_answerable.tsv")
    parser.add_argument("--passages_path", type=str, default="data/clapnq/passages.tsv")
    parser.add_argument("--index_path", type=str, default="indexes/clapnq_stage2.faiss")
    parser.add_argument("--output_dir", type=str, default="results/exp2")

    parser.add_argument("--retriever_base_model", type=str, default="models/bert-base-uncased")
    parser.add_argument("--retriever_adapter_path", type=str, default="outputs/stage2")
    parser.add_argument("--reranker_model", type=str, default="models/bge-reranker-base")
    parser.add_argument("--generator_model", type=str, default="models/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--reranker_mode", type=str, default="cross_encoder", choices=["cross_encoder", "embedding", "none"])

    parser.add_argument("--top_n", type=int, default=30)
    parser.add_argument("--k_values", type=str, default="1,3,5,10")
    parser.add_argument("--best_k", type=int, default=0)
    parser.add_argument("--max_eval_samples", type=int, default=0)
    parser.add_argument("--max_query_length", type=int, default=64)
    parser.add_argument("--max_passage_length", type=int, default=256)
    parser.add_argument("--retriever_batch_size", type=int, default=64)
    parser.add_argument("--reranker_batch_size", type=int, default=32)
    parser.add_argument("--reranker_max_length", type=int, default=512)
    parser.add_argument("--generation_batch_size", type=int, default=1)
    parser.add_argument("--max_input_length", type=int, default=1536)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_generation", action="store_true")
    parser.add_argument("--force_retrieve", action="store_true")
    parser.add_argument("--force_rerank", action="store_true")
    return parser.parse_args()


def resolve_model_path(path_or_name: str) -> str:
    return path_or_name if os.path.exists(path_or_name) else path_or_name


def passages_from_meta(rows: Sequence[Dict]) -> List[Passage]:
    return [Passage(id=row["id"], title=row.get("title", ""), text=row.get("text", "")) for row in rows]


def run_retrieval(args, examples, passages, device) -> List[Dict]:
    cache_path = Path(args.output_dir) / f"retrieval_top{args.top_n}.jsonl"
    if cache_path.exists() and not args.force_retrieve:
        return load_jsonl(str(cache_path))

    index = faiss.read_index(args.index_path)
    model, tokenizer = build_retriever_model(
        base_model_name_or_path=args.retriever_base_model,
        adapter_path=args.retriever_adapter_path,
        trainable_adapter=False,
    )
    model.to(device)
    model.eval()

    rows = []
    questions = [example.question for example in examples]
    q_embs = encode_texts(
        model=model,
        tokenizer=tokenizer,
        texts=questions,
        max_length=args.max_query_length,
        batch_size=args.retriever_batch_size,
        device=device,
    ).numpy().astype("float32")

    scores, indices = index.search(q_embs, args.top_n)
    for example, row_scores, row_indices in zip(examples, scores, indices):
        hits = []
        for score, idx in zip(row_scores, row_indices):
            passage = passages[int(idx)]
            hits.append({"row": int(idx), "id": passage.id, "score": float(score)})
        rows.append({"id": example.id, "question": example.question, "gold_doc_ids": example.gold_doc_ids, "hits": hits})

    save_jsonl(rows, str(cache_path))
    return rows


class BGEReranker:
    def __init__(self, model_name_or_path: str, device: torch.device, max_length: int, batch_size: int):
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(resolve_model_path(model_name_or_path), use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(resolve_model_path(model_name_or_path))
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def score(self, question: str, passages: List[Passage]) -> List[float]:
        scores: List[float] = []
        pairs = [(question, passage.retriever_text) for passage in passages]
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start:start + self.batch_size]
            enc = self.tokenizer(
                [q for q, _ in batch],
                [p for _, p in batch],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            enc = {key: value.to(self.device) for key, value in enc.items()}
            logits = self.model(**enc).logits.view(-1)
            scores.extend(logits.detach().float().cpu().tolist())
        return scores


class EmbeddingReranker:
    def __init__(self, model_name_or_path: str, batch_size: int):
        self.batch_size = batch_size
        self.model = SentenceTransformer(resolve_model_path(model_name_or_path))

    def score(self, question: str, passages: List[Passage]) -> List[float]:
        texts = [passage.retriever_text for passage in passages]
        q_emb = self.model.encode([question], batch_size=1, normalize_embeddings=True, show_progress_bar=False)
        p_emb = self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=False)
        return (p_emb @ q_emb[0]).astype(float).tolist()


def run_reranking(args, retrieval_rows, passages, device) -> List[Dict]:
    cache_path = Path(args.output_dir) / f"reranked_top{args.top_n}.jsonl"
    if cache_path.exists() and not args.force_rerank:
        return load_jsonl(str(cache_path))

    if args.reranker_mode == "none":
        rows = [{**row, "hits": [dict(hit, rerank_score=hit["score"]) for hit in row["hits"]]} for row in retrieval_rows]
        save_jsonl(rows, str(cache_path))
        return rows

    if args.reranker_mode == "embedding":
        reranker = EmbeddingReranker(
            model_name_or_path=args.reranker_model,
            batch_size=args.reranker_batch_size,
        )
    else:
        reranker = BGEReranker(
            model_name_or_path=args.reranker_model,
            device=device,
            max_length=args.reranker_max_length,
            batch_size=args.reranker_batch_size,
        )
    rows = []
    for row in tqdm(retrieval_rows, desc="reranking"):
        candidate_passages = [passages[hit["row"]] for hit in row["hits"]]
        rerank_scores = reranker.score(row["question"], candidate_passages)
        reranked = []
        for hit, score in zip(row["hits"], rerank_scores):
            new_hit = dict(hit)
            new_hit["rerank_score"] = float(score)
            reranked.append(new_hit)
        reranked.sort(key=lambda item: item["rerank_score"], reverse=True)
        rows.append({**row, "hits": reranked})

    save_jsonl(rows, str(cache_path))
    return rows


def retrieval_table(rows: List[Dict], k_values: Sequence[int]) -> List[Dict]:
    table = []
    for k in k_values:
        metric_rows = []
        for row in rows:
            ranked_ids = [hit["id"] for hit in row["hits"][:k]]
            metric_rows.append(retrieval_metrics(ranked_ids, row["gold_doc_ids"], recall_k=k, mrr_k=10, ndcg_k=10))
        metrics = mean_dicts(metric_rows)
        table.append({
            "k": k,
            f"Recall@k": metrics.get(f"recall@{k}", 0.0),
            "MRR@10": metrics.get("mrr@10", 0.0),
            "nDCG@10": metrics.get("ndcg@10", 0.0),
        })
    return table


class QwenGenerator:
    def __init__(self, model_name_or_path: str, device: torch.device, max_input_length: int, max_new_tokens: int, temperature: float):
        self.device = device
        self.max_input_length = max_input_length
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.tokenizer = AutoTokenizer.from_pretrained(resolve_model_path(model_name_or_path), trust_remote_code=True, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            resolve_model_path(model_name_or_path),
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        self.model.eval()

    def _chat_prompt(self, prompt: str) -> str:
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt

    @torch.no_grad()
    def generate_batch(self, prompts: List[str]) -> List[str]:
        chat_prompts = [self._chat_prompt(prompt) for prompt in prompts]
        enc = self.tokenizer(
            chat_prompts,
            padding=True,
            truncation=True,
            max_length=self.max_input_length,
            return_tensors="pt",
        )
        enc = {key: value.to(self.model.device) for key, value in enc.items()}
        do_sample = self.temperature > 0
        generate_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            generate_kwargs["temperature"] = self.temperature
        output_ids = self.model.generate(**enc, **generate_kwargs)

        predictions = []
        input_len = enc["input_ids"].shape[1]
        for seq in output_ids:
            gen_ids = seq[input_len:]
            text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            predictions.append(text)
        return predictions


def run_generation_group(args, generator, examples, prompts: List[str], group_name: str) -> Dict:
    output_path = Path(args.output_dir) / f"predictions_{group_name}.jsonl"
    metric_rows = []
    pred_rows = []
    for start in tqdm(range(0, len(prompts), args.generation_batch_size), desc=f"generating {group_name}"):
        batch_prompts = prompts[start:start + args.generation_batch_size]
        batch_examples = examples[start:start + args.generation_batch_size]
        preds = generator.generate_batch(batch_prompts)
        for example, prompt, pred in zip(batch_examples, batch_prompts, preds):
            scores = max_generation_scores(pred, example.answers)
            metric_rows.append(scores)
            pred_rows.append({
                "id": example.id,
                "question": example.question,
                "prediction": pred,
                "answers": example.answers,
                "scores": scores,
                "prompt": prompt,
            })

    save_jsonl(pred_rows, str(output_path))
    metrics = mean_dicts(metric_rows)
    metrics["num_eval_examples"] = float(len(metric_rows))
    return metrics


def select_best_k(k_generation_rows: List[Dict], fallback_k: int) -> int:
    if not k_generation_rows:
        return fallback_k
    ordered = sorted(k_generation_rows, key=lambda row: (-row.get("F1", 0.0), -row.get("ROUGE-L", 0.0), row["k"]))
    return int(ordered[0]["k"])


def write_markdown_report(args, k_gen_table, k_ret_table, comparison_gen_table, comparison_ret_table, best_k):
    report_path = Path(args.output_dir) / "experiment2_report.md"
    lines = [
        "# 实验二：完整 RAG 系统构建与评估",
        "",
        "## 设置",
        "",
        f"- Retriever: `{args.retriever_base_model}` + `{args.retriever_adapter_path}`",
        f"- Reranker: `{args.reranker_model}`",
        f"- Reranker mode: `{args.reranker_mode}`",
        f"- Generator: `{args.generator_model}`",
        f"- CLAPNQ dev 样本数: `{args.max_eval_samples if args.max_eval_samples > 0 else 'all'}`",
        f"- Top-N: `{args.top_n}`",
        f"- k*: `{best_k}`",
        "",
        "## 表 1：RAG + Reranker 下不同 k 的生成效果",
        "",
        format_metrics_table(k_gen_table, ["k", "F1", "ROUGE-L"]),
        "",
        "## 表 2：RAG + Reranker 下不同 k 的检索效果",
        "",
        format_metrics_table(k_ret_table, ["k", "Recall@k", "MRR@10", "nDCG@10"]),
        "",
        "## 表 3：固定 k* 后四组方法的生成效果",
        "",
        format_metrics_table(comparison_gen_table, ["方法", "k", "F1", "ROUGE-L"]),
        "",
        "## 表 4：固定 k* 后 RAG 方法的检索效果",
        "",
        format_metrics_table(comparison_ret_table, ["方法", "k", "Recall@k", "MRR@10", "nDCG@10"]),
        "",
        "## 简要分析",
        "",
        f"本实验在 CLAPNQ corpus 上用实验一得到的 stage2 BERT LoRA retriever 建立 FAISS 向量索引，先召回 Top-{args.top_n}，再用 BGE reranker 重排。k* 按生成 F1 优先、ROUGE-L 次优先、较小 k 优先的规则选择。",
        "",
        "No RAG 只依赖 generator 自身知识；Random-k 用随机 passage 控制输入变长的影响；without Reranker 反映 dense retriever 直接 Top-k 的效果；with Reranker 是完整 RAG 主系统。检索表中的提升可以解释 reranker 是否把 gold passage 排到更靠前位置，生成表中的 F1/ROUGE-L 则反映这些证据是否被 generator 有效利用。",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return str(report_path)


def main():
    args = parse_args()
    set_seed(args.seed)
    print_gpu_info()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    device = get_device()
    k_values = [int(k.strip()) for k in args.k_values.split(",") if k.strip()]
    examples = read_questions(
        args.questions_path,
        max_samples=args.max_eval_samples if args.max_eval_samples > 0 else None,
    )
    passages = read_passages(args.passages_path)

    retrieval_rows = run_retrieval(args, examples, passages, device)
    reranked_rows = run_reranking(args, retrieval_rows, passages, device)

    k_ret_table = retrieval_table(reranked_rows, k_values)
    save_json(k_ret_table, str(Path(args.output_dir) / "table2_k_retrieval.json"))

    k_gen_table = []
    comparison_gen_table = []
    if not args.skip_generation:
        generator = QwenGenerator(
            model_name_or_path=args.generator_model,
            device=device,
            max_input_length=args.max_input_length,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

        for k in k_values:
            prompts = []
            for row in reranked_rows:
                selected = [passages[hit["row"]] for hit in row["hits"][:k]]
                prompts.append(build_rag_prompt(row["question"], selected))
            metrics = run_generation_group(args, generator, examples, prompts, f"rag_rerank_k{k}")
            k_gen_table.append({"k": k, "F1": metrics.get("f1", 0.0), "ROUGE-L": metrics.get("rouge_l", 0.0)})

        save_json(k_gen_table, str(Path(args.output_dir) / "table1_k_generation.json"))
        best_k = args.best_k if args.best_k > 0 else select_best_k(k_gen_table, fallback_k=k_values[0])

        rng = random.Random(args.seed)
        group_prompts = {
            "No RAG": [build_no_rag_prompt(example.question) for example in examples],
            "Random-k": [
                build_rag_prompt(
                    example.question,
                    [passages[idx] for idx in sample_random_passages(passages, best_k, rng, exclude_ids=example.gold_doc_ids)],
                )
                for example in examples
            ],
            "RAG Top-k without Reranker": [
                build_rag_prompt(row["question"], [passages[hit["row"]] for hit in row["hits"][:best_k]])
                for row in retrieval_rows
            ],
            "RAG Top-k with Reranker": [
                build_rag_prompt(row["question"], [passages[hit["row"]] for hit in row["hits"][:best_k]])
                for row in reranked_rows
            ],
        }
        for group_name, prompts in group_prompts.items():
            metrics = run_generation_group(args, generator, examples, prompts, group_name.lower().replace(" ", "_").replace("-", ""))
            comparison_gen_table.append({
                "方法": group_name,
                "k": "-" if group_name == "No RAG" else best_k,
                "F1": metrics.get("f1", 0.0),
                "ROUGE-L": metrics.get("rouge_l", 0.0),
            })
        save_json(comparison_gen_table, str(Path(args.output_dir) / "table3_comparison_generation.json"))
    else:
        best_k = args.best_k if args.best_k > 0 else k_values[0]
        save_json(k_gen_table, str(Path(args.output_dir) / "table1_k_generation.json"))

    comparison_ret_table = []
    for name, rows in [
        ("RAG Top-k without Reranker", retrieval_rows),
        ("RAG Top-k with Reranker", reranked_rows),
    ]:
        metrics = mean_dicts([
            retrieval_metrics([hit["id"] for hit in row["hits"][:best_k]], row["gold_doc_ids"], recall_k=best_k, mrr_k=10, ndcg_k=10)
            for row in rows
        ])
        comparison_ret_table.append({
            "方法": name,
            "k": best_k,
            "Recall@k": metrics.get(f"recall@{best_k}", 0.0),
            "MRR@10": metrics.get("mrr@10", 0.0),
            "nDCG@10": metrics.get("ndcg@10", 0.0),
        })
    save_json(comparison_ret_table, str(Path(args.output_dir) / "table4_comparison_retrieval.json"))

    summary = {
        "best_k": best_k,
        "table1_k_generation": k_gen_table,
        "table2_k_retrieval": k_ret_table,
        "table3_comparison_generation": comparison_gen_table,
        "table4_comparison_retrieval": comparison_ret_table,
    }
    save_json(summary, str(Path(args.output_dir) / "summary.json"))
    report_path = write_markdown_report(args, k_gen_table, k_ret_table, comparison_gen_table, comparison_ret_table, best_k)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[save] report: {report_path}")


if __name__ == "__main__":
    main()
