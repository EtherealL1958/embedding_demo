import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from exp1_retriever.utils import get_device, print_gpu_info, save_json, set_seed
from exp2_rag.data import Passage, QAExample, load_jsonl, read_passages, read_questions, save_jsonl
from exp2_rag.metrics import format_metrics_table, max_generation_scores, mean_dicts
from exp2_rag.prompts import build_rag_prompt
from exp2_rag.run_experiments import run_reranking, run_retrieval


IGNORE_INDEX = -100


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_questions_path", type=str, default="data/clapnq/question_train_answerable.tsv")
    parser.add_argument("--dev_questions_path", type=str, default="data/clapnq/question_dev_answerable.tsv")
    parser.add_argument("--passages_path", type=str, default="data/clapnq/passages.tsv")
    parser.add_argument("--index_path", type=str, default="indexes/clapnq_stage2.faiss")
    parser.add_argument("--exp2_summary_path", type=str, default="results/exp2/summary.json")
    parser.add_argument("--output_dir", type=str, default="results/exp3")
    parser.add_argument("--adapter_output_dir", type=str, default="outputs/generator_lora")

    parser.add_argument("--retriever_base_model", type=str, default="models/bert-base-uncased")
    parser.add_argument("--retriever_adapter_path", type=str, default="outputs/stage2")
    parser.add_argument("--reranker_model", type=str, default="models/bge-reranker-base")
    parser.add_argument("--generator_model", type=str, default="models/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--reranker_mode", type=str, default="cross_encoder", choices=["cross_encoder", "embedding", "none"])

    parser.add_argument("--top_n", type=int, default=30)
    parser.add_argument("--best_k", type=int, default=0)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_eval_samples", type=int, default=0)
    parser.add_argument("--max_query_length", type=int, default=64)
    parser.add_argument("--max_passage_length", type=int, default=256)
    parser.add_argument("--retriever_batch_size", type=int, default=64)
    parser.add_argument("--reranker_batch_size", type=int, default=32)
    parser.add_argument("--reranker_max_length", type=int, default=512)

    parser.add_argument("--max_input_length", type=int, default=1536)
    parser.add_argument("--max_output_length", type=int, default=256)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--generation_batch_size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)

    parser.add_argument("--num_train_epochs", type=float, default=2.0)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--force_retrieve", action="store_true")
    parser.add_argument("--force_rerank", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_model_path(path_or_name: str) -> str:
    return path_or_name if os.path.exists(path_or_name) else path_or_name


def resolve_best_k(args) -> int:
    if args.best_k > 0:
        return args.best_k
    summary_path = Path(args.exp2_summary_path)
    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)
        if int(summary.get("best_k", 0)) > 0:
            return int(summary["best_k"])
    return 5


def context_args(args, split_name: str) -> SimpleNamespace:
    return SimpleNamespace(
        output_dir=str(Path(args.output_dir) / f"{split_name}_contexts"),
        index_path=args.index_path,
        retriever_base_model=args.retriever_base_model,
        retriever_adapter_path=args.retriever_adapter_path,
        reranker_model=args.reranker_model,
        reranker_mode=args.reranker_mode,
        top_n=args.top_n,
        max_query_length=args.max_query_length,
        max_passage_length=args.max_passage_length,
        retriever_batch_size=args.retriever_batch_size,
        reranker_batch_size=args.reranker_batch_size,
        reranker_max_length=args.reranker_max_length,
        force_retrieve=args.force_retrieve,
        force_rerank=args.force_rerank,
    )


def prepare_contexts(args, examples: List[QAExample], passages: List[Passage], split_name: str, device: torch.device) -> List[Dict]:
    split_args = context_args(args, split_name)
    Path(split_args.output_dir).mkdir(parents=True, exist_ok=True)
    retrieval_rows = run_retrieval(split_args, examples, passages, device)
    return run_reranking(split_args, retrieval_rows, passages, device)


def load_tokenizer(model_name_or_path: str):
    tokenizer = AutoTokenizer.from_pretrained(resolve_model_path(model_name_or_path), trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def chat_prompt(tokenizer, prompt: str) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    return prompt


def chat_prompt_with_answer(tokenizer, prompt: str, answer: str) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]
        return tokenizer.apply_chat_template(messages, tokenize=False)
    return f"{prompt}\n{answer}{tokenizer.eos_token or ''}"


def encode_sft_example(tokenizer, prompt: str, answer: str, max_input_length: int, max_output_length: int) -> Dict[str, List[int]]:
    prompt_text = chat_prompt(tokenizer, prompt)
    full_text = chat_prompt_with_answer(tokenizer, prompt, answer)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

    max_length = max_input_length + max_output_length
    if len(full_ids) > max_length:
        # Keep the answer and the tail of the prompt, where the latest passages usually live.
        full_ids = full_ids[-max_length:]
        prompt_cut = max(0, len(prompt_ids) - (max_length - max_output_length))
        prompt_len = max(0, len(prompt_ids) - prompt_cut)
    else:
        prompt_len = len(prompt_ids)

    labels = list(full_ids)
    labels[:min(prompt_len, len(labels))] = [IGNORE_INDEX] * min(prompt_len, len(labels))
    if all(label == IGNORE_INDEX for label in labels):
        labels[-1] = full_ids[-1]
    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}


def select_passages(row: Dict, passages: List[Passage], k: int) -> List[Passage]:
    return [passages[hit["row"]] for hit in row["hits"][:k]]


def build_training_rows(
    examples: List[QAExample],
    context_rows: List[Dict],
    passages: List[Passage],
    tokenizer,
    best_k: int,
    max_input_length: int,
    max_output_length: int,
) -> List[Dict]:
    rows = []
    for example, context_row in zip(examples, context_rows):
        if not example.answers:
            continue
        prompt = build_rag_prompt(example.question, select_passages(context_row, passages, best_k))
        rows.append(encode_sft_example(tokenizer, prompt, example.answers[0], max_input_length, max_output_length))
    return rows


class SFTDataset(Dataset):
    def __init__(self, rows: List[Dict]):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


class SFTCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        max_len = max(len(feature["input_ids"]) for feature in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            batch["input_ids"].append([self.tokenizer.pad_token_id] * pad_len + feature["input_ids"])
            batch["attention_mask"].append([0] * pad_len + feature["attention_mask"])
            batch["labels"].append([IGNORE_INDEX] * pad_len + feature["labels"])
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def load_base_generator(model_name_or_path: str):
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    return AutoModelForCausalLM.from_pretrained(
        resolve_model_path(model_name_or_path),
        torch_dtype=dtype,
        trust_remote_code=True,
    )


def train_generator(args, train_rows: List[Dict], tokenizer) -> str:
    model = load_base_generator(args.generator_model)
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    save_strategy = "steps" if args.save_steps > 0 else "epoch"
    training_args = TrainingArguments(
        output_dir=args.adapter_output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        bf16=torch.cuda.is_available(),
        fp16=False,
        logging_steps=args.logging_steps,
        save_strategy=save_strategy,
        save_steps=args.save_steps if args.save_steps > 0 else 500,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        optim="adamw_torch",
        gradient_checkpointing=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=SFTDataset(train_rows),
        data_collator=SFTCollator(tokenizer),
    )
    trainer.train()
    trainer.save_model(args.adapter_output_dir)
    tokenizer.save_pretrained(args.adapter_output_dir)
    return args.adapter_output_dir


class Generator:
    def __init__(
        self,
        model_name_or_path: str,
        adapter_path: Optional[str],
        max_input_length: int,
        max_new_tokens: int,
        temperature: float,
    ):
        self.tokenizer = load_tokenizer(model_name_or_path)
        self.max_input_length = max_input_length
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        base_model = load_base_generator(model_name_or_path)
        if adapter_path:
            base_model = PeftModel.from_pretrained(base_model, adapter_path)
        self.model = base_model
        self.model.eval()
        if torch.cuda.is_available():
            self.model.to("cuda")

    @torch.no_grad()
    def generate_batch(self, prompts: List[str]) -> List[str]:
        chat_prompts = [chat_prompt(self.tokenizer, prompt) for prompt in prompts]
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
        input_len = enc["input_ids"].shape[1]
        return [self.tokenizer.decode(seq[input_len:], skip_special_tokens=True).strip() for seq in output_ids]


def id_to_passage(passages: List[Passage]) -> Dict[str, Passage]:
    return {passage.id: passage for passage in passages}


def build_gold_prompts(examples: List[QAExample], passages: List[Passage]) -> List[str]:
    passage_by_id = id_to_passage(passages)
    prompts = []
    for example in examples:
        gold_passages = [passage_by_id[doc_id] for doc_id in example.gold_doc_ids if doc_id in passage_by_id]
        prompts.append(build_rag_prompt(example.question, gold_passages))
    return prompts


def build_rag_prompts(context_rows: List[Dict], passages: List[Passage], best_k: int) -> List[str]:
    return [build_rag_prompt(row["question"], select_passages(row, passages, best_k)) for row in context_rows]


def evaluate_group(args, generator: Generator, examples: List[QAExample], prompts: List[str], group_name: str) -> Dict:
    output_path = Path(args.output_dir) / f"predictions_{group_name}.jsonl"
    metric_rows = []
    pred_rows = []
    for start in tqdm(range(0, len(prompts), args.generation_batch_size), desc=f"generating {group_name}"):
        batch_prompts = prompts[start:start + args.generation_batch_size]
        batch_examples = examples[start:start + args.generation_batch_size]
        predictions = generator.generate_batch(batch_prompts)
        for example, prompt, prediction in zip(batch_examples, batch_prompts, predictions):
            scores = max_generation_scores(prediction, example.answers)
            metric_rows.append(scores)
            pred_rows.append(
                {
                    "id": example.id,
                    "question": example.question,
                    "prediction": prediction,
                    "answers": example.answers,
                    "scores": scores,
                    "prompt": prompt,
                }
            )
    save_jsonl(pred_rows, str(output_path))
    metrics = mean_dicts(metric_rows)
    metrics["num_eval_examples"] = float(len(metric_rows))
    return metrics


def evaluate_generator_pairs(args, examples: List[QAExample], passages: List[Passage], dev_contexts: List[Dict], best_k: int) -> List[Dict]:
    gold_prompts = build_gold_prompts(examples, passages)
    rag_prompts = build_rag_prompts(dev_contexts, passages, best_k)
    groups = [
        ("原始 Generator，无 RAG，gold passage", "raw_gold", None, gold_prompts, "-"),
        ("微调 Generator，无 RAG，gold passage", "lora_gold", args.adapter_output_dir, gold_prompts, "-"),
        ("原始 Generator，RAG with Reranker", "raw_rag_reranker", None, rag_prompts, best_k),
        ("微调 Generator，RAG with Reranker", "lora_rag_reranker", args.adapter_output_dir, rag_prompts, best_k),
    ]

    table = []
    for method, group_name, adapter_path, prompts, k_value in groups:
        generator = Generator(
            model_name_or_path=args.generator_model,
            adapter_path=adapter_path,
            max_input_length=args.max_input_length,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        metrics = evaluate_group(args, generator, examples, prompts, group_name)
        table.append(
            {
                "方法": method,
                "k": k_value,
                "EM": metrics.get("em", 0.0),
                "F1": metrics.get("f1", 0.0),
                "ROUGE-L": metrics.get("rouge_l", 0.0),
            }
        )
        del generator
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return table


def load_predictions(path: Path) -> List[Dict]:
    return load_jsonl(str(path)) if path.exists() else []


def summarize_examples(output_dir: Path) -> List[Dict]:
    raw = load_predictions(output_dir / "predictions_raw_rag_reranker.jsonl")
    lora = load_predictions(output_dir / "predictions_lora_rag_reranker.jsonl")
    if not raw or not lora:
        return []
    rows = []
    for raw_row, lora_row in zip(raw, lora):
        delta = lora_row["scores"]["f1"] - raw_row["scores"]["f1"]
        rows.append({"delta_f1": delta, "raw": raw_row, "lora": lora_row})
    rows.sort(key=lambda row: row["delta_f1"], reverse=True)
    selected = []
    for label, candidates in [
        ("improved", rows[:1]),
        ("regressed", list(reversed(rows[-1:]))),
        ("still_hard", sorted(rows, key=lambda row: row["lora"]["scores"]["f1"])[:1]),
    ]:
        if not candidates:
            continue
        row = candidates[0]
        selected.append(
            {
                "type": label,
                "question": row["raw"]["question"],
                "reference": row["raw"]["answers"][0] if row["raw"]["answers"] else "",
                "raw_prediction": row["raw"]["prediction"],
                "lora_prediction": row["lora"]["prediction"],
                "raw_f1": row["raw"]["scores"]["f1"],
                "lora_f1": row["lora"]["scores"]["f1"],
                "delta_f1": row["delta_f1"],
            }
        )
    return selected


def write_markdown_report(args, table: List[Dict], best_k: int, examples_summary: List[Dict]) -> str:
    report_path = Path(args.output_dir) / "experiment3_report.md"
    lines = [
        "# 实验三：Generator 微调与端到端 RAG 效果评估",
        "",
        "## 设置",
        "",
        f"- Generator base: `{args.generator_model}`",
        f"- Generator LoRA adapter: `{args.adapter_output_dir}`",
        f"- Retriever: `{args.retriever_base_model}` + `{args.retriever_adapter_path}`",
        f"- Reranker: `{args.reranker_model}`",
        f"- Top-N: `{args.top_n}`",
        f"- k*: `{best_k}`",
        f"- Train samples: `{args.max_train_samples if args.max_train_samples > 0 else 'all'}`",
        f"- Dev samples: `{args.max_eval_samples if args.max_eval_samples > 0 else 'all'}`",
        f"- Epochs: `{args.num_train_epochs}`",
        f"- Learning rate: `{args.learning_rate}`",
        f"- LoRA r/alpha/dropout: `{args.lora_r}/{args.lora_alpha}/{args.lora_dropout}`",
        "",
        "## 结果表",
        "",
        format_metrics_table(table, ["方法", "k", "EM", "F1", "ROUGE-L"]),
        "",
        "## 简要分析",
        "",
    ]
    by_method = {row["方法"]: row for row in table}
    raw_gold = by_method.get("原始 Generator，无 RAG，gold passage", {})
    lora_gold = by_method.get("微调 Generator，无 RAG，gold passage", {})
    raw_rag = by_method.get("原始 Generator，RAG with Reranker", {})
    lora_rag = by_method.get("微调 Generator，RAG with Reranker", {})
    if raw_gold and lora_gold:
        lines.append(
            f"在 gold passage 条件下，微调后 F1 从 {raw_gold.get('F1', 0.0):.4f} 变为 {lora_gold.get('F1', 0.0):.4f}，ROUGE-L 从 {raw_gold.get('ROUGE-L', 0.0):.4f} 变为 {lora_gold.get('ROUGE-L', 0.0):.4f}。"
        )
    if raw_rag and lora_rag:
        lines.append(
            f"在完整 RAG 条件下，微调后 F1 从 {raw_rag.get('F1', 0.0):.4f} 变为 {lora_rag.get('F1', 0.0):.4f}，ROUGE-L 从 {raw_rag.get('ROUGE-L', 0.0):.4f} 变为 {lora_rag.get('ROUGE-L', 0.0):.4f}。"
        )
    lines.extend(
        [
            "",
            "训练样本使用 train question 通过同一 RAG 数据库检索、reranker 重排后的 passages 构造，没有直接把 train gold passage 注入训练输入；dev 的 RAG 评估同样只使用检索结果，gold passage 只用于无检索上限/读文能力对比。",
        ]
    )
    if examples_summary:
        lines.extend(["", "## 样例分析", ""])
        for item in examples_summary:
            lines.extend(
                [
                    f"### {item['type']}",
                    "",
                    f"- Question: {item['question']}",
                    f"- Reference: {item['reference']}",
                    f"- Raw prediction: {item['raw_prediction']}",
                    f"- LoRA prediction: {item['lora_prediction']}",
                    f"- F1 change: {item['raw_f1']:.4f} -> {item['lora_f1']:.4f} (delta {item['delta_f1']:.4f})",
                    "",
                ]
            )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return str(report_path)


def main():
    args = parse_args()
    set_seed(args.seed)
    print_gpu_info()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    best_k = resolve_best_k(args)

    device = get_device()
    passages = read_passages(args.passages_path)
    tokenizer = load_tokenizer(args.generator_model)

    train_examples = read_questions(
        args.train_questions_path,
        max_samples=args.max_train_samples if args.max_train_samples > 0 else None,
    )
    dev_examples = read_questions(
        args.dev_questions_path,
        max_samples=args.max_eval_samples if args.max_eval_samples > 0 else None,
    )

    if not args.skip_train:
        train_contexts = prepare_contexts(args, train_examples, passages, "train", device)
        train_rows = build_training_rows(
            train_examples,
            train_contexts,
            passages,
            tokenizer,
            best_k,
            args.max_input_length,
            args.max_output_length,
        )
        save_json({"num_train_rows": len(train_rows), "best_k": best_k}, str(Path(args.output_dir) / "train_summary.json"))
        train_generator(args, train_rows, tokenizer)

    table: List[Dict] = []
    examples_summary: List[Dict] = []
    if not args.skip_eval:
        if not Path(args.adapter_output_dir, "adapter_config.json").exists():
            raise FileNotFoundError(f"Missing LoRA adapter at {args.adapter_output_dir}. Run without --skip_train first.")
        dev_contexts = prepare_contexts(args, dev_examples, passages, "dev", device)
        table = evaluate_generator_pairs(args, dev_examples, passages, dev_contexts, best_k)
        save_json(table, str(Path(args.output_dir) / "table_generator_finetune_comparison.json"))
        examples_summary = summarize_examples(Path(args.output_dir))
        save_json(examples_summary, str(Path(args.output_dir) / "example_analysis.json"))

    summary = {"best_k": best_k, "table": table, "examples": examples_summary}
    save_json(summary, str(Path(args.output_dir) / "summary.json"))
    report_path = write_markdown_report(args, table, best_k, examples_summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[save] report: {report_path}")


if __name__ == "__main__":
    main()
