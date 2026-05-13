import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import DPREvalDataset, eval_collate_fn
from .metrics import aggregate_metrics, metrics_to_markdown, ranking_metrics_from_scores
from .model import build_retriever_model, encode_texts
from .utils import ensure_dir, get_device, print_gpu_info, save_json, save_metrics_csv, set_seed


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--eval_path", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)

    parser.add_argument("--base_model", type=str, default="google-bert/bert-base-uncased")
    parser.add_argument("--adapter_path", type=str, default=None)

    parser.add_argument("--max_eval_samples", type=int, default=3000)
    parser.add_argument("--max_query_length", type=int, default=64)
    parser.add_argument("--max_passage_length", type=int, default=256)
    parser.add_argument("--query_batch_size", type=int, default=64)
    parser.add_argument("--passage_batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=8)

    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def evaluate_one_batch(model, tokenizer, batch, args, device):
    """
    对一个 eval batch 完成候选集排序评估。

    需要完成：
    1. 编码 batch 中的 questions；
    2. 对每个 question 编码其 candidate_passages；
    3. 计算 query 与 candidate passages 的相似度；
    4. 调用 ranking_metrics_from_scores 计算指标；
    5. 返回 rows 和 valid_count。
    """

    rows = []
    valid_count = 0
    questions = [item["question"] for item in batch]

    q_embs = encode_texts(
        model=model,
        tokenizer=tokenizer,
        texts=questions,
        max_length=args.max_query_length,
        batch_size=args.query_batch_size,
        device=device,
    )

    for item, q_emb in zip(batch, q_embs):
        passages = item["candidate_passages"]
        if not passages:
            continue

        p_embs = encode_texts(
            model=model,
            tokenizer=tokenizer,
            texts=passages,
            max_length=args.max_passage_length,
            batch_size=args.passage_batch_size,
            device=device,
        )

        scores = torch.matmul(p_embs, q_emb).numpy()
        row = ranking_metrics_from_scores(
            scores=scores,
            positive_indices=item["positive_indices"],
            recall_ks=(1, 5, 10, 20, 30, 50),
            mrr_k=10,
        )
        rows.append(row)
        valid_count += 1

    return rows, valid_count


def run_evaluation(model, tokenizer, loader, args, device):
    all_rows = []
    valid_count = 0

    for batch in tqdm(loader, desc="evaluating"):
        if batch is None:
            continue

        rows, count = evaluate_one_batch(
            model=model,
            tokenizer=tokenizer,
            batch=batch,
            args=args,
            device=device,
        )

        all_rows.extend(rows)
        valid_count += count

    metrics = aggregate_metrics(all_rows)
    metrics["num_eval_examples"] = float(valid_count)
    return metrics


def main():
    args = parse_args()
    set_seed(args.seed)
    print_gpu_info()

    device = get_device()

    dataset = DPREvalDataset(
        path=args.eval_path,
        max_samples=args.max_eval_samples if args.max_eval_samples > 0 else None,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=eval_collate_fn,
    )

    model, tokenizer = build_retriever_model(
        base_model_name_or_path=args.base_model,
        use_lora=False,
        adapter_path=args.adapter_path,
        trainable_adapter=False,
    )

    model.to(device)
    model.eval()

    metrics = run_evaluation(
        model=model,
        tokenizer=tokenizer,
        loader=loader,
        args=args,
        device=device,
    )

    ensure_dir(os.path.dirname(args.output_json) or ".")
    save_json(metrics, args.output_json)
    save_metrics_csv(metrics, args.output_csv)

    print("\n[result]")
    print(metrics_to_markdown(metrics))
    print(f"\n[save] {args.output_json}")
    print(f"[save] {args.output_csv}")


if __name__ == "__main__":
    main()
