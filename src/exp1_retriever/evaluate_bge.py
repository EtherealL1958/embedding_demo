import argparse
import os

import numpy as np
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import DPREvalDataset, eval_collate_fn
from .metrics import aggregate_metrics, metrics_to_markdown, ranking_metrics_from_scores
from .utils import ensure_dir, save_json, save_metrics_csv, set_seed


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--eval_path", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)

    parser.add_argument("--model_name", type=str, default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--max_eval_samples", type=int, default=3000)
    parser.add_argument("--max_query_length", type=int, default=64)
    parser.add_argument("--max_passage_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=8)

    parser.add_argument(
        "--query_instruction",
        type=str,
        default="Represent this sentence for searching relevant passages: ",
    )
    parser.add_argument("--no_query_instruction", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    print(f"[model] loading {args.model_name}")
    model = SentenceTransformer(args.model_name)

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

    all_rows = []
    valid_count = 0

    for batch in tqdm(loader, desc="evaluating bge"):
        if batch is None:
            continue

        questions = [item["question"] for item in batch]
        if not args.no_query_instruction:
            questions_for_encode = [args.query_instruction + q for q in questions]
        else:
            questions_for_encode = questions

        q_embs = model.encode(
            questions_for_encode,
            batch_size=args.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        for item, q_emb in zip(batch, q_embs):
            passages = item["candidate_passages"]
            p_embs = model.encode(
                passages,
                batch_size=args.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

            scores = np.matmul(p_embs, q_emb)
            row = ranking_metrics_from_scores(
                scores=scores,
                positive_indices=item["positive_indices"],
                recall_ks=(1, 5, 10, 20, 30, 50),
                mrr_k=10,
            )
            all_rows.append(row)
            valid_count += 1

    metrics = aggregate_metrics(all_rows)
    metrics["num_eval_examples"] = float(valid_count)

    ensure_dir(os.path.dirname(args.output_json) or ".")
    save_json(metrics, args.output_json)
    save_metrics_csv(metrics, args.output_csv)

    print("\n[result]")
    print(metrics_to_markdown(metrics))
    print(f"\n[save] {args.output_json}")
    print(f"[save] {args.output_csv}")


if __name__ == "__main__":
    main()
