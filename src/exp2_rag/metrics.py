import collections
import math
import re
import string
from typing import Dict, Iterable, List, Sequence

import numpy as np
from rouge_score import rouge_scorer


def normalize_answer(text: str) -> str:
    def remove_articles(s: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", s)

    def white_space_fix(s: str) -> str:
        return " ".join(s.split())

    def remove_punc(s: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in s if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(text.lower())))


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    common = collections.Counter(pred_tokens) & collections.Counter(gold_tokens)
    num_same = sum(common.values())
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return float(pred_tokens == gold_tokens)
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


_ROUGE = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def rouge_l_score(prediction: str, ground_truth: str) -> float:
    if not prediction.strip() or not ground_truth.strip():
        return float(prediction.strip() == ground_truth.strip())
    return float(_ROUGE.score(ground_truth, prediction)["rougeL"].fmeasure)


def max_generation_scores(prediction: str, references: Sequence[str]) -> Dict[str, float]:
    if not references:
        return {"em": 0.0, "f1": 0.0, "rouge_l": 0.0}
    return {
        "em": max(exact_match_score(prediction, ref) for ref in references),
        "f1": max(f1_score(prediction, ref) for ref in references),
        "rouge_l": max(rouge_l_score(prediction, ref) for ref in references),
    }


def retrieval_metrics(
    ranked_doc_ids: Sequence[str],
    gold_doc_ids: Iterable[str],
    recall_k: int,
    mrr_k: int = 10,
    ndcg_k: int = 10,
) -> Dict[str, float]:
    gold = set(gold_doc_ids)
    if not gold:
        return {f"recall@{recall_k}": 0.0, f"mrr@{mrr_k}": 0.0, f"ndcg@{ndcg_k}": 0.0}

    top_recall = ranked_doc_ids[:recall_k]
    hits = sum(1 for doc_id in top_recall if doc_id in gold)
    recall = hits / len(gold)

    rr = 0.0
    for rank, doc_id in enumerate(ranked_doc_ids[:mrr_k], start=1):
        if doc_id in gold:
            rr = 1.0 / rank
            break

    dcg = 0.0
    for rank, doc_id in enumerate(ranked_doc_ids[:ndcg_k], start=1):
        if doc_id in gold:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(gold), ndcg_k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    ndcg = dcg / idcg if idcg > 0 else 0.0

    return {f"recall@{recall_k}": float(recall), f"mrr@{mrr_k}": float(rr), f"ndcg@{ndcg_k}": float(ndcg)}


def mean_dicts(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row})
    return {key: float(np.mean([row.get(key, 0.0) for row in rows])) for key in keys}


def format_metrics_table(rows: List[Dict], columns: List[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)

