from typing import Dict, Iterable, List

import numpy as np


def ranking_metrics_from_scores(
    scores: Iterable[float],
    positive_indices: List[int],
    recall_ks=(1, 5, 10, 20, 30, 50),
    mrr_k: int = 10,
) -> Dict[str, float]:
    """
    TODO:
    根据候选 passage 的 scores 和正样本下标 positive_indices，
    计算 Recall@k 和 MRR@k。

    参数：
        scores:
            每个 candidate passage 的模型得分，分数越高排名越靠前。
        positive_indices:
            正样本 passage 在 candidate passages 中的下标。
        recall_ks:
            需要计算的 Recall@k 列表。
        mrr_k:
            MRR 的截断位置。

    返回：
        {
            "recall@1": ...,
            "recall@5": ...,
            "mrr@10": ...
        }
    """

    scores = np.asarray(list(scores), dtype=np.float64)
    positive_set = set(int(i) for i in positive_indices)

    if len(scores) == 0 or len(positive_set) == 0:
        out = {f"recall@{k}": 0.0 for k in recall_ks}
        out[f"mrr@{mrr_k}"] = 0.0
        return out

    # TODO 1:
    # 按照 score 从大到小排序，得到 ranked_indices。
    # 提示：np.argsort 默认是从小到大。
    ranked_indices = None

    out = {}

    # TODO 2:
    # 计算 Recall@k。
    # Recall@k = Top-k 中命中的正样本数 / 全部正样本数。
    for k in recall_ks:
        out[f"recall@{k}"] = None

    # TODO 3:
    # 计算 MRR@mrr_k。
    # 找到 Top-mrr_k 内第一个正样本的排名 rank，
    # 若存在，则 RR = 1 / rank；否则 RR = 0。
    rr = None

    out[f"mrr@{mrr_k}"] = rr
    return out


def aggregate_metrics(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}

    keys = sorted(rows[0].keys())
    return {
        key: float(np.mean([row.get(key, 0.0) for row in rows]))
        for key in keys
    }


def metrics_to_markdown(metrics: Dict[str, float]) -> str:
    if not metrics:
        return "No metrics."

    preferred_order = [
        "mrr@10",
        "recall@1",
        "recall@5",
        "recall@10",
        "recall@20",
        "recall@30",
        "recall@50",
    ]

    lines = ["| metric | value |", "|---|---:|"]
    used = set()

    for key in preferred_order:
        if key in metrics:
            lines.append(f"| {key} | {metrics[key]:.6f} |")
            used.add(key)

    for key in sorted(metrics.keys()):
        if key not in used:
            lines.append(f"| {key} | {metrics[key]:.6f} |")

    return "\n".join(lines)