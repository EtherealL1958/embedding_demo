import json
import gzip
import random
from typing import Any, Dict, List, Optional

from torch.utils.data import Dataset


def load_json_or_jsonl(path: str):
    open_fn = gzip.open if path.endswith(".gz") else open
    with open_fn(path, "rt", encoding="utf-8") as f:
        if path.endswith(".json") or path.endswith(".json.gz"):
            return json.load(f)
        return [json.loads(line) for line in f if line.strip()]


def passage_to_text(passage: Optional[Dict[str, Any]]) -> str:
    if not passage:
        return ""
    title = passage.get("title", "")
    text = passage.get("text", "")
    if title:
        return title.strip() + "\n" + text.strip()
    return text.strip()


class DPRTrainDataset(Dataset):
    """
    DPR-NQ train 数据格式：
    {
      "question": "...",
      "positive_ctxs": [{"title": "...", "text": "..."}],
      "negative_ctxs": [{"title": "...", "text": "..."}],
      "hard_negative_ctxs": [{"title": "...", "text": "..."}]
    }

    stage=1：优先使用 negative_ctxs
    stage=2：优先使用 hard_negative_ctxs，没有时退化为 negative_ctxs
    """

    def __init__(
        self,
        path: str,
        stage: int = 1,
        hard_neg_path: Optional[str] = None,
        max_samples: Optional[int] = None,
    ):
        self.data = load_json_or_jsonl(path)
        self.stage = stage
        self.hard_neg_data = load_json_or_jsonl(hard_neg_path) if hard_neg_path else None

        if max_samples is not None and max_samples > 0:
            self.data = self.data[:max_samples]
            if self.hard_neg_data is not None:
                self.hard_neg_data = self.hard_neg_data[:max_samples]

    def __len__(self):
        return len(self.data)

    def _external_hard_negs(self, idx: int) -> List[Dict[str, Any]]:
        if self.hard_neg_data is None:
            return []

        item = self.hard_neg_data[idx]
        for key in ["hard_negative_ctxs", "hard_negatives", "ctxs"]:
            value = item.get(key)
            if isinstance(value, list):
                return value
        return []

    def __getitem__(self, idx: int):
        item = self.data[idx]
        question = item.get("question", "").strip()
        positive_ctxs = item.get("positive_ctxs", []) or []

        if not question or not positive_ctxs:
            return None

        positive = random.choice(positive_ctxs)

        normal_negs = item.get("negative_ctxs", []) or []
        hard_negs = item.get("hard_negative_ctxs", []) or []
        hard_negs = self._external_hard_negs(idx) + hard_negs

        # TODO: 根据训练阶段选择负例。
        # stage=1 使用普通负例 normal_negs；
        # stage=2 优先使用困难负例 hard_negs，没有时退回 normal_negs。
        raise NotImplementedError("TODO: choose neg_pool for stage1/stage2")

        if not neg_pool:
            return None

        negative = random.choice(neg_pool)

        return {
            "question": question,
            "positive_passage": positive,
            "negative_passage": negative,
        }


def make_train_collate_fn(tokenizer, max_query_length: int, max_passage_length: int):
    def collate_fn(batch):
        batch = [
            x for x in batch
            if x is not None and x.get("positive_passage") and x.get("negative_passage")
        ]

        if len(batch) == 0:
            return None

        questions = [x["question"] for x in batch]
        positives = [passage_to_text(x["positive_passage"]) for x in batch]
        negatives = [passage_to_text(x["negative_passage"]) for x in batch]

        q_enc = tokenizer(
            questions,
            padding=True,
            truncation=True,
            max_length=max_query_length,
            return_tensors="pt",
        )

        pos_enc = tokenizer(
            positives,
            padding=True,
            truncation=True,
            max_length=max_passage_length,
            return_tensors="pt",
        )

        neg_enc = tokenizer(
            negatives,
            padding=True,
            truncation=True,
            max_length=max_passage_length,
            return_tensors="pt",
        )

        return {
            "query": q_enc,
            "positive": pos_enc,
            "negative": neg_enc,
        }

    return collate_fn


class DPREvalDataset(Dataset):
    """
    课程基础版评估：
    candidate passages = positive_ctxs + negative_ctxs + hard_negative_ctxs
    然后在候选集合内部排序，计算 MRR@10 / Recall@k。
    """

    def __init__(self, path: str, max_samples: Optional[int] = None):
        self.data = load_json_or_jsonl(path)
        if max_samples is not None and max_samples > 0:
            self.data = self.data[:max_samples]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        item = self.data[idx]
        question = item.get("question", "").strip()
        positive_ctxs = item.get("positive_ctxs", []) or []

        if not question or not positive_ctxs:
            return None

        candidates = item.get("candidate_passages")
        if candidates:
            candidate_texts = [passage_to_text(p) for p in candidates]
            if "positive_indices" in item:
                positive_indices = item["positive_indices"]
            else:
                positive_texts = set(passage_to_text(p) for p in positive_ctxs)
                positive_indices = [
                    i for i, text in enumerate(candidate_texts)
                    if text in positive_texts
                ]
        else:
            negative_ctxs = item.get("negative_ctxs", []) or []
            hard_negative_ctxs = item.get("hard_negative_ctxs", []) or []
            candidates = positive_ctxs + negative_ctxs + hard_negative_ctxs
            candidate_texts = [passage_to_text(p) for p in candidates]
            positive_indices = list(range(len(positive_ctxs)))

        candidate_texts = [x for x in candidate_texts if x]
        positive_indices = [
            i for i in positive_indices
            if isinstance(i, int) and 0 <= i < len(candidate_texts)
        ]

        if not candidate_texts or not positive_indices:
            return None

        return {
            "question": question,
            "candidate_passages": candidate_texts,
            "positive_indices": positive_indices,
        }


def eval_collate_fn(batch):
    batch = [x for x in batch if x is not None]
    if len(batch) == 0:
        return None
    return batch
