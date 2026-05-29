import csv
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


@dataclass
class Passage:
    id: str
    text: str
    title: str = ""

    @property
    def retriever_text(self) -> str:
        return f"{self.title}\n{self.text}" if self.title else self.text

    @property
    def prompt_text(self) -> str:
        return f"{self.title}: {self.text}" if self.title else self.text


@dataclass
class QAExample:
    id: str
    question: str
    gold_doc_ids: List[str]
    answers: List[str]


def read_passages(path: str) -> List[Passage]:
    passages: List[Passage] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            passages.append(
                Passage(
                    id=row["id"],
                    text=row.get("text", ""),
                    title=row.get("title", ""),
                )
            )
    return passages


def read_questions(path: str, max_samples: Optional[int] = None) -> List[QAExample]:
    examples: List[QAExample] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            examples.append(
                QAExample(
                    id=row["id"],
                    question=row["question"],
                    gold_doc_ids=_split_doc_ids(row.get("doc-id-list", "")),
                    answers=_split_answers(row.get("answers", "")),
                )
            )
            if max_samples is not None and len(examples) >= max_samples:
                break
    return examples


def _split_doc_ids(value: str) -> List[str]:
    if not value:
        return []
    pieces = re.split(r"\s*(?:,|;|::)\s*", value.strip())
    return [piece for piece in pieces if piece]


def _split_answers(value: str) -> List[str]:
    if not value:
        return []
    answers = [part.strip().strip('"') for part in value.split("::")]
    return [answer for answer in answers if answer]


def save_jsonl(rows: Iterable[Dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sample_random_passages(
    passages: List[Passage],
    k: int,
    rng: random.Random,
    exclude_ids: Optional[Iterable[str]] = None,
) -> List[int]:
    exclude = set(exclude_ids or [])
    candidate_indices = [idx for idx, passage in enumerate(passages) if passage.id not in exclude]
    if len(candidate_indices) <= k:
        return candidate_indices
    return rng.sample(candidate_indices, k)

