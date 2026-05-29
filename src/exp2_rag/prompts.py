from typing import List

from .data import Passage


def build_rag_prompt(question: str, passages: List[Passage]) -> str:
    passage_lines = []
    for idx, passage in enumerate(passages, start=1):
        passage_lines.append(f"[{idx}] {passage.prompt_text}")

    return (
        "You are given a question and several passages.\n"
        "Answer the question using only the information in the passages.\n"
        "If the passages do not contain enough information, answer \"I don't know\".\n\n"
        f"Question:\n{question}\n\n"
        "Passages:\n"
        + "\n".join(passage_lines)
        + "\n\nAnswer:"
    )


def build_no_rag_prompt(question: str) -> str:
    return (
        "Answer the following question.\n"
        "If you do not know the answer, answer \"I don't know\".\n\n"
        f"Question:\n{question}\n\n"
        "Answer:"
    )

