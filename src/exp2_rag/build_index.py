import argparse
from pathlib import Path

import faiss
import numpy as np
import torch
from tqdm import tqdm

from exp1_retriever.model import build_retriever_model, encode_texts
from exp1_retriever.utils import get_device, print_gpu_info, set_seed
from exp2_rag.data import read_passages, save_jsonl


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--passages_path", type=str, default="data/clapnq/passages.tsv")
    parser.add_argument("--base_model", type=str, default="models/bert-base-uncased")
    parser.add_argument("--adapter_path", type=str, default="outputs/stage2")
    parser.add_argument("--index_path", type=str, default="indexes/clapnq_stage2.faiss")
    parser.add_argument("--meta_path", type=str, default="indexes/clapnq_passages.jsonl")
    parser.add_argument("--embeddings_path", type=str, default=None)
    parser.add_argument("--max_passage_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    print_gpu_info()
    device = get_device()

    passages = read_passages(args.passages_path)
    texts = [passage.retriever_text for passage in passages]

    model, tokenizer = build_retriever_model(
        base_model_name_or_path=args.base_model,
        adapter_path=args.adapter_path,
        trainable_adapter=False,
    )
    model.to(device)
    model.eval()

    chunks = []
    for start in tqdm(range(0, len(texts), args.batch_size), desc="encoding passages"):
        batch_texts = texts[start:start + args.batch_size]
        emb = encode_texts(
            model=model,
            tokenizer=tokenizer,
            texts=batch_texts,
            max_length=args.max_passage_length,
            batch_size=args.batch_size,
            device=device,
        )
        chunks.append(emb.numpy().astype("float32"))

    embeddings = np.concatenate(chunks, axis=0)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    Path(args.index_path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, args.index_path)
    if args.embeddings_path:
        np.save(args.embeddings_path, embeddings)

    save_jsonl(
        ({"row": idx, "id": passage.id, "title": passage.title, "text": passage.text} for idx, passage in enumerate(passages)),
        args.meta_path,
    )

    print(f"[save] index: {args.index_path}")
    print(f"[save] meta: {args.meta_path}")
    print(f"[info] passages: {len(passages)}")
    print(f"[info] dim: {embeddings.shape[1]}")


if __name__ == "__main__":
    main()

