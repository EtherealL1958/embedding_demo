import csv
import json
import os
import random
from typing import Any, Dict
from collections.abc import Mapping

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)


def move_to_device(batch: Any, device: torch.device):
    """
    递归把 batch 里的 tensor 移到 GPU。
    重点：transformers tokenizer 返回的是 BatchEncoding，
    它不是普通 dict，但有 .to(device) 方法。
    """
    if batch is None:
        return None

    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)

    # 处理 transformers.BatchEncoding
    if hasattr(batch, "to") and callable(getattr(batch, "to")):
        try:
            return batch.to(device)
        except Exception:
            pass

    if isinstance(batch, Mapping):
        return {k: move_to_device(v, device) for k, v in batch.items()}

    if isinstance(batch, list):
        return [move_to_device(x, device) for x in batch]

    if isinstance(batch, tuple):
        return tuple(move_to_device(x, device) for x in batch)

    return batch


def save_json(obj: Dict, path: str):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_metrics_csv(metrics: Dict[str, float], path: str):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key in sorted(metrics.keys()):
            writer.writerow([key, metrics[key]])


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def amp_dtype(precision: str):
    precision = precision.lower()
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return None


def print_gpu_info():
    if not torch.cuda.is_available():
        print("[device] CUDA 不可用，当前会用 CPU，训练会很慢。")
        return

    print(f"[device] CUDA 可用，GPU 数量: {torch.cuda.device_count()}")
    print(f"[device] 当前 GPU: {torch.cuda.get_device_name(0)}")
    print(f"[device] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '未设置')}")
