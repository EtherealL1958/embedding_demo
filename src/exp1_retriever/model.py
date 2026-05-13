from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

try:
    from peft import LoraConfig, get_peft_model, PeftModel
except Exception:
    LoraConfig = None
    get_peft_model = None
    PeftModel = None


def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    使用 attention_mask 排除 padding token，
    对有效 token 的 hidden states 做 mean pooling。

    输入：
        last_hidden_state: [B, L, H]
        attention_mask:    [B, L]

    输出：
        pooled embedding:  [B, H]
    """
    mask = attention_mask.unsqueeze(-1).to(dtype=last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


class DPRBiEncoder(nn.Module):
    """
    简化版 bi-encoder：
    query 和 passage 共享同一个 BERT encoder。
    """

    def __init__(self, encoder: nn.Module, normalize: bool = True):
        super().__init__()
        self.encoder = encoder
        self.normalize = normalize

    def encode(self, batch):
        inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
        }

        if "token_type_ids" in batch:
            inputs["token_type_ids"] = batch["token_type_ids"]

        outputs = self.encoder(**inputs)

        pooled = mean_pooling(
            outputs.last_hidden_state,
            batch["attention_mask"],
        )

        if self.normalize:
            pooled = F.normalize(pooled, p=2, dim=-1)

        return pooled

    def forward(self, query_batch, pos_batch, neg_batch=None):
        q_emb = self.encode(query_batch)
        pos_emb = self.encode(pos_batch)
        neg_emb = self.encode(neg_batch) if neg_batch is not None else None
        return q_emb, pos_emb, neg_emb


def apply_lora_to_encoder(
    encoder: nn.Module,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules: str,
) -> nn.Module:
    """
    给 BERT encoder 添加 LoRA adapter。

    要求：
    1. 将 lora_target_modules 从字符串转成 list；
    2. 构造 LoraConfig；
    3. 调用 get_peft_model 包装 encoder；
    4. 返回带 LoRA 的 encoder。
    """

    if LoraConfig is None or get_peft_model is None:
        raise ImportError("peft 没有安装，无法使用 LoRA。")

    target_modules = [
        module.strip()
        for module in lora_target_modules.split(",")
        if module.strip()
    ]
    if not target_modules:
        raise ValueError("lora_target_modules 不能为空。")

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
    )

    encoder = get_peft_model(encoder, lora_config)

    return encoder


def build_retriever_model(
    base_model_name_or_path: str,
    use_lora: bool = False,
    adapter_path: Optional[str] = None,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: str = "query,value",
    trainable_adapter: bool = False,
) -> Tuple[DPRBiEncoder, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path, use_fast=True)
    encoder = AutoModel.from_pretrained(base_model_name_or_path)

    if adapter_path:
        if PeftModel is None:
            raise ImportError("peft 没有安装，无法加载 LoRA adapter。")

        encoder = PeftModel.from_pretrained(
            encoder,
            adapter_path,
            is_trainable=trainable_adapter,
        )

    elif use_lora:
        encoder = apply_lora_to_encoder(
            encoder=encoder,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_modules=lora_target_modules,
        )

    model = DPRBiEncoder(encoder=encoder, normalize=True)
    return model, tokenizer


@torch.no_grad()
def encode_texts(
    model: DPRBiEncoder,
    tokenizer,
    texts: List[str],
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    all_embs = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]

        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        enc = {k: v.to(device) for k, v in enc.items()}

        emb = model.encode(enc)
        all_embs.append(emb.detach().cpu())

    return torch.cat(all_embs, dim=0)


def count_trainable_parameters(model: nn.Module):
    trainable = 0
    total = 0

    for p in model.parameters():
        numel = p.numel()
        total += numel

        if p.requires_grad:
            trainable += numel

    return trainable, total
