import argparse
import os
import time

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from .data import DPRTrainDataset, make_train_collate_fn
from .loss import ContrastiveRetrievalLoss
from .model import build_retriever_model, count_trainable_parameters
from .utils import (
    amp_dtype,
    ensure_dir,
    get_device,
    move_to_device,
    print_gpu_info,
    save_json,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--base_model", type=str, default="google-bert/bert-base-uncased")
    parser.add_argument("--resume_adapter", type=str, default=None)

    parser.add_argument("--stage", type=int, default=1, choices=[1, 2])
    parser.add_argument("--hard_neg_path", type=str, default=None)

    parser.add_argument("--max_train_samples", type=int, default=50000)
    parser.add_argument("--max_query_length", type=int, default=64)
    parser.add_argument("--max_passage_length", type=int, default=256)

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.05)

    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str, default="query,value")

    parser.add_argument("--precision", type=str, default="fp16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_steps", type=int, default=20)

    return parser.parse_args()


def build_dataloader(args, tokenizer):
    dataset = DPRTrainDataset(
        path=args.train_path,
        stage=args.stage,
        hard_neg_path=args.hard_neg_path,
        max_samples=args.max_train_samples if args.max_train_samples > 0 else None,
    )

    collate_fn = make_train_collate_fn(
        tokenizer=tokenizer,
        max_query_length=args.max_query_length,
        max_passage_length=args.max_passage_length,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    return dataset, loader


def build_optimizer_and_scheduler(model, loader, args):
    """
    TODO:
    构建 AdamW optimizer 和 linear warmup scheduler。

    要求：
    1. 只优化 requires_grad=True 的参数；
    2. 根据 grad_accum_steps 计算 optimizer step 数；
    3. 根据 warmup_ratio 计算 warmup_steps；
    4. 返回 optimizer, scheduler。
    """

    raise NotImplementedError(
        "TODO: build optimizer and scheduler in build_optimizer_and_scheduler()."
    )


def compute_train_loss(model, batch, loss_fn, args, device, use_amp, dtype):
    """
    TODO:
    完成一次 forward，并返回用于 backward 的 loss。

    要求：
    1. 将 batch 移动到 device；
    2. 在 autocast 下计算 query / positive / negative embeddings；
    3. 调用 loss_fn 计算 contrastive loss；
    4. 按 grad_accum_steps 缩放 loss；
    5. 返回 loss。
    """

    raise NotImplementedError(
        "TODO: compute forward loss in compute_train_loss()."
    )


def optimizer_update(model, optimizer, scheduler, scaler, args):
    """
    TODO:
    完成一次 optimizer 更新。

    要求：
    1. unscale 梯度；
    2. 做 gradient clipping；
    3. optimizer.step；
    4. scaler.update；
    5. scheduler.step；
    6. 清空梯度。
    """

    raise NotImplementedError(
        "TODO: implement optimizer update in optimizer_update()."
    )


def train_loop(model, loader, loss_fn, optimizer, scheduler, scaler, args, device):
    use_amp = torch.cuda.is_available() and args.precision in ["fp16", "bf16"]
    dtype = amp_dtype(args.precision)

    global_step = 0
    running_loss = 0.0

    optimizer.zero_grad(set_to_none=True)

    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}")

        for step, batch in enumerate(pbar, start=1):
            if batch is None:
                continue

            loss = compute_train_loss(
                model=model,
                batch=batch,
                loss_fn=loss_fn,
                args=args,
                device=device,
                use_amp=use_amp,
                dtype=dtype,
            )

            scaler.scale(loss).backward()

            if step % args.grad_accum_steps == 0:
                optimizer_update(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    args=args,
                )

                global_step += 1
                running_loss += loss.item() * args.grad_accum_steps

                if global_step % args.log_steps == 0:
                    avg_loss = running_loss / args.log_steps
                    lr = scheduler.get_last_lr()[0]

                    pbar.set_postfix(
                        {
                            "loss": f"{avg_loss:.4f}",
                            "lr": f"{lr:.2e}",
                            "step": global_step,
                        }
                    )

                    running_loss = 0.0

    return global_step


def save_training_outputs(model, tokenizer, args, elapsed_seconds):
    print(f"[save] saving adapter to {args.output_dir}")

    model.encoder.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    save_json(
        {
            "base_model": args.base_model,
            "resume_adapter": args.resume_adapter,
            "train_path": args.train_path,
            "stage": args.stage,
            "max_train_samples": args.max_train_samples,
            "max_query_length": args.max_query_length,
            "max_passage_length": args.max_passage_length,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "lr": args.lr,
            "temperature": args.temperature,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "elapsed_seconds": elapsed_seconds,
        },
        os.path.join(args.output_dir, "training_args.json"),
    )


def main():
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.output_dir)

    print_gpu_info()
    device = get_device()

    model, tokenizer = build_retriever_model(
        base_model_name_or_path=args.base_model,
        use_lora=(args.resume_adapter is None),
        adapter_path=args.resume_adapter,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        trainable_adapter=True,
    )

    trainable, total = count_trainable_parameters(model)
    print(
        f"[model] trainable params: {trainable:,} / total params: {total:,} "
        f"({100 * trainable / total:.4f}%)"
    )

    model.to(device)
    model.train()

    dataset, loader = build_dataloader(args, tokenizer)

    print(f"[data] train samples: {len(dataset)}")
    print(f"[train] stage: {args.stage}")
    print(f"[train] output_dir: {args.output_dir}")

    optimizer, scheduler = build_optimizer_and_scheduler(
        model=model,
        loader=loader,
        args=args,
    )

    loss_fn = ContrastiveRetrievalLoss(temperature=args.temperature)

    scaler = torch.cuda.amp.GradScaler(
        enabled=(torch.cuda.is_available() and args.precision == "fp16")
    )

    start_time = time.time()

    global_step = train_loop(
        model=model,
        loader=loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        args=args,
        device=device,
    )

    elapsed = time.time() - start_time

    save_training_outputs(
        model=model,
        tokenizer=tokenizer,
        args=args,
        elapsed_seconds=elapsed,
    )

    print(f"[done] training finished.")


if __name__ == "__main__":
    main()