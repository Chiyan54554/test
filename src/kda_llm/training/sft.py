"""Supervised fine-tuning for answer-masked conversation tensors."""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import torch

from kda_llm.models import KDAConfig, KDALanguageModel, parameter_count
from kda_llm.models.kernels import LigerFusedLinearCrossEntropyLoss

from .checkpoints import CHECKPOINT_VERSION


def run_sft(args: object) -> None:
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    artifact = torch.load(args.dataset, map_location="cpu", weights_only=True)
    if not isinstance(artifact, dict) or not isinstance(artifact.get("input_ids"), torch.Tensor) or not isinstance(artifact.get("labels"), torch.Tensor):
        raise ValueError("--dataset must be created by kda-prepare-sft")
    input_ids, labels = artifact["input_ids"].long(), artifact["labels"].long()
    if input_ids.ndim != 2 or labels.shape != input_ids.shape or not input_ids.size(0):
        raise ValueError("SFT dataset tensors must be non-empty [examples, sequence] tensors")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("config"), dict) or not isinstance(checkpoint.get("model"), dict):
        raise ValueError("--checkpoint must be a KDA training checkpoint")
    config = KDAConfig(**checkpoint["config"])
    if input_ids.max().item() >= config.vocab_size:
        raise ValueError("SFT dataset tokenizer does not match the base checkpoint vocabulary")
    model: KDALanguageModel | torch.nn.Module = KDALanguageModel(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    if args.compile:
        model = torch.compile(model, dynamic=False)
    use_fused_loss = args.fused_cross_entropy and device.type == "cuda" and LigerFusedLinearCrossEntropyLoss is not None
    if args.fused_cross_entropy and not use_fused_loss:
        raise RuntimeError("fused cross entropy requires Linux CUDA with liger-kernel")
    optimizer_kwargs: dict[str, object] = {"lr": args.lr, "betas": (0.9, 0.95), "weight_decay": args.weight_decay}
    if args.fused_optimizer and device.type == "cuda":
        optimizer_kwargs["fused"] = True
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
    total_steps = args.epochs * math.ceil(input_ids.size(0) / args.batch_size)
    warmup_steps = max(1, round(total_steps * args.warmup_ratio))
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"SFT device: {device}; examples: {input_ids.size(0):,}; steps: {total_steps:,}; parameters: {parameter_count(model):,}")
    step, started_at = 0, perf_counter()
    model.train()
    for epoch in range(args.epochs):
        order = torch.randperm(input_ids.size(0))
        for start in range(0, input_ids.size(0), args.batch_size):
            indices = order[start : start + args.batch_size]
            x, y = input_ids[indices].to(device, non_blocking=True), labels[indices].to(device, non_blocking=True)
            progress = step / max(1, total_steps)
            lr = args.lr * min(1.0, (step + 1) / warmup_steps) * (1.0 - 0.9 * progress)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" and torch.cuda.is_bf16_supported() else nullcontext()
            with autocast:
                _, loss = model(x, y, use_fused_cross_entropy=use_fused_loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            if step % args.log_every == 0 or step == total_steps:
                elapsed = perf_counter() - started_at
                processed = epoch * input_ids.size(0) + min(start + indices.numel(), input_ids.size(0))
                print(f"SFT epoch {epoch + 1}/{args.epochs} | step {step:,}/{total_steps:,} | loss {loss.item():.4f} | lr {lr:.2e} | {processed / elapsed:,.0f} examples/s", flush=True)
        model_to_save = getattr(model, "_orig_mod", model)
        path = output_dir / f"kda-sft-epoch-{epoch + 1}.pt"
        torch.save({"format_version": CHECKPOINT_VERSION, "training_stage": "sft", "model": model_to_save.state_dict(), "optimizer": optimizer.state_dict(), "step": step, "config": asdict(config), "base_checkpoint": args.checkpoint}, path)
        print(f"saved {path}")
