"""Training orchestration independent from command-line parsing."""

from __future__ import annotations

import math
from argparse import ArgumentParser, Namespace
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from kda_llm.models import KDAConfig, KDALanguageModel, parameter_count
from kda_llm.models.kernels import LigerFusedLinearCrossEntropyLoss, chunk_kda

from .checkpoints import restore_checkpoint, save_checkpoint
from .data import BatchPrefetcher, get_batch, load_train_sources, move_batch_to_device
from .profiling import GPUProfiler
from .schedule import format_duration, learning_rate


@torch.inference_mode()
def estimate_loss(model: KDALanguageModel, tokens: np.memmap, batch_size: int, seq_len: int, device: torch.device, steps: int) -> float:
    model.eval()
    losses = []
    for _ in range(steps):
        x, y = move_batch_to_device(*get_batch(tokens, batch_size, seq_len), device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def _resolve_device(args: Namespace, parser: ArgumentParser) -> torch.device:
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available to PyTorch")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    if args.require_kda_kernel and device.type == "cuda" and chunk_kda is None:
        parser.error("chunk_kda is unavailable; install the cuda extra or pass --no-require-kda-kernel")
    if args.fused_cross_entropy and (device.type != "cuda" or LigerFusedLinearCrossEntropyLoss is None):
        parser.error("fused cross entropy requires Linux CUDA and `uv sync --extra cuda`")
    return device


def run_training(args: Namespace, parser: ArgumentParser) -> None:
    if args.train_sources:
        if args.train_data_was_set:
            parser.error("provide exactly one of --train-data or --train-sources")
        args.train_data = None
    elif not args.train_data:
        parser.error("provide --train-data or --train-sources")
    if args.log_every <= 0 or args.profile_steps < 0 or args.profile_warmup_steps < 0:
        parser.error("log and profile step counts must be non-negative, with --log-every above zero")

    model_config = KDAConfig(**args.model_config_values)
    if args.seq_len > model_config.max_seq_len:
        parser.error("seq_len exceeds the model max_seq_len")
    tokens_per_step = args.batch_size * args.grad_accum * args.seq_len
    if args.max_tokens is not None:
        if args.max_tokens <= 0:
            parser.error("--max-tokens must be a positive integer")
        target_tokens = args.max_tokens
    elif args.steps is not None:
        if args.steps <= 0:
            parser.error("--steps must be a positive integer")
        target_tokens = args.steps * tokens_per_step
    else:
        parser.error("provide --max-tokens for training, or --steps for a smoke test")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = _resolve_device(args, parser)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    use_amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    try:
        sources, weights = load_train_sources(args.train_data, args.train_sources)
    except ValueError as error:
        parser.error(str(error))
    val_tokens = np.memmap(args.val_data, dtype=np.uint16, mode="r") if args.val_data else None

    model: KDALanguageModel | torch.nn.Module = KDALanguageModel(model_config).to(device)
    if args.compile:
        model = torch.compile(model, dynamic=False)
    optimizer_kwargs: dict[str, object] = {"lr": args.lr, "betas": (0.9, 0.95), "weight_decay": 0.1}
    use_fused_optimizer = args.fused_optimizer and device.type == "cuda"
    if use_fused_optimizer:
        optimizer_kwargs["fused"] = True
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)

    start_step, tokens_seen, legacy_projection = 0, 0, False
    model_to_manage = getattr(model, "_orig_mod", model)
    if args.resume_from:
        try:
            start_step, tokens_seen, legacy_projection = restore_checkpoint(args.resume_from, model_to_manage, optimizer, model_config, tokens_per_step)
        except ValueError as error:
            parser.error(str(error))
    if tokens_seen >= target_tokens:
        parser.error("checkpoint already reaches the requested token budget")
    remaining_steps = math.ceil((target_tokens - tokens_seen) / tokens_per_step)
    final_step = start_step + remaining_steps
    profile_start = args.profile_start_step if args.profile_start_step is not None else start_step
    profiler = None
    if args.profile_steps:
        if device.type != "cuda" or not start_step <= profile_start < final_step:
            parser.error("GPU profiler start step must be within the remaining CUDA training steps")
        profiler = GPUProfiler(profile_start, args.profile_warmup_steps, args.profile_steps, args.profile_dir)
        if profiler.end_step > final_step:
            parser.error("profiler warmup and active steps exceed the remaining training steps")

    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    backend = "chunk_kda" if device.type == "cuda" and chunk_kda is not None else "reference_recurrence"
    print(f"device: {device}; backend: {backend}; compiled: {args.compile}; fused AdamW: {use_fused_optimizer}; parameters: {parameter_count(model):,}")
    print(f"training budget: {target_tokens:,} tokens; steps: {final_step:,}; tokens/step: {tokens_per_step:,}")
    if start_step:
        print(f"resuming from step {start_step:,} ({tokens_seen:,} tokens)")
    if legacy_projection:
        print("legacy input projections detected; restored model weights and reset optimizer state")
    for (source_path, _), weight in zip(sources, weights, strict=True):
        print(f"training source: {source_path} ({weight:.1%})")

    prefetcher = BatchPrefetcher(sources, weights, args.batch_size, args.seq_len)
    window_started_at, window_tokens = perf_counter(), 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    try:
        for local_step in range(remaining_steps):
            step = start_step + local_step
            if profiler is not None and step == profiler.start_step:
                profiler.start()
                print(f"GPU profiler started at step {step:,}")
            next_tokens_seen = min(target_tokens, tokens_seen + tokens_per_step)
            lr = learning_rate(next_tokens_seen, target_tokens, args.warmup_steps * tokens_per_step, args.lr)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            accumulated_loss = 0.0
            for _ in range(args.grad_accum):
                x, y = move_batch_to_device(*prefetcher.next(), device)
                amp_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
                with amp_context:
                    _, loss = model(x, y, use_fused_cross_entropy=args.fused_cross_entropy)
                    loss = loss / args.grad_accum
                loss.backward()
                accumulated_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            window_tokens, tokens_seen = window_tokens + tokens_per_step, next_tokens_seen
            if profiler is not None:
                result = profiler.step(step)
                if result:
                    print(f"GPU profiler saved {result[0]} and {result[1]}")
            if (step + 1) % args.log_every == 0 or local_step + 1 == remaining_steps:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                    peak_gib = torch.cuda.max_memory_allocated(device) / 1024**3
                else:
                    peak_gib = 0.0
                tokens_per_second = window_tokens / (perf_counter() - window_started_at)
                print(f"progress {tokens_seen / target_tokens:6.2%} | step {step + 1:,}/{final_step:,} | {tokens_per_second:,.0f} tokens/s | ETA {format_duration((target_tokens - tokens_seen) / tokens_per_second)} | VRAM {peak_gib:.2f} GiB | loss {accumulated_loss:.4f} | lr {lr:.2e}", flush=True)
                window_started_at, window_tokens = perf_counter(), 0
            if val_tokens is not None and args.eval_every > 0 and step > 0 and step % args.eval_every == 0:
                print(f"step {step:6d} | validation loss {estimate_loss(model, val_tokens, args.batch_size, args.seq_len, device, args.eval_steps):.4f}")
                window_started_at, window_tokens = perf_counter(), 0
            if (args.save_every > 0 and (step + 1) % args.save_every == 0) or local_step + 1 == remaining_steps:
                path = save_checkpoint(output_dir, model_to_manage, optimizer, step + 1, tokens_seen, args.max_tokens, sources, weights)
                print(f"saved {path}")
                window_started_at, window_tokens = perf_counter(), 0
    finally:
        prefetcher.close()
