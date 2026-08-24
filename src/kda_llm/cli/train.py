"""Train the compact Chinese KDA language model from uint16 token streams."""

from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from kda_llm.model import KDAConfig, KDALanguageModel, LigerFusedLinearCrossEntropyLoss, chunk_kda, parameter_count
from kda_llm.config import load_json_object


def get_batch(tokens: np.memmap, batch_size: int, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    if len(tokens) <= seq_len:
        raise ValueError("token stream must be longer than seq_len")
    starts = np.random.randint(0, len(tokens) - seq_len - 1, size=batch_size)
    batch = np.stack([tokens[start : start + seq_len + 1] for start in starts]).astype(np.int64)
    batch_tensor = torch.from_numpy(batch)
    return batch_tensor[:, :-1], batch_tensor[:, 1:]


def move_batch_to_device(x: torch.Tensor, y: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if device.type == "cuda":
        x, y = x.pin_memory(), y.pin_memory()
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def load_train_sources(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> tuple[list[tuple[str, np.memmap]], np.ndarray]:
    if args.train_sources:
        if args.train_data_was_set:
            parser.error("provide exactly one of --train-data or --train-sources")
        args.train_data = None
    elif not args.train_data:
        parser.error("provide --train-data or --train-sources")

    if args.train_data:
        return [(args.train_data, np.memmap(args.train_data, dtype=np.uint16, mode="r"))], np.array([1.0])

    manifest_path = Path(args.train_sources)
    with manifest_path.open("r", encoding="utf-8") as manifest_file:
        raw_sources = json.load(manifest_file)
    if not isinstance(raw_sources, list) or not raw_sources:
        parser.error("--train-sources must point to a non-empty JSON array")

    sources: list[tuple[str, np.memmap]] = []
    weights: list[float] = []
    for index, source in enumerate(raw_sources, start=1):
        if not isinstance(source, dict) or not isinstance(source.get("path"), str):
            parser.error(f"source {index} must include a string path field")
        weight = source.get("weight")
        if (
            not isinstance(weight, (int, float))
            or isinstance(weight, bool)
            or not math.isfinite(weight)
            or weight <= 0
        ):
            parser.error(f"source {index} weight must be a positive number")
        source_path = Path(source["path"])
        if not source_path.is_absolute():
            source_path = manifest_path.parent / source_path
        sources.append((str(source_path), np.memmap(source_path, dtype=np.uint16, mode="r")))
        weights.append(float(weight))

    normalized_weights = np.asarray(weights, dtype=np.float64)
    normalized_weights /= normalized_weights.sum()
    return sources, normalized_weights


def get_weighted_batch(
    sources: list[tuple[str, np.memmap]],
    weights: np.ndarray,
    batch_size: int,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    source_index = np.random.choice(len(sources), p=weights)
    return get_batch(sources[source_index][1], batch_size, seq_len)


class BatchPrefetcher:
    """Prepare the next CPU batch while CUDA processes the current batch."""

    def __init__(self, sources: list[tuple[str, np.memmap]], weights: np.ndarray, batch_size: int, seq_len: int) -> None:
        self.sources, self.weights = sources, weights
        self.batch_size, self.seq_len = batch_size, seq_len
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kda-prefetch")
        self.future: Future[tuple[torch.Tensor, torch.Tensor]] | None = None

    def next(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.future is None:
            self.future = self.executor.submit(get_weighted_batch, self.sources, self.weights, self.batch_size, self.seq_len)
        batch = self.future.result()
        self.future = self.executor.submit(get_weighted_batch, self.sources, self.weights, self.batch_size, self.seq_len)
        return batch

    def close(self) -> None:
        self.executor.shutdown(wait=True, cancel_futures=True)


@torch.no_grad()
def estimate_loss(
    model: KDALanguageModel,
    tokens: np.memmap,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    steps: int = 20,
) -> float:
    model.eval()
    losses = []
    for _ in range(steps):
        x, y = move_batch_to_device(*get_batch(tokens, batch_size, seq_len), device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def learning_rate(tokens_seen: int, target_tokens: int, warmup_tokens: int, peak_lr: float) -> float:
    if tokens_seen < warmup_tokens:
        return peak_lr * tokens_seen / max(1, warmup_tokens)
    progress = (tokens_seen - warmup_tokens) / max(1, target_tokens - warmup_tokens)
    return peak_lr * 0.1 + peak_lr * 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


def format_duration(seconds: float) -> str:
    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}"


def main() -> None:
    bootstrap_parser = argparse.ArgumentParser(add_help=False)
    bootstrap_parser.add_argument("--train-config")
    bootstrap_args, _ = bootstrap_parser.parse_known_args()
    parser = argparse.ArgumentParser(description="Train a 32M Chinese KDA language model.")
    parser.add_argument("--train-config", help="JSON file containing training hyperparameters")
    parser.set_defaults(train_data_was_set=False)
    parser.add_argument(
        "--train-data",
        default="runs/smoke/corpus/train.bin",
        help="single uint16 .bin token stream",
    )
    parser.add_argument("--train-sources", help="JSON array of weighted uint16 token streams")
    parser.add_argument("--model-config", default="configs/model_32m.json", help="KDA architecture JSON")
    parser.add_argument("--val-data", default="runs/smoke/corpus/valid.bin", help="optional uint16 .bin validation stream")
    parser.add_argument("--out-dir", default="runs/smoke/checkpoints")
    parser.add_argument("--resume-from", help="checkpoint written by kda-train to continue from")
    parser.add_argument("--max-tokens", type=int, help="total training tokens; derives the number of optimizer steps")
    parser.add_argument("--steps", type=int, help="legacy explicit step count, intended only for smoke tests")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=5000, help="checkpoint interval; 0 saves only at the end")
    parser.add_argument("--eval-every", type=int, default=2000, help="validation interval; 0 disables validation")
    parser.add_argument("--eval-steps", type=int, default=5, help="validation batches per evaluation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=20, help="progress update interval in optimizer steps")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fused-cross-entropy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--require-kda-kernel", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--profile-start-step", type=int, help="global optimizer step at which to start the GPU profiler")
    parser.add_argument("--profile-warmup-steps", type=int, default=5, help="unrecorded profiler warmup steps")
    parser.add_argument("--profile-steps", type=int, default=0, help="number of GPU profiler steps; 0 disables profiling")
    parser.add_argument("--profile-dir", default="runs/smoke/profiles", help="directory for profiler traces and summaries")
    if bootstrap_args.train_config:
        parser.set_defaults(**load_json_object(
            bootstrap_args.train_config,
            {"max_tokens", "batch_size", "grad_accum", "seq_len", "lr", "warmup_steps", "save_every", "eval_every", "eval_steps", "seed", "log_every", "compile", "fused_cross_entropy", "require_kda_kernel"},
        ))
    args = parser.parse_args()
    args.train_data_was_set = "--train-data" in sys.argv[1:]

    model_config = KDAConfig(**load_json_object(args.model_config, set(KDAConfig.__dataclass_fields__)))
    if args.seq_len > model_config.max_seq_len:
        raise ValueError("seq_len exceeds the model max_seq_len")
    if args.log_every <= 0:
        parser.error("--log-every must be a positive integer")
    if args.profile_steps < 0 or args.profile_warmup_steps < 0:
        parser.error("profile step counts must be non-negative")
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
    total_steps = math.ceil(target_tokens / tokens_per_step)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available to PyTorch")
    if args.device == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_name = args.device
    device = torch.device(device_name)
    if args.require_kda_kernel and device.type == "cuda" and chunk_kda is None:
        parser.error("chunk_kda is unavailable; install the cuda extra or pass --no-require-kda-kernel")
    if args.fused_cross_entropy and (device.type != "cuda" or LigerFusedLinearCrossEntropyLoss is None):
        parser.error("fused cross entropy requires Linux CUDA and `uv sync --extra cuda`")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    use_amp = device.type == "cuda" and torch.cuda.is_bf16_supported()

    train_sources, train_weights = load_train_sources(args, parser)
    val_tokens = np.memmap(args.val_data, dtype=np.uint16, mode="r") if args.val_data else None
    model = KDALanguageModel(model_config).to(device)
    if args.compile:
        model = torch.compile(model, dynamic=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    start_step = 0
    tokens_seen = 0
    if args.resume_from:
        checkpoint = torch.load(args.resume_from, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or "model" not in checkpoint or "optimizer" not in checkpoint:
            parser.error("--resume-from must point to a kda-train checkpoint")
        checkpoint_config = checkpoint.get("config")
        if checkpoint_config is not None and checkpoint_config != asdict(model_config):
            parser.error("checkpoint model config does not match --model-config")
        # The fused input projection changes optimizer parameter identities. Model
        # weights are upgraded losslessly, but old Adam moments cannot be mapped
        # safely, so start fresh optimizer state for those legacy checkpoints.
        legacy_input_projections = any(key.endswith(".attention.qkv_proj.weight") for key in checkpoint["model"])
        model_to_load = getattr(model, "_orig_mod", model)
        model_to_load.load_state_dict(checkpoint["model"], strict=True)
        if legacy_input_projections:
            print("legacy input projections detected; restored model weights and reset optimizer state")
        else:
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint.get("step", 0))
        tokens_seen = int(checkpoint.get("tokens_seen", start_step * tokens_per_step))
        if tokens_seen >= target_tokens:
            parser.error("checkpoint already reaches the requested token budget")
    remaining_steps = math.ceil((target_tokens - tokens_seen) / tokens_per_step)
    final_step = start_step + remaining_steps
    profile_start_step = args.profile_start_step if args.profile_start_step is not None else start_step
    profile_end_step = profile_start_step + args.profile_warmup_steps + args.profile_steps
    if args.profile_steps and device.type != "cuda":
        parser.error("GPU profiling requires --device cuda")
    if args.profile_steps and not start_step <= profile_start_step < final_step:
        parser.error("--profile-start-step must be within the remaining training steps")
    if args.profile_steps and profile_end_step > final_step:
        parser.error("profiler warmup and active steps exceed the remaining training steps")
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    backend = "chunk_kda" if device.type == "cuda" and chunk_kda is not None else "reference_recurrence"
    print(f"device: {device}; backend: {backend}; compiled: {args.compile}; parameters: {parameter_count(model):,}")
    print(f"training budget: {target_tokens:,} tokens; steps: {final_step:,}; tokens/step: {tokens_per_step:,}")
    if start_step:
        print(f"resuming from step {start_step:,} ({tokens_seen:,} tokens)")
    for (source_path, _), weight in zip(train_sources, train_weights, strict=True):
        print(f"training source: {source_path} ({weight:.1%})")

    prefetcher = BatchPrefetcher(train_sources, train_weights, args.batch_size, args.seq_len)
    started_at = perf_counter()
    window_started_at = started_at
    window_tokens = 0
    profiler: torch.profiler.profile | None = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for local_step in range(remaining_steps):
        step = start_step + local_step
        if args.profile_steps and step == profile_start_step:
            Path(args.profile_dir).mkdir(parents=True, exist_ok=True)
            profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=0, warmup=args.profile_warmup_steps, active=args.profile_steps),
                record_shapes=True,
                profile_memory=True,
            )
            profiler.__enter__()
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
        window_tokens += tokens_per_step
        tokens_seen = next_tokens_seen
        if profiler is not None:
            profiler.step()
        if profiler is not None and step + 1 == profile_end_step:
            profiler.__exit__(None, None, None)
            trace_path = Path(args.profile_dir) / f"trace-step-{profile_start_step}-{profile_end_step}.json"
            summary_path = Path(args.profile_dir) / f"summary-step-{profile_start_step}-{profile_end_step}.txt"
            profiler.export_chrome_trace(str(trace_path))
            summary_path.write_text(
                profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=30), encoding="utf-8"
            )
            print(f"GPU profiler saved {trace_path} and {summary_path}")
            profiler = None

        if (step + 1) % args.log_every == 0 or local_step + 1 == remaining_steps:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                peak_gib = torch.cuda.max_memory_allocated(device) / 1024**3
            else:
                peak_gib = 0.0
            elapsed = perf_counter() - window_started_at
            tokens_per_second = window_tokens / elapsed
            remaining = max(0, target_tokens - tokens_seen) / tokens_per_second
            progress = min(1.0, tokens_seen / target_tokens)
            print(
                f"progress {progress:6.2%} | step {step + 1:,}/{final_step:,} | "
                f"{tokens_per_second:,.0f} tokens/s | "
                f"ETA {format_duration(remaining)} | VRAM {peak_gib:.2f} GiB | loss {accumulated_loss:.4f} | lr {lr:.2e}",
                flush=True,
            )
            window_started_at, window_tokens = perf_counter(), 0
        if val_tokens is not None and args.eval_every > 0 and step > 0 and step % args.eval_every == 0:
            val_loss = estimate_loss(model, val_tokens, args.batch_size, args.seq_len, device, args.eval_steps)
            print(f"step {step:6d} | validation loss {val_loss:.4f}")
            window_started_at, window_tokens = perf_counter(), 0
        if (args.save_every > 0 and (step + 1) % args.save_every == 0) or local_step + 1 == remaining_steps:
            model_to_save = getattr(model, "_orig_mod", model)
            checkpoint = {
                "model": model_to_save.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step + 1,
                "tokens_seen": tokens_seen,
                "max_tokens": args.max_tokens,
                "config": asdict(model_to_save.config),
                "training_sources": [
                    {"path": source_path, "weight": float(weight)}
                    for (source_path, _), weight in zip(train_sources, train_weights, strict=True)
                ],
            }
            checkpoint_path = output_dir / f"kda-step-{step + 1}.pt"
            torch.save(checkpoint, checkpoint_path)
            print(f"saved {checkpoint_path}")
            window_started_at, window_tokens = perf_counter(), 0
    prefetcher.close()


if __name__ == "__main__":
    main()
