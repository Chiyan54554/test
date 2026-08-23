"""Train the compact Chinese KDA language model from uint16 token streams."""

from __future__ import annotations

import argparse
import json
import math
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from kda_llm.model import KDAConfig, KDALanguageModel, parameter_count


def get_batch(tokens: np.memmap, batch_size: int, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if len(tokens) <= seq_len:
        raise ValueError("token stream must be longer than seq_len")
    starts = np.random.randint(0, len(tokens) - seq_len - 1, size=batch_size)
    batch = np.stack([tokens[start : start + seq_len + 1] for start in starts]).astype(np.int64)
    batch_tensor = torch.from_numpy(batch).to(device, non_blocking=True)
    return batch_tensor[:, :-1], batch_tensor[:, 1:]


def load_train_sources(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> tuple[list[tuple[str, np.memmap]], np.ndarray]:
    if bool(args.train_data) == bool(args.train_sources):
        parser.error("provide exactly one of --train-data or --train-sources")

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
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    source_index = np.random.choice(len(sources), p=weights)
    return get_batch(sources[source_index][1], batch_size, seq_len, device)


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
        x, y = get_batch(tokens, batch_size, seq_len, device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def learning_rate(step: int, total_steps: int, warmup_steps: int, peak_lr: float) -> float:
    if step < warmup_steps:
        return peak_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return peak_lr * 0.1 + peak_lr * 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a 32M Chinese KDA language model.")
    parser.add_argument("--train-data", help="single uint16 .bin token stream")
    parser.add_argument("--train-sources", help="JSON array of weighted uint16 token streams")
    parser.add_argument("--val-data", help="optional uint16 .bin validation stream")
    parser.add_argument("--out-dir", default="checkpoints")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.seq_len > KDAConfig().max_seq_len:
        raise ValueError("seq_len exceeds the model max_seq_len")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available to PyTorch")
    if args.device == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_name = args.device
    device = torch.device(device_name)
    use_amp = device.type == "cuda" and torch.cuda.is_bf16_supported()

    train_sources, train_weights = load_train_sources(args, parser)
    val_tokens = np.memmap(args.val_data, dtype=np.uint16, mode="r") if args.val_data else None
    model = KDALanguageModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"device: {device}; parameters: {parameter_count(model):,}")
    for (source_path, _), weight in zip(train_sources, train_weights, strict=True):
        print(f"training source: {source_path} ({weight:.1%})")

    for step in range(args.steps):
        lr = learning_rate(step, args.steps, args.warmup_steps, args.lr)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        for _ in range(args.grad_accum):
            x, y = get_weighted_batch(
                train_sources, train_weights, args.batch_size, args.seq_len, device
            )
            amp_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
            with amp_context:
                _, loss = model(x, y)
                loss = loss / args.grad_accum
            loss.backward()
            accumulated_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % 20 == 0:
            print(f"step {step:6d} | loss {accumulated_loss:.4f} | lr {lr:.2e}")
        if val_tokens is not None and step > 0 and step % args.eval_every == 0:
            val_loss = estimate_loss(model, val_tokens, args.batch_size, args.seq_len, device)
            print(f"step {step:6d} | validation loss {val_loss:.4f}")
        if (step + 1) % args.save_every == 0 or step + 1 == args.steps:
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step + 1,
                "config": asdict(model.config),
                "training_sources": [
                    {"path": source_path, "weight": float(weight)}
                    for (source_path, _), weight in zip(train_sources, train_weights, strict=True)
                ],
            }
            checkpoint_path = output_dir / f"kda-step-{step + 1}.pt"
            torch.save(checkpoint, checkpoint_path)
            print(f"saved {checkpoint_path}")


if __name__ == "__main__":
    main()
