"""Memory-mapped token sources and asynchronous batch preparation."""

from __future__ import annotations

import json
import math
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch


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


def load_train_sources(train_data: str | None, train_sources: str | None) -> tuple[list[tuple[str, np.memmap]], np.ndarray]:
    if train_data:
        return [(train_data, np.memmap(train_data, dtype=np.uint16, mode="r"))], np.array([1.0])
    if not train_sources:
        raise ValueError("provide --train-data or --train-sources")
    manifest_path = Path(train_sources)
    raw_sources = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError("--train-sources must point to a non-empty JSON array")
    sources: list[tuple[str, np.memmap]] = []
    weights: list[float] = []
    for index, source in enumerate(raw_sources, start=1):
        if not isinstance(source, dict) or not isinstance(source.get("path"), str):
            raise ValueError(f"source {index} must include a string path field")
        weight = source.get("weight")
        if not isinstance(weight, (int, float)) or isinstance(weight, bool) or not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"source {index} weight must be a positive number")
        source_path = Path(source["path"])
        if not source_path.is_absolute():
            source_path = manifest_path.parent / source_path
        sources.append((str(source_path), np.memmap(source_path, dtype=np.uint16, mode="r")))
        weights.append(float(weight))
    normalized = np.asarray(weights, dtype=np.float64)
    normalized /= normalized.sum()
    return sources, normalized


def get_weighted_batch(sources: list[tuple[str, np.memmap]], weights: np.ndarray, batch_size: int, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    return get_batch(sources[np.random.choice(len(sources), p=weights)][1], batch_size, seq_len)


class BatchPrefetcher:
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
