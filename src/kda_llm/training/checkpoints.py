"""Versioned checkpoints with compatibility for pre-v2 KDA runs."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch

from kda_llm.models import KDAConfig, KDALanguageModel

CHECKPOINT_VERSION = 2


def restore_checkpoint(path: str, model: KDALanguageModel, optimizer: torch.optim.Optimizer, config: KDAConfig, tokens_per_step: int) -> tuple[int, int, bool]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint or "optimizer" not in checkpoint:
        raise ValueError("--resume-from must point to a kda-train checkpoint")
    if checkpoint.get("config") is not None and checkpoint["config"] != asdict(config):
        raise ValueError("checkpoint model config does not match --model-config")
    model_state = checkpoint["model"]
    if not isinstance(model_state, dict):
        raise ValueError("checkpoint model state is invalid")
    legacy_projection = any(key.endswith(".attention.qkv_proj.weight") for key in model_state)
    model.load_state_dict(model_state, strict=True)
    if not legacy_projection:
        optimizer.load_state_dict(checkpoint["optimizer"])
    step = int(checkpoint.get("step", 0))
    return step, int(checkpoint.get("tokens_seen", step * tokens_per_step)), legacy_projection


def save_checkpoint(output_dir: Path, model: KDALanguageModel, optimizer: torch.optim.Optimizer, step: int, tokens_seen: int, max_tokens: int | None, sources: list[tuple[str, object]], weights: object) -> Path:
    checkpoint = {
        "format_version": CHECKPOINT_VERSION,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "tokens_seen": tokens_seen,
        "max_tokens": max_tokens,
        "config": asdict(model.config),
        "training_sources": [{"path": path, "weight": float(weight)} for (path, _), weight in zip(sources, weights, strict=True)],
    }
    path = output_dir / f"kda-step-{step}.pt"
    torch.save(checkpoint, path)
    return path
