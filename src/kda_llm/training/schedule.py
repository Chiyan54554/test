"""Token-based training schedules and progress formatting."""

from __future__ import annotations

import math


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
