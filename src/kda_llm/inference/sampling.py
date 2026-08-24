"""Sampling strategies for autoregressive generation."""

from __future__ import annotations

import torch


def sample_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
    history: torch.Tensor | None = None,
    repetition_penalty: float = 1.0,
) -> torch.Tensor:
    if repetition_penalty < 1.0:
        raise ValueError("repetition penalty must be at least 1")
    logits = logits / temperature
    if repetition_penalty > 1.0 and history is not None:
        logits = logits.clone()
        for batch_index in range(logits.size(0)):
            previous = history[batch_index].unique()
            previous_logits = logits[batch_index, previous]
            logits[batch_index, previous] = torch.where(
                previous_logits < 0,
                previous_logits * repetition_penalty,
                previous_logits / repetition_penalty,
            )
    if top_k:
        cutoff = torch.topk(logits, min(top_k, logits.size(-1))).values[..., -1, None]
        logits = logits.masked_fill(logits < cutoff, float("-inf"))
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probabilities = torch.softmax(sorted_logits, dim=-1)
        remove = probabilities.cumsum(dim=-1) - probabilities >= top_p
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_indices, sorted_logits.masked_fill(remove, float("-inf")))
    return torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
