"""Compatibility exports for the pre-v2 KDA model module.

New code should import from :mod:`kda_llm.models`.
"""

from __future__ import annotations

import torch

from .models import KDAAttentionCache, KDAConfig, KDABlock, KDALanguageModel, KimiDeltaAttention, parameter_count
from .models.kernels import LigerFusedLinearCrossEntropyLoss, chunk_kda

__all__ = [
    "KDAAttentionCache", "KDAConfig", "KDABlock", "KDALanguageModel", "KimiDeltaAttention",
    "LigerFusedLinearCrossEntropyLoss", "chunk_kda", "parameter_count",
]


def main() -> None:
    torch.manual_seed(7)
    model = KDALanguageModel()
    tokens = torch.randint(0, model.config.vocab_size, (2, 16))
    logits, loss = model(tokens[:, :-1], tokens[:, 1:])
    print(f"trainable parameters: {parameter_count(model):,}")
    print(f"logits shape: {tuple(logits.shape)}")
    print(f"self-test loss: {loss.item():.4f}")


if __name__ == "__main__":
    main()
