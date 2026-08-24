"""Model configuration shared by training and inference."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KDAConfig:
    vocab_size: int = 8192
    d_model: int = 512
    n_layers: int = 9
    n_heads: int = 4
    ffn_dim: int = 992
    max_seq_len: int = 2048

    @property
    def head_dim(self) -> int:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        return self.d_model // self.n_heads
