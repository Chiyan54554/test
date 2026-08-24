"""Top-level decoder-only KDA language model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import KDAAttentionCache, KimiDeltaAttention
from .config import KDAConfig
from .kernels import LigerFusedLinearCrossEntropyLoss, fused_linear_cross_entropy
from .layers import RMSNorm, RotaryCache, SwiGLU


class KDABlock(nn.Module):
    def __init__(self, config: KDAConfig, rope_cache: RotaryCache) -> None:
        super().__init__()
        self.attn_norm, self.attention = RMSNorm(config.d_model), KimiDeltaAttention(config, rope_cache)
        self.ffn_norm, self.ffn = RMSNorm(config.d_model), SwiGLU(config.d_model, config.ffn_dim)
        self.residual_scale = (2 * config.n_layers) ** -0.5

    def forward(self, x: torch.Tensor, cache: KDAAttentionCache | None = None, use_cache: bool = False, position_offset: int = 0) -> tuple[torch.Tensor, KDAAttentionCache | None]:
        attention_output, next_cache = self.attention(self.attn_norm(x), cache, use_cache, position_offset)
        x = x + self.residual_scale * attention_output
        return x + self.residual_scale * self.ffn(self.ffn_norm(x)), next_cache


class KDALanguageModel(nn.Module):
    def __init__(self, config: KDAConfig = KDAConfig()) -> None:
        super().__init__()
        self.config, self.rope_cache = config, RotaryCache()
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList(KDABlock(config, self.rope_cache) for _ in range(config.n_layers))
        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.fused_loss = LigerFusedLinearCrossEntropyLoss() if LigerFusedLinearCrossEntropyLoss else None
        self.apply(self._init_weights)
        residual_std = 0.02 / (2 * config.n_layers) ** 0.5
        for layer in self.layers:
            nn.init.normal_(layer.attention.out_proj.weight, mean=0.0, std=residual_std)
            nn.init.normal_(layer.ffn.out_proj.weight, mean=0.0, std=residual_std)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor | None = None, past_states: list[KDAAttentionCache] | None = None, use_cache: bool = False, position_offset: int = 0, use_fused_cross_entropy: bool = False) -> tuple[torch.Tensor | None, torch.Tensor | None] | tuple[torch.Tensor | None, torch.Tensor | None, list[KDAAttentionCache]]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.size(1) > self.config.max_seq_len:
            raise ValueError(f"sequence length exceeds {self.config.max_seq_len}")
        if past_states is not None and len(past_states) != len(self.layers):
            raise ValueError("past_states must contain one cache per layer")
        x, next_states = self.token_embedding(input_ids), []
        for index, layer in enumerate(self.layers):
            x, layer_cache = layer(x, past_states[index] if past_states else None, use_cache, position_offset)
            if layer_cache is not None:
                next_states.append(layer_cache)
        hidden_states, loss = self.final_norm(x), None
        if targets is not None and use_fused_cross_entropy:
            if self.fused_loss is None:
                raise RuntimeError("fused cross entropy requires the liger-kernel cuda extra")
            loss = fused_linear_cross_entropy(self.fused_loss, self.lm_head.weight, hidden_states.reshape(-1, hidden_states.size(-1)), targets.reshape(-1))
            logits = None
        else:
            logits = self.lm_head(hidden_states)
            if targets is not None:
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return (logits, loss, next_states) if use_cache else (logits, loss)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
