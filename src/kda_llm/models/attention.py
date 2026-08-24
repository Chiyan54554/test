"""Kimi Delta Attention and its recurrent generation cache."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import KDAConfig
from .kernels import chunk_kda
from .layers import CausalDepthwiseConv1d, RMSNorm, RotaryCache, apply_rope


@dataclass
class KDAAttentionCache:
    recurrent_state: torch.Tensor | None = None
    conv_state: torch.Tensor | None = None


class KimiDeltaAttention(nn.Module):
    """Multi-head KDA with a fused input projection and legacy checkpoint upgrade."""

    def __init__(self, config: KDAConfig, rope_cache: RotaryCache) -> None:
        super().__init__()
        self.n_heads, self.head_dim, self.rope_cache = config.n_heads, config.head_dim, rope_cache
        self.input_proj = nn.Linear(config.d_model, 5 * config.d_model + config.n_heads, bias=False)
        self.alpha_bias = nn.Parameter(torch.zeros(config.d_model))
        self.beta_bias = nn.Parameter(torch.zeros(config.n_heads))
        self.qkv_conv = CausalDepthwiseConv1d(3 * config.d_model)
        self.q_norm, self.k_norm = RMSNorm(self.head_dim), RMSNorm(self.head_dim)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

    def _load_from_state_dict(self, state_dict: dict[str, torch.Tensor], prefix: str, local_metadata: dict[str, object], strict: bool, missing_keys: list[str], unexpected_keys: list[str], error_msgs: list[str]) -> None:
        fused_weight = prefix + "input_proj.weight"
        legacy_weights = (prefix + "qkv_proj.weight", prefix + "alpha_proj.weight", prefix + "gate_proj.weight", prefix + "beta_proj.weight")
        if fused_weight not in state_dict and all(key in state_dict for key in legacy_weights):
            state_dict[fused_weight] = torch.cat([state_dict[key] for key in legacy_weights], dim=0)
            state_dict[prefix + "alpha_bias"] = state_dict[prefix + "alpha_proj.bias"]
            state_dict[prefix + "beta_bias"] = state_dict[prefix + "beta_proj.bias"]
        for key in (*legacy_weights, prefix + "alpha_proj.bias", prefix + "beta_proj.bias"):
            state_dict.pop(key, None)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def _chunk_kernel(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, decay_logits: torch.Tensor, beta_logits: torch.Tensor, initial_state: torch.Tensor | None, output_final_state: bool) -> tuple[torch.Tensor, torch.Tensor | None]:
        dtype = torch.bfloat16 if q.is_cuda and torch.cuda.is_bf16_supported() else q.dtype
        q, k, v, decay_logits, beta_logits = (tensor.to(dtype) for tensor in (q, k, v, decay_logits, beta_logits))
        result = chunk_kda(q=q, k=k, v=v, g=torch.log(torch.sigmoid(decay_logits).clamp_min(torch.finfo(dtype).tiny)), beta=torch.sigmoid(beta_logits), scale=1.0, initial_state=initial_state, output_final_state=output_final_state, use_gate_in_kernel=False, use_qk_l2norm_in_kernel=True, use_beta_sigmoid_in_kernel=False)
        return result if isinstance(result, tuple) else (result, None)

    @staticmethod
    def _reference_recurrence(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, decay_logits: torch.Tensor, beta_logits: torch.Tensor, initial_state: torch.Tensor | None, output_final_state: bool) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, seq_len, n_heads, head_dim = q.shape
        q, k, v = F.normalize(q, dim=-1).transpose(1, 2), F.normalize(k, dim=-1).transpose(1, 2), v.transpose(1, 2)
        alpha, beta = torch.sigmoid(decay_logits.transpose(1, 2)), torch.sigmoid(beta_logits.transpose(1, 2)).unsqueeze(-1)
        state = initial_state if initial_state is not None else q.new_zeros(batch_size, n_heads, head_dim, head_dim)
        outputs: list[torch.Tensor] = []
        for index in range(seq_len):
            k_t, v_t = k[:, :, index], v[:, :, index]
            state = state * alpha[:, :, index].unsqueeze(-1)
            residual = beta[:, :, index] * (v_t - torch.einsum("bhkv,bhk->bhv", state, k_t))
            state = state + k_t.unsqueeze(-1) * residual.unsqueeze(-2)
            outputs.append(torch.einsum("bhkv,bhk->bhv", state, q[:, :, index]))
        return torch.stack(outputs, dim=2).transpose(1, 2), state.detach() if output_final_state else None

    def forward(self, x: torch.Tensor, cache: KDAAttentionCache | None = None, use_cache: bool = False, position_offset: int = 0) -> tuple[torch.Tensor, KDAAttentionCache | None]:
        batch_size, seq_len, _ = x.shape
        qkv, decay_logits, gate, beta_logits = self.input_proj(x).split((3 * self.n_heads * self.head_dim, self.n_heads * self.head_dim, self.n_heads * self.head_dim, self.n_heads), dim=-1)
        qkv, conv_state = self.qkv_conv(qkv, state=cache.conv_state if cache else None, use_cache=use_cache)
        q, k, v = qkv.chunk(3, dim=-1)
        split_heads = lambda tensor: tensor.view(batch_size, seq_len, self.n_heads, self.head_dim)
        q = apply_rope(self.q_norm(split_heads(q)), self.rope_cache, position_offset)
        k = apply_rope(self.k_norm(split_heads(k)), self.rope_cache, position_offset)
        v, decay_logits, beta_logits = split_heads(v), split_heads(decay_logits + self.alpha_bias), beta_logits + self.beta_bias
        initial_state = cache.recurrent_state if cache else None
        if chunk_kda is not None and x.is_cuda and self.head_dim == 128:
            y, recurrent_state = self._chunk_kernel(q, k, v, decay_logits, beta_logits, initial_state, use_cache)
        else:
            y, recurrent_state = self._reference_recurrence(q, k, v, decay_logits, beta_logits, initial_state, use_cache)
        next_cache = KDAAttentionCache(recurrent_state, conv_state) if use_cache else None
        return self.out_proj(y.reshape(batch_size, seq_len, -1) * F.silu(gate)), next_cache
