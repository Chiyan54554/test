"""A compact, modern 32M-parameter Kimi Delta Attention language model.

This reference implementation favors correctness and clarity over kernel-level
speed. For long-sequence training, replace the token loop in KimiDeltaAttention
with a chunkwise scan implemented in Triton or CUDA.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from fla.ops.kda import chunk_kda
except ImportError:
    chunk_kda = None


@dataclass(frozen=True)
class KDAConfig:
    # Compact BPE vocabulary with UTF-8 byte fallback for rare characters.
    vocab_size: int = 8192
    d_model: int = 512
    n_layers: int = 9
    # Four 128-d heads match the FlashKDA kernel's supported state dimensions.
    n_heads: int = 4
    ffn_dim: int = 992
    max_seq_len: int = 2048

    @property
    def head_dim(self) -> int:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        return self.d_model // self.n_heads


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class RotaryCache:
    """Share RoPE tables across all layers for a fixed CUDA device and dtype."""

    def __init__(self) -> None:
        self.key: tuple[torch.device, torch.dtype, int] | None = None
        self.cos = torch.empty(0)
        self.sin = torch.empty(0)

    def get(self, length: int, head_dim: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        key = (device, dtype, head_dim)
        if self.key != key or self.cos.size(0) < length:
            positions = torch.arange(length, device=device, dtype=dtype)
            frequencies = torch.arange(0, head_dim, 2, device=device, dtype=dtype)
            angles = torch.outer(positions, 1.0 / (10000 ** (frequencies / head_dim)))
            self.cos, self.sin, self.key = angles.cos(), angles.sin(), key
        return self.cos[:length], self.sin[:length]


def apply_rope(x: torch.Tensor, rope_cache: RotaryCache, position_offset: int = 0) -> torch.Tensor:
    """Apply rotary positions to [batch, sequence, heads, head_dim] tensors."""
    _, seq_len, _, head_dim = x.shape
    if head_dim % 2:
        raise ValueError("RoPE requires an even head dimension")
    cos, sin = rope_cache.get(position_offset + seq_len, head_dim, x.device, x.dtype)
    cos, sin = cos[position_offset:][None, :, None], sin[position_offset:][None, :, None]
    x_even, x_odd = x[..., ::2], x[..., 1::2]
    return torch.stack((x_even * cos - x_odd * sin, x_even * sin + x_odd * cos), dim=-1).flatten(-2)


class CausalDepthwiseConv1d(nn.Module):
    """A lightweight local-context mixer, as used in current linear-attention models."""

    def __init__(self, channels: int, kernel_size: int = 4) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(channels, channels, kernel_size, groups=channels, bias=True)

    def forward(
        self, x: torch.Tensor, state: torch.Tensor | None = None, use_cache: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = x.transpose(1, 2)
        if state is None:
            convolved = self.conv(F.pad(x, (self.kernel_size - 1, 0)))
        else:
            convolved = self.conv(torch.cat((state, x), dim=-1))
        next_state = None
        if use_cache:
            combined = x if state is None else torch.cat((state, x), dim=-1)
            next_state = combined[..., -(self.kernel_size - 1) :].detach()
        return convolved.transpose(1, 2), next_state


@dataclass
class KDAAttentionCache:
    """Per-layer state needed to continue KDA and its causal convolution."""

    recurrent_state: torch.Tensor | None = None
    conv_state: torch.Tensor | None = None


class KimiDeltaAttention(nn.Module):
    """Multi-head KDA with per-value-channel decay and delta-rule writes."""

    def __init__(self, config: KDAConfig, rope_cache: RotaryCache) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.rope_cache = rope_cache
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.qkv_conv = CausalDepthwiseConv1d(3 * config.d_model)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.alpha_proj = nn.Linear(config.d_model, config.d_model)
        self.beta_proj = nn.Linear(config.d_model, config.n_heads)
        self.gate_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

    def _chunk_kernel(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        decay_logits: torch.Tensor,
        beta_logits: torch.Tensor,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Dispatch KDA to FLA's chunkwise Triton/FlashKDA backend."""
        # FLA's current Triton KDA kernels expect matching dtypes for all dot
        # operands. Prefer bf16 on CUDA-capable GPUs and cast every kernel input
        # consistently so autocast does not leave mixed fp32/bf16 operands.
        kernel_dtype = torch.bfloat16 if q.is_cuda and torch.cuda.is_bf16_supported() else q.dtype
        q, k, v = q.to(kernel_dtype), k.to(kernel_dtype), v.to(kernel_dtype)
        decay_logits, beta_logits = decay_logits.to(kernel_dtype), beta_logits.to(kernel_dtype)
        decay = torch.log(torch.sigmoid(decay_logits).clamp_min(torch.finfo(decay_logits.dtype).tiny))
        beta = torch.sigmoid(beta_logits)
        result = chunk_kda(
            q=q,
            k=k,
            v=v,
            g=decay,
            beta=beta,
            scale=1.0,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_gate_in_kernel=False,
            use_qk_l2norm_in_kernel=True,
            use_beta_sigmoid_in_kernel=False,
        )
        output, final_state = result if isinstance(result, tuple) else (result, None)
        return output, final_state

    @staticmethod
    def _reference_recurrence(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        decay_logits: torch.Tensor,
        beta_logits: torch.Tensor,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Numerical fallback matching the KDA state update used by the kernel."""
        batch_size, seq_len, n_heads, head_dim = q.shape
        q = F.normalize(q, dim=-1).transpose(1, 2)
        k = F.normalize(k, dim=-1).transpose(1, 2)
        v = v.transpose(1, 2)
        decay_logits = decay_logits.transpose(1, 2)
        alpha = torch.sigmoid(decay_logits)
        beta = torch.sigmoid(beta_logits).unsqueeze(-1)
        state = initial_state if initial_state is not None else q.new_zeros(batch_size, n_heads, head_dim, head_dim)
        outputs: list[torch.Tensor] = []

        for token_index in range(seq_len):
            k_t = k[:, :, token_index]
            v_t = v[:, :, token_index]
            # KDA: decay key channels before applying the delta-rule erase/write.
            state = state * alpha[:, :, token_index].unsqueeze(-1)
            predicted_v = torch.einsum("bhkv,bhk->bhv", state, k_t)
            residual = beta[:, :, token_index] * (v_t - predicted_v)
            state = state + k_t.unsqueeze(-1) * residual.unsqueeze(-2)
            outputs.append(torch.einsum("bhkv,bhk->bhv", state, q[:, :, token_index]))

        return torch.stack(outputs, dim=2).transpose(1, 2), state.detach() if output_final_state else None

    def forward(
        self,
        x: torch.Tensor,
        cache: KDAAttentionCache | None = None,
        use_cache: bool = False,
        position_offset: int = 0,
    ) -> tuple[torch.Tensor, KDAAttentionCache | None]:
        batch_size, seq_len, _ = x.shape
        qkv, conv_state = self.qkv_conv(
            self.qkv_proj(x), state=cache.conv_state if cache else None, use_cache=use_cache
        )
        q, k, v = qkv.chunk(3, dim=-1)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(batch_size, seq_len, self.n_heads, self.head_dim)

        # QK normalization and RoPE are applied before the chunkwise KDA kernel.
        q = apply_rope(self.q_norm(split_heads(q)), self.rope_cache, position_offset)
        k = apply_rope(self.k_norm(split_heads(k)), self.rope_cache, position_offset)
        v = split_heads(v)
        decay_logits = split_heads(self.alpha_proj(x))
        beta_logits = self.beta_proj(x).transpose(1, 2)

        can_use_kernel = chunk_kda is not None and x.is_cuda and self.head_dim == 128
        if can_use_kernel:
            y, recurrent_state = self._chunk_kernel(
                q, k, v, decay_logits, beta_logits,
                initial_state=cache.recurrent_state if cache else None,
                output_final_state=use_cache,
            )
        else:
            y, recurrent_state = self._reference_recurrence(
                q, k, v, decay_logits, beta_logits,
                initial_state=cache.recurrent_state if cache else None,
                output_final_state=use_cache,
            )

        y = y.reshape(batch_size, seq_len, -1)
        next_cache = KDAAttentionCache(recurrent_state, conv_state) if use_cache else None
        return self.out_proj(y * F.silu(self.gate_proj(x))), next_cache


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int) -> None:
        super().__init__()
        self.in_proj = nn.Linear(d_model, 2 * hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.in_proj(x).chunk(2, dim=-1)
        return self.out_proj(value * F.silu(gate))


class KDABlock(nn.Module):
    def __init__(self, config: KDAConfig, rope_cache: RotaryCache) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attention = KimiDeltaAttention(config, rope_cache)
        self.ffn_norm = RMSNorm(config.d_model)
        self.ffn = SwiGLU(config.d_model, config.ffn_dim)
        self.residual_scale = (2 * config.n_layers) ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        cache: KDAAttentionCache | None = None,
        use_cache: bool = False,
        position_offset: int = 0,
    ) -> tuple[torch.Tensor, KDAAttentionCache | None]:
        attention_output, next_cache = self.attention(
            self.attn_norm(x), cache=cache, use_cache=use_cache, position_offset=position_offset
        )
        x = x + self.residual_scale * attention_output
        return x + self.residual_scale * self.ffn(self.ffn_norm(x)), next_cache


class KDALanguageModel(nn.Module):
    def __init__(self, config: KDAConfig = KDAConfig()) -> None:
        super().__init__()
        self.config = config
        self.rope_cache = RotaryCache()
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList(KDABlock(config, self.rope_cache) for _ in range(config.n_layers))
        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # Weight tying saves ~4.2M parameters and is standard for language models.
        self.lm_head.weight = self.token_embedding.weight
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

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        past_states: list[KDAAttentionCache] | None = None,
        use_cache: bool = False,
        position_offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor | None] | tuple[torch.Tensor, torch.Tensor | None, list[KDAAttentionCache]]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.size(1) > self.config.max_seq_len:
            raise ValueError(f"sequence length exceeds {self.config.max_seq_len}")

        x = self.token_embedding(input_ids)
        if past_states is not None and len(past_states) != len(self.layers):
            raise ValueError("past_states must contain one cache per layer")
        next_states: list[KDAAttentionCache] = []
        for index, layer in enumerate(self.layers):
            x, layer_cache = layer(
                x,
                cache=past_states[index] if past_states else None,
                use_cache=use_cache,
                position_offset=position_offset,
            )
            if layer_cache is not None:
                next_states.append(layer_cache)
        logits = self.lm_head(self.final_norm(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        if use_cache:
            return logits, loss, next_states
        return logits, loss


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


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
