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


def apply_rope(x: torch.Tensor) -> torch.Tensor:
    """Apply rotary positions to [batch, heads, sequence, head_dim] tensors."""
    _, _, seq_len, head_dim = x.shape
    if head_dim % 2:
        raise ValueError("RoPE requires an even head dimension")
    positions = torch.arange(seq_len, device=x.device, dtype=x.dtype)
    frequencies = torch.arange(0, head_dim, 2, device=x.device, dtype=x.dtype)
    inv_freq = 1.0 / (10000 ** (frequencies / head_dim))
    angles = torch.outer(positions, inv_freq)
    cos = angles.cos()[None, None]
    sin = angles.sin()[None, None]
    x_even, x_odd = x[..., ::2], x[..., 1::2]
    return torch.stack((x_even * cos - x_odd * sin, x_even * sin + x_odd * cos), dim=-1).flatten(-2)


class CausalDepthwiseConv1d(nn.Module):
    """A lightweight local-context mixer, as used in current linear-attention models."""

    def __init__(self, channels: int, kernel_size: int = 4) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(channels, channels, kernel_size, groups=channels, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = F.pad(x, (self.kernel_size - 1, 0))
        return self.conv(x).transpose(1, 2)


class KimiDeltaAttention(nn.Module):
    """Multi-head KDA with per-value-channel decay and delta-rule writes."""

    def __init__(self, config: KDAConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
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
    ) -> torch.Tensor:
        """Dispatch KDA to FLA's chunkwise Triton/FlashKDA backend."""
        # FLA's current Triton KDA kernels expect matching dtypes for all dot
        # operands. Prefer bf16 on CUDA-capable GPUs and cast every kernel input
        # consistently so autocast does not leave mixed fp32/bf16 operands.
        kernel_dtype = torch.bfloat16 if q.is_cuda and torch.cuda.is_bf16_supported() else q.dtype
        q = q.to(kernel_dtype)
        k = k.to(kernel_dtype)
        v = v.to(kernel_dtype)
        decay_logits = decay_logits.to(kernel_dtype)
        beta_logits = beta_logits.to(kernel_dtype)
        decay = torch.log(torch.sigmoid(decay_logits).clamp_min(torch.finfo(decay_logits.dtype).tiny))
        beta = torch.sigmoid(beta_logits)
        result = chunk_kda(
            q=q.transpose(1, 2).contiguous(),
            k=k.transpose(1, 2).contiguous(),
            v=v.transpose(1, 2).contiguous(),
            g=decay.transpose(1, 2).contiguous(),
            beta=beta.transpose(1, 2).contiguous(),
            scale=1.0,
            output_final_state=False,
            use_gate_in_kernel=False,
            use_qk_l2norm_in_kernel=True,
            use_beta_sigmoid_in_kernel=False,
        )
        # FLA returns only output unless output_final_state=True in most versions.
        output = result[0] if isinstance(result, tuple) else result
        return output.transpose(1, 2).contiguous()

    @staticmethod
    def _reference_recurrence(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        decay_logits: torch.Tensor,
        beta_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Numerical fallback matching the KDA state update used by the kernel."""
        batch_size, n_heads, seq_len, head_dim = q.shape
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        alpha = torch.sigmoid(decay_logits)
        beta = torch.sigmoid(beta_logits).unsqueeze(-1)
        state = q.new_zeros(batch_size, n_heads, head_dim, head_dim)
        outputs: list[torch.Tensor] = []

        for token_index in range(seq_len):
            k_t = k[:, :, token_index]
            v_t = v[:, :, token_index]
            # KDA: decay key channels before applying the delta-rule erase/write.
            state = state * alpha[:, :, token_index].unsqueeze(-2)
            predicted_v = torch.einsum("bhvk,bhk->bhv", state, k_t)
            residual = beta[:, :, token_index] * (v_t - predicted_v)
            state = state + residual.unsqueeze(-1) * k_t.unsqueeze(-2)
            outputs.append(torch.einsum("bhvk,bhk->bhv", state, q[:, :, token_index]))

        return torch.stack(outputs, dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        q, k, v = self.qkv_conv(self.qkv_proj(x)).chunk(3, dim=-1)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # QK normalization and RoPE are applied before the chunkwise KDA kernel.
        q = apply_rope(self.q_norm(split_heads(q)))
        k = apply_rope(self.k_norm(split_heads(k)))
        v = split_heads(v)
        decay_logits = split_heads(self.alpha_proj(x))
        beta_logits = self.beta_proj(x).transpose(1, 2)

        can_use_kernel = chunk_kda is not None and x.is_cuda and self.head_dim == 128
        if can_use_kernel:
            y = self._chunk_kernel(q, k, v, decay_logits, beta_logits)
        else:
            y = self._reference_recurrence(q, k, v, decay_logits, beta_logits)

        y = y.transpose(1, 2).contiguous()
        y = y.view(batch_size, seq_len, -1)
        return self.out_proj(y * F.silu(self.gate_proj(x)))


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int) -> None:
        super().__init__()
        self.in_proj = nn.Linear(d_model, 2 * hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.in_proj(x).chunk(2, dim=-1)
        return self.out_proj(value * F.silu(gate))


class KDABlock(nn.Module):
    def __init__(self, config: KDAConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attention = KimiDeltaAttention(config)
        self.ffn_norm = RMSNorm(config.d_model)
        self.ffn = SwiGLU(config.d_model, config.ffn_dim)
        self.residual_scale = (2 * config.n_layers) ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.residual_scale * self.attention(self.attn_norm(x))
        return x + self.residual_scale * self.ffn(self.ffn_norm(x))


class KDALanguageModel(nn.Module):
    def __init__(self, config: KDAConfig = KDAConfig()) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList(KDABlock(config) for _ in range(config.n_layers))
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
        self, input_ids: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.size(1) > self.config.max_seq_len:
            raise ValueError(f"sequence length exceeds {self.config.max_seq_len}")

        x = self.token_embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        logits = self.lm_head(self.final_norm(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
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
