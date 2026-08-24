"""Reusable neural-network layers for the KDA model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .kernels import LigerRMSNormFunction, LigerSiLUMulFunction, liger_rms_norm, liger_silu_mul


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda and LigerRMSNormFunction is not None:
            return liger_rms_norm(x, self.weight, self.eps)
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


class RotaryCache:
    """Share RoPE tables across layers for a fixed device and dtype."""

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
    _, seq_len, _, head_dim = x.shape
    if head_dim % 2:
        raise ValueError("RoPE requires an even head dimension")
    cos, sin = rope_cache.get(position_offset + seq_len, head_dim, x.device, x.dtype)
    cos, sin = cos[position_offset:][None, :, None], sin[position_offset:][None, :, None]
    x_even, x_odd = x[..., ::2], x[..., 1::2]
    return torch.stack((x_even * cos - x_odd * sin, x_even * sin + x_odd * cos), dim=-1).flatten(-2)


class CausalDepthwiseConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 4) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(channels, channels, kernel_size, groups=channels, bias=True)

    def forward(
        self, x: torch.Tensor, state: torch.Tensor | None = None, use_cache: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = x.transpose(1, 2)
        convolved = self.conv(F.pad(x, (self.kernel_size - 1, 0))) if state is None else self.conv(torch.cat((state, x), dim=-1))
        next_state = None
        if use_cache:
            combined = x if state is None else torch.cat((state, x), dim=-1)
            next_state = combined[..., -(self.kernel_size - 1) :].detach()
        return convolved.transpose(1, 2), next_state


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int) -> None:
        super().__init__()
        self.in_proj = nn.Linear(d_model, 2 * hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.in_proj(x).chunk(2, dim=-1)
        activated = liger_silu_mul(gate, value) if x.is_cuda and LigerSiLUMulFunction is not None else value * F.silu(gate)
        return self.out_proj(activated)
