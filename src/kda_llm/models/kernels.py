"""Optional CUDA kernels and safe compiler boundaries."""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    from fla.ops.kda import chunk_kda
except ImportError:
    chunk_kda = None

try:
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
except ImportError:
    LigerFusedLinearCrossEntropyLoss = None

try:
    from liger_kernel.ops.rms_norm import LigerRMSNormFunction
    from liger_kernel.ops.swiglu import LigerSiLUMulFunction
except ImportError:
    LigerRMSNormFunction = None
    LigerSiLUMulFunction = None


@torch.compiler.disable
def fused_linear_cross_entropy(
    loss_module: nn.Module, weight: torch.Tensor, hidden_states: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    return loss_module(weight, hidden_states, targets)


@torch.compiler.disable
def liger_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return LigerRMSNormFunction.apply(x, weight, eps)


@torch.compiler.disable
def liger_silu_mul(gate: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    return LigerSiLUMulFunction.apply(gate, value)
