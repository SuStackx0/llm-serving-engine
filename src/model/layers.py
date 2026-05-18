"""
Primitive transformer layers: RMSNorm and SwiGLU MLP.
All operations are expressed in plain torch so they run on MPS/CPU/CUDA.
"""

import math
import torch
import torch.nn.functional as F
from torch import Tensor


def rms_norm(x: Tensor, weight: Tensor, eps: float = 1e-5) -> Tensor:
    """Root-Mean-Square Layer Normalization (no centering).

    RMSNorm(x) = x / sqrt(mean(x²) + ε)  *  weight

    Args:
        x:      [..., hidden_size]
        weight: [hidden_size]  (learned per-dimension scale, γ)
        eps:    numerical stability constant

    Returns:
        Normalized tensor, same shape as x.
    """
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    return x_normed * weight


def swiglu_mlp(
    x: Tensor,
    gate_proj: Tensor,
    up_proj: Tensor,
    down_proj: Tensor,
) -> Tensor:
    """SwiGLU feed-forward block used in TinyLlama / Llama-2.

    output = (SiLU(x @ W_gate) * (x @ W_up)) @ W_down

    Args:
        x:          [T, hidden_size]
        gate_proj:  [intermediate_size, hidden_size]
        up_proj:    [intermediate_size, hidden_size]
        down_proj:  [hidden_size, intermediate_size]

    Returns:
        [T, hidden_size]
    """
    gate = F.silu(x @ gate_proj.T)   # [T, intermediate_size]
    up   = x @ up_proj.T             # [T, intermediate_size]
    return (gate * up) @ down_proj.T  # [T, hidden_size]
