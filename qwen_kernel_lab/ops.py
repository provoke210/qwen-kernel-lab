"""Public operators with a correctness-first PyTorch fallback.

Importing the package does not JIT compile native code. Install the extension with
``pip install -e .`` (CPU) or ``QKL_BUILD_CUDA=1 pip install -e .`` (CUDA).
"""

from __future__ import annotations

import importlib
from typing import Optional

import torch
import torch.nn.functional as F


_NATIVE_ERROR: Optional[BaseException] = None
try:
    importlib.import_module("qwen_kernel_lab._C")

    _NATIVE_LOADED = True
except (ImportError, OSError) as exc:
    _NATIVE_LOADED = False
    _NATIVE_ERROR = exc


def native_extension_loaded() -> bool:
    return _NATIVE_LOADED


def native_extension_error() -> Optional[BaseException]:
    return _NATIVE_ERROR


def _validate_rms_inputs(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor
) -> None:
    if x.shape != residual.shape:
        raise ValueError(f"x and residual must match, got {x.shape} and {residual.shape}")
    if x.ndim < 1:
        raise ValueError("x must have at least one dimension")
    if weight.ndim != 1 or weight.numel() != x.shape[-1]:
        raise ValueError(
            f"weight must be 1D with {x.shape[-1]} elements, got {tuple(weight.shape)}"
        )
    if x.device != residual.device or x.device != weight.device:
        raise ValueError("x, residual, and weight must be on the same device")
    if x.dtype != residual.dtype or x.dtype != weight.dtype:
        raise ValueError("x, residual, and weight must have the same dtype")


def add_rms_norm_reference(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Numerically stable reference used as the correctness oracle.

    The variance is accumulated in FP32, matching common inference kernels.
    """
    _validate_rms_inputs(x, residual, weight)
    summed = x + residual
    variance = summed.float().square().mean(dim=-1, keepdim=True)
    normalized = summed.float() * torch.rsqrt(variance + eps)
    return (normalized * weight.float()).to(dtype=x.dtype)


def add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    *,
    use_native: bool = True,
) -> torch.Tensor:
    """Fused residual addition and RMS normalization.

    Native kernels are inference-only. Autograd inputs automatically use the
    PyTorch reference so training behavior remains correct.
    """
    _validate_rms_inputs(x, residual, weight)
    should_use_native = use_native and _NATIVE_LOADED and not any(
        tensor.requires_grad for tensor in (x, residual, weight)
    )
    if should_use_native:
        return torch.ops.qwen_kernel_lab.add_rms_norm(x, residual, weight, eps)
    return add_rms_norm_reference(x, residual, weight, eps)

def add_rms_norm_residual_reference(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return both normalized activations and the updated residual stream."""
    _validate_rms_inputs(x, residual, weight)
    summed = x + residual
    variance = summed.float().square().mean(dim=-1, keepdim=True)
    normalized = summed.float() * torch.rsqrt(variance + eps) * weight.float()
    return normalized.to(dtype=x.dtype), summed


def add_rms_norm_residual(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    *,
    use_native: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse residual addition and RMSNorm while preserving the residual sum."""
    _validate_rms_inputs(x, residual, weight)
    should_use_native = use_native and _NATIVE_LOADED and not any(
        tensor.requires_grad for tensor in (x, residual, weight)
    )
    if should_use_native:
        return torch.ops.qwen_kernel_lab.add_rms_norm_residual(
            x, residual, weight, eps
        )
    return add_rms_norm_residual_reference(x, residual, weight, eps)



def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    *,
    use_native: bool = True,
) -> torch.Tensor:
    """RMSNorm compatibility helper.

    This helper is intended for correctness integration with Hugging Face Qwen.
    The performance path is ``add_rms_norm``, where residual addition is fused.
    """
    residual = torch.zeros_like(x)
    return add_rms_norm(x, residual, weight, eps, use_native=use_native)


def silu_mul_reference(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    if gate.shape != up.shape:
        raise ValueError(f"gate and up must match, got {gate.shape} and {up.shape}")
    if gate.device != up.device or gate.dtype != up.dtype:
        raise ValueError("gate and up must have the same device and dtype")
    return F.silu(gate) * up


def silu_mul(
    gate: torch.Tensor,
    up: torch.Tensor,
    *,
    use_native: bool = True,
) -> torch.Tensor:
    """Fused SiLU(gate) * up used by Qwen-family gated MLPs."""
    if gate.shape != up.shape:
        raise ValueError(f"gate and up must match, got {gate.shape} and {up.shape}")
    if gate.device != up.device or gate.dtype != up.dtype:
        raise ValueError("gate and up must have the same device and dtype")
    should_use_native = use_native and _NATIVE_LOADED and not any(
        tensor.requires_grad for tensor in (gate, up)
    )
    if should_use_native:
        return torch.ops.qwen_kernel_lab.silu_mul(gate, up)
    return silu_mul_reference(gate, up)

