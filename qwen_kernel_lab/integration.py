"""Small, version-tolerant Qwen integration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from types import MethodType
from typing import List

import torch
from torch import nn

from .ops import add_rms_norm_residual, rms_norm, silu_mul


class KernelRMSNorm(nn.Module):
    """Drop-in inference RMSNorm used to validate Qwen model integration."""

    def __init__(self, weight: torch.Tensor, eps: float):
        super().__init__()
        self.weight = nn.Parameter(weight.detach().clone(), requires_grad=False)
        self.variance_epsilon = float(eps)

    @classmethod
    def from_module(cls, module: nn.Module) -> "KernelRMSNorm":
        eps = getattr(module, "variance_epsilon", getattr(module, "eps", 1e-6))
        return cls(module.weight, eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return rms_norm(hidden_states, self.weight, self.variance_epsilon)


@dataclass
class Replacement:
    path: str
    old_class: str


def replace_qwen_rmsnorm(model: nn.Module) -> List[Replacement]:
    """Replace modules named ``*RMSNorm`` without depending on a HF version.

    The replacement validates model-level numerical compatibility. It does not
    claim fusion speedup because the residual addition lives outside Hugging
    Face RMSNorm modules.
    """
    replacements: List[Replacement] = []

    def visit(parent: nn.Module, prefix: str = "") -> None:
        for name, child in list(parent.named_children()):
            path = f"{prefix}.{name}" if prefix else name
            is_rms_norm = child.__class__.__name__.lower().endswith("rmsnorm")
            if is_rms_norm and hasattr(child, "weight"):
                setattr(parent, name, KernelRMSNorm.from_module(child))
                replacements.append(Replacement(path, child.__class__.__name__))
            else:
                visit(child, path)

    visit(model)
    return replacements

def replace_qwen_mlp(model: nn.Module) -> List[Replacement]:
    """Route Qwen gated MLP activations through the fused SiLU-mul operator."""
    replacements: List[Replacement] = []

    for path, module in model.named_modules():
        class_name = module.__class__.__name__
        class_name_lower = class_name.lower()
        required = ("gate_proj", "up_proj", "down_proj")
        is_qwen_mlp = "qwen" in class_name_lower and class_name_lower.endswith("mlp")
        if not is_qwen_mlp or not all(hasattr(module, name) for name in required):
            continue
        if getattr(module, "_qkl_silu_mul_enabled", False):
            continue

        def fused_forward(self: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
            gate = self.gate_proj(hidden_states)
            up = self.up_proj(hidden_states)
            return self.down_proj(silu_mul(gate, up))

        module.forward = MethodType(fused_forward, module)
        module._qkl_silu_mul_enabled = True
        replacements.append(Replacement(path, class_name))

    return replacements

def replace_qwen_decoder_rmsnorm(model: nn.Module) -> List[Replacement]:
    """Fuse attention residual addition with post-attention RMSNorm."""
    replacements: List[Replacement] = []

    for path, module in model.named_modules():
        class_name = module.__class__.__name__
        class_name_lower = class_name.lower()
        required = ("input_layernorm", "post_attention_layernorm", "self_attn", "mlp")
        is_qwen_decoder = (
            "qwen" in class_name_lower and class_name_lower.endswith("decoderlayer")
        )
        if not is_qwen_decoder or not all(
            hasattr(module, name) for name in required
        ):
            continue
        if getattr(module, "_qkl_residual_rmsnorm_enabled", False):
            continue

        module._qkl_original_forward = module.forward

        def fused_forward(
            self: nn.Module,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            position_ids: torch.Tensor | None = None,
            past_key_values=None,
            use_cache: bool | None = False,
            position_embeddings=None,
            **kwargs,
        ) -> torch.Tensor:
            # Decode is latency-sensitive at sequence length 1. Keep the
            # original residual/RMSNorm path there while retaining any MLP
            # operator replacement already installed on the module.
            if hidden_states.shape[-2] == 1:
                return self._qkl_original_forward(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )

            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            norm = self.post_attention_layernorm
            eps = getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-6))
            hidden_states, residual = add_rms_norm_residual(
                hidden_states, residual, norm.weight, eps
            )
            hidden_states = self.mlp(hidden_states)
            return residual + hidden_states

        module.forward = MethodType(fused_forward, module)
        module._qkl_residual_rmsnorm_enabled = True
        replacements.append(Replacement(path, class_name))

    return replacements

