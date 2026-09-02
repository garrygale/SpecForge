# coding=utf-8
"""Generalized weight + activation quantization QAT layer for DFlash drafts.

This module is the single source of truth for the per-channel fake-quant math
used by training, NPU evaluation, and on-the-fly NPU quantization. The
quantized layer is a drop-in replacement for ``nn.Linear``; state-dict keys
remain identical when ``channel_balanced`` is disabled.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def stochastic_round(x: torch.Tensor) -> torch.Tensor:
    return torch.floor(x + torch.rand_like(x))


def quantize_weight(
    weight: torch.Tensor,
    weight_bit: int = 4,
    stochastic_weight: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-channel fake-quantization.

    Returns the dequantized weight and the per-channel scale.
    """
    qmax = 2 ** (weight_bit - 1) - 1
    scale = weight.abs().amax(dim=1, keepdim=True) / qmax
    scale = scale.clamp(min=1e-6)
    w_scaled = weight / scale
    w_int = (
        stochastic_round(w_scaled)
        if stochastic_weight
        else torch.round(w_scaled)
    ).clamp(-qmax, qmax)
    return w_int * scale, scale


def quantize_activation(
    x: torch.Tensor,
    act_bit: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-token symmetric fake-quantization for activations."""
    qmax = 2 ** (act_bit - 1) - 1
    scale = x.abs().detach().amax(dim=-1, keepdim=True) / qmax
    scale = scale.clamp(min=1e-6)
    x_int = torch.round(x / scale).clamp(-qmax, qmax)
    return x_int * scale, scale


def pack_int4(w_int4: torch.Tensor) -> torch.Tensor:
    assert w_int4.shape[-1] % 2 == 0, "in_features must be even"
    w_uint4 = (w_int4 + 8).to(torch.uint8)
    low = w_uint4[..., 0::2]
    high = w_uint4[..., 1::2]
    return low | (high << 4)


def unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    stacked = torch.stack([low, high], dim=-1)
    flat = stacked.flatten(start_dim=-2)
    return flat.to(torch.int8) - 8


class QuantizedLinear(nn.Module):
    """Drop-in QAT replacement for ``nn.Linear`` with STE everywhere."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        w_bit: int = 4,
        a_bit: Optional[int] = None,
        channel_balanced: bool = False,
        stochastic_weight: bool = False,
        *,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.channel_balanced = channel_balanced
        self.stochastic_weight = stochastic_weight
        if channel_balanced and a_bit is None:
            raise ValueError(
                "Channel balancing requires activation quantization"
            )

        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, **factory_kwargs)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        if channel_balanced:
            self.log_channel_scale = nn.Parameter(
                torch.zeros(in_features, **factory_kwargs)
            )
        else:
            self.log_channel_scale = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = None
        if self.log_channel_scale is not None:
            scale = torch.exp(self.log_channel_scale)
            x = x / scale
        if self.a_bit is not None:
            x_q, _ = quantize_activation(x, self.a_bit)
            x = x + (x_q - x).detach()
        weight = self.weight
        if scale is not None:
            weight = weight * scale.unsqueeze(0)
        weight_deq, _ = quantize_weight(
            weight,
            self.w_bit,
            stochastic_weight=self.stochastic_weight,
        )
        weight_deq = weight + (weight_deq - weight).detach()
        return F.linear(x, weight_deq, self.bias)


def _path_matches(full_path: str, name: str, exclude: set[str]) -> bool:
    segments = set(full_path.split("."))
    return (
        full_path in exclude
        or name in exclude
        or not segments.isdisjoint(exclude)
    )


def replace_linear_with_quantized(
    module: nn.Module,
    w_bit: int = 4,
    a_bit: Optional[int] = None,
    channel_balanced: bool = False,
    exclude_names: Optional[list[str]] = None,
    include_only: Optional[set[str]] = None,
    stochastic_weight: bool = False,
) -> None:
    """Recursively replace ``nn.Linear`` modules in-place with QAT layers."""
    import warnings

    matched: set[str] = set()
    exclude = set(exclude_names or [])

    def walk(mod: nn.Module, path_prefix: str = "") -> None:
        for name, child in mod.named_children():
            full_path = f"{path_prefix}.{name}" if path_prefix else name
            if isinstance(child, nn.Linear):
                if include_only is not None and full_path not in include_only:
                    walk(child, full_path)
                    continue
                if exclude and _path_matches(full_path, name, exclude):
                    matched.update(exclude & (set(full_path.split(".")) | {name, full_path}))
                    continue
                new = QuantizedLinear(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    w_bit=w_bit,
                    a_bit=a_bit,
                    channel_balanced=channel_balanced,
                    stochastic_weight=stochastic_weight,
                    device=child.weight.device,
                    dtype=child.weight.dtype,
                )
                new.weight.data.copy_(child.weight.data)
                if child.bias is not None:
                    new.bias.data.copy_(child.bias.data)
                setattr(mod, name, new)
            else:
                walk(child, full_path)

    walk(module)
    unmatched = exclude - matched
    if unmatched:
        warnings.warn(
            f"qat_exclude names not found in model: {sorted(unmatched)}"
        )


__all__ = [
    "QuantizedLinear",
    "pack_int4",
    "quantize_activation",
    "quantize_weight",
    "replace_linear_with_quantized",
    "stochastic_round",
    "unpack_int4",
]
