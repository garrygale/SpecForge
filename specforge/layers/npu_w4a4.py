# coding=utf-8
"""Ascend NPU W4A4 inference layer for DFlash/Domino drafts."""

from __future__ import annotations

import warnings
from typing import Optional

import torch
import torch.nn as nn

from .wxay import quantize_weight, stochastic_round


class NPUW4A4Linear(nn.Module):
    """NPU dynamic W4A4 replacement for ``nn.Linear``."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: Optional[torch.Tensor],
        weight_packed: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_offset: torch.Tensor,
        smooth_scale: Optional[torch.Tensor] = None,
        *,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("weight", weight_packed.to(device=device))
        self.register_buffer("weight_scale", weight_scale.to(device=device))
        self.register_buffer("weight_offset", weight_offset.to(device=device))
        if smooth_scale is not None:
            self.register_buffer("smooth_scale", smooth_scale.to(device=device))
        else:
            self.smooth_scale = None
        if bias is not None:
            self.register_buffer("bias", bias.to(device=device))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import torch_npu

        shape = x.shape
        if x.dim() > 2:
            x = x.reshape(-1, x.shape[-1])
        if self.smooth_scale is not None:
            x = x / self.smooth_scale.to(x.dtype)
        x_packed, pertoken_scale = torch_npu.npu_dynamic_quant(
            x, dst_type=torch.quint4x2
        )
        out = torch_npu.npu_quant_matmul(
            x_packed,
            self.weight.t(),
            self.weight_scale,
            pertoken_scale=pertoken_scale,
            bias=self.bias,
            output_dtype=x.dtype,
        )
        if len(shape) > 2:
            out = out.reshape(*shape[:-1], self.out_features)
        return out


def _path_matches(full_path: str, name: str, exclude: set[str]) -> bool:
    segments = set(full_path.split("."))
    return (
        full_path in exclude
        or name in exclude
        or not segments.isdisjoint(exclude)
    )


def replace_linear_with_npu_w4a4(
    module: nn.Module,
    w_bit: int = 4,
    exclude_names: Optional[list[str]] = None,
    include_only: Optional[set[str]] = None,
    stochastic_weight: bool = False,
) -> None:
    """Replace every eligible ``nn.Linear`` with an NPU W4A4 layer."""
    import torch_npu  # noqa: F401

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
                    matched.update(
                        exclude
                        & (set(full_path.split(".")) | {name, full_path})
                    )
                    continue
                if child.in_features % 8 != 0:
                    raise ValueError(
                        "W4A4 requires in_features divisible by 8, got "
                        f"{child.in_features} for layer '{full_path}'"
                    )
                smooth_scale = None
                weight_float = child.weight.data.float()
                log_scale = getattr(child, "log_channel_scale", None)
                if log_scale is not None:
                    scale = torch.exp(log_scale.data.float())
                    smooth_scale = scale.clone()
                    weight_float = weight_float * scale.unsqueeze(0)
                _, w_scale = quantize_weight(
                    weight_float,
                    weight_bit=w_bit,
                    stochastic_weight=stochastic_weight,
                )
                qmax = 2 ** (w_bit - 1) - 1
                w_int4_raw = (
                    stochastic_round(weight_float / w_scale)
                    if stochastic_weight
                    else torch.round(weight_float / w_scale)
                ).clamp(-qmax, qmax).to(torch.int32)
                w_packed = torch_npu.npu_convert_weight_to_int4pack(
                    w_int4_raw.contiguous(), inner_k_tiles=1
                )
                w_scale = w_scale.squeeze(-1)
                w_offset = torch.zeros(
                    child.out_features, dtype=torch.float32, device="cpu"
                )
                bias = (
                    child.bias.data.clone()
                    if child.bias is not None
                    else None
                )
                new = NPUW4A4Linear(
                    child.in_features,
                    child.out_features,
                    bias=bias,
                    weight_packed=w_packed,
                    weight_scale=w_scale.float(),
                    weight_offset=w_offset,
                    smooth_scale=smooth_scale,
                    device=child.weight.device,
                    dtype=child.weight.dtype,
                )
                setattr(mod, name, new)
            else:
                walk(child, full_path)

    walk(module)
    unmatched = exclude - matched
    if unmatched:
        warnings.warn(
            f"qat_exclude names not found in model: {sorted(unmatched)}"
        )


__all__ = ["NPUW4A4Linear", "replace_linear_with_npu_w4a4"]
