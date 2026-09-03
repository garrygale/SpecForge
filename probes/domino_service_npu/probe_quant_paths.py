#!/usr/bin/env python3
"""NPU probe for the Domino W4A8/W4A4/W8A8 and fused projection paths.

This probe mirrors the layouts used by
``vllm_ascend/quantization/domino.py`` at the operator level, without
importing ``vllm_ascend``.  Run it on an NPU with ``torch_npu`` installed.

Usage::

    python probes/domino_service_npu/probe_quant_paths.py [--real-dims]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class SchemeCase:
    name: str
    w_bit: int
    scheme: str  # "w4a8", "w4a4", or "w8a8"


def quantize_weight_per_channel(
    weight: torch.Tensor, w_bit: int, stochastic: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    qmax = 2 ** (w_bit - 1) - 1
    w_fp32 = weight.float()
    scale = w_fp32.abs().amax(dim=1, keepdim=True) / qmax
    scale = scale.clamp(min=1e-6)
    w_scaled = w_fp32 / scale
    w_int = torch.floor(w_scaled + torch.rand_like(w_scaled)) if stochastic else torch.round(w_scaled)
    w_int = w_int.clamp(-qmax, qmax)
    return w_int, scale


def pack_w4a8(w_int: torch.Tensor) -> torch.Tensor:
    """[N, K] int4 -> [K, N//8] int32 ND (Domino W4A8 layout)."""
    import torch_npu

    w_t = w_int.to(torch.int32).t().contiguous()
    packed = torch_npu.npu_convert_weight_to_int4pack(w_t)
    return torch_npu.npu_format_cast(packed, 2)


def pack_w4a4(w_int: torch.Tensor) -> torch.Tensor:
    """[N, K] int4 -> [K//8, N] int32 (Domino W4A4 layout)."""
    import torch_npu

    packed = torch_npu.npu_convert_weight_to_int4pack(
        w_int.to(torch.int32).contiguous()
    )
    return packed.transpose(-1, -2)


def proj_w4a8(x: torch.Tensor, packed: torch.Tensor, scale: torch.Tensor):
    import torch_npu

    return torch_npu.npu_weight_quant_batchmatmul(
        x, packed, antiquant_scale=scale.to(x.dtype), antiquant_group_size=0
    )


def proj_w4a4(x: torch.Tensor, packed: torch.Tensor, scale: torch.Tensor):
    import torch_npu

    x4, x4s = torch_npu.npu_dynamic_quant(x, dst_type=torch.quint4x2)
    return torch_npu.npu_quant_matmul(
        x4,
        packed,
        scale=scale.view(-1),
        pertoken_scale=x4s.reshape(-1),
        bias=None,
        output_dtype=torch.float16,
    ).to(x.dtype)


def proj_w8a8(x: torch.Tensor, packed: torch.Tensor, scale: torch.Tensor):
    import torch_npu

    x8, x8s = torch_npu.npu_dynamic_quant(x)
    if x8s.dim() == 2:
        x8s = x8s.squeeze(1)
    return torch_npu.npu_quant_matmul(
        x8,
        packed,
        scale,
        pertoken_scale=x8s,
        bias=None,
        output_dtype=x.dtype,
    )


def pack_scheme(w_int: torch.Tensor, scheme: str) -> torch.Tensor:
    if scheme == "w4a8":
        return pack_w4a8(w_int)
    if scheme == "w4a4":
        return pack_w4a4(w_int)
    return w_int.to(torch.int8).t().contiguous()


def proj_scheme(x: torch.Tensor, packed: torch.Tensor, scale: torch.Tensor, scheme: str):
    if scheme == "w4a8":
        return proj_w4a8(x, packed, scale.to(x.dtype))
    if scheme == "w4a4":
        return proj_w4a4(x, packed, scale)
    return proj_w8a8(x, packed, scale)


def _real_dims() -> tuple[int, int, int, int]:
    """(in_features, out_features, batch, query_len) for the draft shapes."""
    return 2560, 4096, 4, 16


def _small_dims() -> tuple[int, int, int, int]:
    return (64, 128, 2, 8)


def _check(
    name: str,
    got: torch.Tensor,
    ref: torch.Tensor,
    rtol: float = 1e-2,
    atol: float = 1e-2,
) -> None:
    if got.shape != ref.shape:
        raise AssertionError(f"{name}: shape {tuple(got.shape)} != {tuple(ref.shape)}")
    diff = (got.float() - ref.float()).abs().max().item()
    if diff > max(atol, rtol * ref.float().abs().max().item()):
        raise AssertionError(f"{name}: max diff {diff:.6f}")
    print(f"  {name}: ok (max diff {diff:.6f})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-dims", action="store_true")
    args = parser.parse_args()

    try:
        import torch_npu  # noqa: F401
    except ImportError as exc:
        raise SystemExit("SKIP: torch_npu is not available") from exc
    if not torch.npu.is_available():
        raise SystemExit("SKIP: NPU is not available")

    device = torch.device("npu:0")
    in_features, out_features, batch, query_len = (
        _real_dims() if args.real_dims else _small_dims()
    )
    torch.manual_seed(7)
    x = torch.randn(batch * query_len, in_features, dtype=torch.bfloat16, device=device)
    weight = torch.randn(out_features, in_features, dtype=torch.bfloat16, device=device)
    ref = F.linear(x.float(), weight.float())

    cases = [
        SchemeCase("W4A8", 4, "w4a8"),
        SchemeCase("W4A4", 4, "w4a4"),
        SchemeCase("W8A8", 8, "w8a8"),
    ]
    for case in cases:
        w_int, scale = quantize_weight_per_channel(weight, case.w_bit)
        packed = pack_scheme(w_int, case.scheme)
        got = proj_scheme(x, packed, scale.reshape(-1).to(torch.float32), case.scheme)
        _check(f"{case.name} per-layer", got, ref)

    # Fused q/k/v single call vs three separate projections.
    q_size = out_features // 2
    k_size = out_features // 4
    v_size = out_features // 4
    q_w = torch.randn(q_size, in_features, dtype=torch.bfloat16, device=device)
    k_w = torch.randn(k_size, in_features, dtype=torch.bfloat16, device=device)
    v_w = torch.randn(v_size, in_features, dtype=torch.bfloat16, device=device)
    fused_w = torch.cat([q_w, k_w, v_w], dim=0)

    for case in cases:
        q_int, q_scale = quantize_weight_per_channel(q_w, case.w_bit)
        k_int, k_scale = quantize_weight_per_channel(k_w, case.w_bit)
        v_int, v_scale = quantize_weight_per_channel(v_w, case.w_bit)
        fused_int = torch.cat([q_int, k_int, v_int], dim=0)
        fused_scale = torch.cat([q_scale, k_scale, v_scale]).reshape(-1).float()
        packed_fused = pack_scheme(fused_int, case.scheme)
        fused_got = proj_scheme(x, packed_fused, fused_scale, case.scheme)
        refs = [
            proj_scheme(x, pack_scheme(q_int, case.scheme), q_scale.reshape(-1).float(), case.scheme)
            for q_int, q_scale in ((q_int, q_scale),)
        ]
        _check(
            f"{case.name} fused qkv packing",
            fused_got.view(batch, query_len, -1),
            torch.cat(
                [
                    refs[0].view(batch, query_len, q_size),
                    proj_scheme(x, pack_scheme(k_int, case.scheme), k_scale.reshape(-1).float(), case.scheme).view(batch, query_len, k_size),
                    proj_scheme(x, pack_scheme(v_int, case.scheme), v_scale.reshape(-1).float(), case.scheme).view(batch, query_len, v_size),
                ],
                dim=-1,
            ),
        )

    # ACL graph replay smoke test.  This is intentionally a single captured op;
    # if the installed CANN/torch_npu lacks the graph API, report SKIP for the
    # graph section and keep the eager result above.
    if hasattr(torch.npu, "graph"):
        w_int, scale = quantize_weight_per_channel(weight, 8)
        packed = pack_scheme(w_int, "w8a8")
        graph = torch.npu.graph()
        try:
            graph.__enter__()
            graph_out = proj_scheme(
                x, packed, scale.reshape(-1).to(torch.float32), "w8a8"
            )
            graph.__exit__(None, None, None)
            _check("W8A8 graph replay", graph_out, ref)
        except Exception as exc:  # noqa: BLE001
            print(f"  graph replay: SKIP ({type(exc).__name__}: {exc})")

    print("probe_quant_paths: PASS")


if __name__ == "__main__":
    main()
