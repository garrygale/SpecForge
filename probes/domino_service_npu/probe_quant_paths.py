#!/usr/bin/env python3
"""NPU probe for the Domino W4A8/W4A4/W8A8 fused QKV path.

This follows the validated ``vllm-ascend/benchmarks/probe_qkv_fusion.py``
structure: it compares fused single-pack projections against the separate
per-layer projections (the service invariant), rather than against an fp32
reference that includes quantization rounding.  It also captures the fused
path with ``torch.npu.NPUGraph`` and checks replay parity.

Usage::

    python probes/domino_service_npu/probe_quant_paths.py [--real-dims]
"""

from __future__ import annotations

import argparse

import torch
import torch_npu

ACL_FORMAT_ND = 2
TOLERANCE = 0.05


def _dims(real: bool) -> tuple[int, int, int, int, int]:
    # (K, NQ, NKV, D, M)
    if real:
        return 2560, 4096, 1024, 7, 28
    return 64, 64, 16, 3, 7


def _pack_w4a8(w_int: torch.Tensor) -> torch.Tensor:
    """[N, K] int4 -> [K, N//8] int32 ND (Domino W4A8 layout)."""
    packed = torch_npu.npu_convert_weight_to_int4pack(
        w_int.to(torch.int32).t().contiguous()
    )
    return torch_npu.npu_format_cast(packed, ACL_FORMAT_ND)


def _pack_w4a4(w_int: torch.Tensor) -> torch.Tensor:
    """[N, K] int4 -> [K//8, N] int32 (Domino W4A4 layout)."""
    return torch_npu.npu_convert_weight_to_int4pack(
        w_int.to(torch.int32).contiguous()
    ).transpose(-1, -2)


def _build_case(scheme: str, k: int, nq: int, nkv: int, d: int):
    sep = []
    fused = None
    if scheme == "bf16":
        for _ in range(d):
            w = torch.randn(nq + 2 * nkv, k, dtype=torch.bfloat16, device="npu")
            sep.append({"q": w[:nq], "k": w[nq : nq + nkv], "v": w[nq + nkv :]})
        return sep, None

    fused = []
    for _ in range(d):
        w_int = torch.randint(-7, 8, (nq + 2 * nkv, k), dtype=torch.int32, device="npu")
        scale = torch.rand(nq + 2 * nkv, device="npu") * 0.9 + 0.1
        if scheme == "w4a8":
            sep.append(
                {
                    "q": (_pack_w4a8(w_int[:nq]), scale[:nq].to(torch.bfloat16)),
                    "k": (_pack_w4a8(w_int[nq : nq + nkv]), scale[nq : nq + nkv].to(torch.bfloat16)),
                    "v": (_pack_w4a8(w_int[nq + nkv :]), scale[nq + nkv :].to(torch.bfloat16)),
                }
            )
            fused.append((_pack_w4a8(w_int), scale.to(torch.bfloat16)))
        elif scheme == "w4a4":
            sep.append(
                {
                    "q": (_pack_w4a4(w_int[:nq]), scale[:nq]),
                    "k": (_pack_w4a4(w_int[nq : nq + nkv]), scale[nq : nq + nkv]),
                    "v": (_pack_w4a4(w_int[nq + nkv :]), scale[nq + nkv :]),
                }
            )
            fused.append((_pack_w4a4(w_int), scale))
        else:
            sep.append(
                {
                    "q": (w_int[:nq].to(torch.int8).t().contiguous(), scale[:nq]),
                    "k": (w_int[nq : nq + nkv].to(torch.int8).t().contiguous(), scale[nq : nq + nkv]),
                    "v": (w_int[nq + nkv :].to(torch.int8).t().contiguous(), scale[nq + nkv :]),
                }
            )
            fused.append((w_int.to(torch.int8).t().contiguous(), scale))
    return sep, fused


def _proj_bf16(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.linear(x, w)


def _proj_w4a8(x: torch.Tensor, packed: torch.Tensor, scale: torch.Tensor):
    return torch_npu.npu_weight_quant_batchmatmul(
        x,
        packed,
        antiquant_scale=scale,
        antiquant_group_size=0,
    )


def _proj_w4a4(x: torch.Tensor, packed: torch.Tensor, scale: torch.Tensor):
    x4, x4s = torch_npu.npu_dynamic_quant(x, dst_type=torch.quint4x2)
    return torch_npu.npu_quant_matmul(
        x4,
        packed,
        scale=scale.view(-1),
        pertoken_scale=x4s.reshape(-1),
        bias=None,
        output_dtype=torch.float16,
    ).to(x.dtype)


def _proj_w8a8(x: torch.Tensor, w8: torch.Tensor, scale: torch.Tensor):
    x8, x8s = torch_npu.npu_dynamic_quant(x)
    if x8s.dim() == 2:
        x8s = x8s.squeeze(1)
    return torch_npu.npu_quant_matmul(
        x8,
        w8,
        scale,
        pertoken_scale=x8s,
        bias=None,
        output_dtype=x.dtype,
    )


def _run_sep(
    scheme: str,
    x: torch.Tensor,
    sep,
) -> torch.Tensor:
    outs = []
    for layer in sep:
        if scheme == "bf16":
            q = _proj_bf16(x, layer["q"])
            k = _proj_bf16(x, layer["k"])
            v = _proj_bf16(x, layer["v"])
        elif scheme == "w4a8":
            q = _proj_w4a8(x, *layer["q"])
            k = _proj_w4a8(x, *layer["k"])
            v = _proj_w4a8(x, *layer["v"])
        elif scheme == "w4a4":
            q = _proj_w4a4(x, *layer["q"])
            k = _proj_w4a4(x, *layer["k"])
            v = _proj_w4a4(x, *layer["v"])
        else:
            q = _proj_w8a8(x, *layer["q"])
            k = _proj_w8a8(x, *layer["k"])
            v = _proj_w8a8(x, *layer["v"])
        outs.append(torch.cat([q, k, v], dim=-1))
    return torch.stack(outs)


def _run_fused(scheme: str, x: torch.Tensor, fused):
    outs = []
    for layer in fused:
        if scheme == "bf16":
            outs.append(_proj_bf16(x, layer))
        elif scheme == "w4a8":
            outs.append(_proj_w4a8(x, *layer))
        elif scheme == "w4a4":
            outs.append(_proj_w4a4(x, *layer))
        else:
            outs.append(_proj_w8a8(x, *layer))
    return torch.stack(outs)


def _run_mixed_sep(x: torch.Tensor, sep_w4a4, sep_w4a8):
    outs = []
    for i, (sep4, sep8) in enumerate(zip(sep_w4a4, sep_w4a8, strict=True)):
        layer = sep4 if i == 0 else sep8
        if i == 0:
            q = _proj_w4a4(x, *layer["q"])
            k = _proj_w4a4(x, *layer["k"])
            v = _proj_w4a4(x, *layer["v"])
        else:
            q = _proj_w4a8(x, *layer["q"])
            k = _proj_w4a8(x, *layer["k"])
            v = _proj_w4a8(x, *layer["v"])
        outs.append(torch.cat([q, k, v], dim=-1))
    return torch.stack(outs)


def _run_mixed_fused(x: torch.Tensor, fused_w4a4, fused_w4a8):
    outs = []
    for i, (f4, f8) in enumerate(zip(fused_w4a4, fused_w4a8, strict=True)):
        layer = f4 if i == 0 else f8
        if i == 0:
            outs.append(_proj_w4a4(x, *layer))
        else:
            outs.append(_proj_w4a8(x, *layer))
    return torch.stack(outs)


def _check(name: str, out_sep: torch.Tensor, out_fused: torch.Tensor) -> bool:
    err = (out_sep.float() - out_fused.float()).abs().max().item()
    ok = err <= TOLERANCE
    print(f"{name:18s} max_err={err:.6f} {'OK' if ok else 'FAIL'}", flush=True)
    return ok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-dims", action="store_true")
    args = parser.parse_args()

    torch.npu.config.allow_internal_format = True
    print("allow_internal_format=True (service-like)")
    k, nq, nkv, d, m = _dims(args.real_dims)
    nqkv = nq + 2 * nkv
    print(f"K={k} NQ={nq} NKV={nkv} NQKV={nqkv} D={d} M={m}")
    torch.manual_seed(0)

    sep_bf16, fused_bf16_weights = _build_case("bf16", k, nq, nkv, d)
    sep_w4a8, fused_w4a8 = _build_case("w4a8", k, nq, nkv, d)
    sep_w4a4, fused_w4a4 = _build_case("w4a4", k, nq, nkv, d)
    sep_w8a8, fused_w8a8 = _build_case("w8a8", k, nq, nkv, d)

    x = torch.randn(m, k, dtype=torch.bfloat16, device="npu")
    all_ok = True

    def _bf16_fused_weights():
        return [
            torch.cat(
                [sep_bf16[i]["q"], sep_bf16[i]["k"], sep_bf16[i]["v"]],
                dim=0,
            )
            for i in range(d)
        ]

    all_ok &= _check(
        "bf16",
        _run_sep("bf16", x, sep_bf16),
        _run_fused("bf16", x, _bf16_fused_weights()),
    )
    all_ok &= _check("w4a8", _run_sep("w4a8", x, sep_w4a8), _run_fused("w4a8", x, fused_w4a8))
    all_ok &= _check("w4a4", _run_sep("w4a4", x, sep_w4a4), _run_fused("w4a4", x, fused_w4a4))
    all_ok &= _check("w8a8", _run_sep("w8a8", x, sep_w8a8), _run_fused("w8a8", x, fused_w8a8))
    all_ok &= _check(
        "mixed",
        _run_mixed_sep(x, sep_w4a4, sep_w4a8),
        _run_mixed_fused(x, fused_w4a4, fused_w4a8),
    )

    # ACL graph replay parity for each fused scheme.
    for scheme, fused in (
        ("bf16", _bf16_fused_weights()),
        ("w4a8", fused_w4a8),
        ("w4a4", fused_w4a4),
        ("w8a8", fused_w8a8),
    ):
        try:
            graph = torch.npu.NPUGraph()
            stream = torch.npu.Stream()
            with torch.npu.graph(graph, stream=stream, capture_error_mode="global"):
                graph_out = _run_fused(scheme, x, fused)
            graph.replay()
            torch.npu.synchronize()
            eager_out = _run_fused(scheme, x, fused)
            replay_err = (graph_out.float() - eager_out.float()).abs().max().item()
            ok = replay_err == 0.0
            print(
                f"{scheme} graph replay err={replay_err:.6f} "
                f"{'OK' if ok else 'FAIL'}",
                flush=True,
            )
            all_ok &= ok
        except Exception as exc:  # noqa: BLE001
            print(f"{scheme} graph replay FAIL {type(exc).__name__}: {exc}", flush=True)
            all_ok = False

    print("RESULT:", "PASS" if all_ok else "FAIL")
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
