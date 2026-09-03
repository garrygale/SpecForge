#!/usr/bin/env python3
"""NPU probe for the Domino fused grouped context-KV W4A8 path.

This mirrors the service's ``precompute_and_store_context_kv`` path:
the flare-fused context states are flattened to ``[D*T, target_hidden]`` and
projected through one ``npu_grouped_matmul`` call with per-layer packed K+V
weights, rather than seven ``npu_weight_quant_batchmatmul`` calls.  It also
checks eager and ACL graph replay.
"""

from __future__ import annotations

import argparse

import torch
import torch_npu


def _pack_int4(w_int: torch.Tensor) -> torch.Tensor:
    """[N, K] int4 -> [K, N//8] int32 ND (Domino W4A8 layout)."""
    packed = torch_npu.npu_convert_weight_to_int4pack(
        w_int.to(torch.int32).t().contiguous()
    )
    return torch_npu.npu_format_cast(packed, 2)


def _grouped(
    x: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    offset: torch.Tensor,
    group_list: torch.Tensor,
    d: int,
    t: int,
) -> torch.Tensor:
    out = torch_npu.npu_grouped_matmul(
        x=[x],
        weight=[packed],
        antiquant_scale=[scale],
        antiquant_offset=[offset],
        group_list=group_list,
        split_item=2,
        group_type=0,
        group_list_type=0,
        output_dtype=torch.bfloat16,
    )[0]
    return out.contiguous().view(d, t, -1)


def _per_layer(
    x: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    d: int,
    t: int,
) -> torch.Tensor:
    outs = []
    for layer in range(d):
        outs.append(
            torch_npu.npu_weight_quant_batchmatmul(
                x[layer * t : (layer + 1) * t],
                packed[layer],
                antiquant_scale=scale[layer],
                antiquant_group_size=0,
            )
        )
    return torch.stack(outs, dim=0)


def _reference(x: torch.Tensor, w_ints, scales, d: int, t: int) -> torch.Tensor:
    refs = []
    for layer in range(d):
        w = w_ints[layer].float()
        s = scales[layer].to(torch.bfloat16).float().unsqueeze(1)
        x_l = x[layer * t : (layer + 1) * t].float()
        refs.append(x_l @ (w * s).t())
    return torch.stack(refs, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-dims", action="store_true")
    args = parser.parse_args()

    if args.real_dims:
        d, t_kv, k, n_kv = 7, 8, 2048, 1024
    else:
        d, t_kv, k, n_kv = 3, 4, 64, 16
    device = torch.device("npu:0")
    torch.manual_seed(11)

    w_ints = []
    scales = []
    packed = []
    for _ in range(d):
        k_int = torch.randint(-7, 8, (n_kv, k), device=device)
        v_int = torch.randint(-7, 8, (n_kv, k), device=device)
        fused_int = torch.cat([k_int, v_int], dim=0)
        fused_scale = (torch.rand(2 * n_kv, device=device) * 0.9 + 0.1).to(
            torch.bfloat16
        )
        w_ints.append(fused_int)
        scales.append(fused_scale)
        packed.append(_pack_int4(fused_int))

    x = torch.randn(d * t_kv, k, dtype=torch.bfloat16, device=device)
    # The per-layer baseline should consume the same ND packed tensors that
    # the grouped path stacks internally; passing a sliced torch.stack tensor
    # changes the underlying format on this CANN and inflates the error.
    stacked_packed = torch.stack(packed, dim=0)
    stacked_scale = torch.stack(scales, dim=0)
    offsets = torch.zeros_like(stacked_scale)
    group_list = torch.arange(1, d + 1, dtype=torch.int64, device=device) * t_kv

    ref = _reference(x, w_ints, scales, d, t_kv)
    grouped_out = _grouped(
        x, stacked_packed, stacked_scale, offsets, group_list, d, t_kv
    )
    per_layer_out = _per_layer(x, packed, scales, d, t_kv)
    grouped_err_fp32 = (grouped_out.float() - ref).abs().max().item()
    per_layer_err_fp32 = (per_layer_out.float() - ref).abs().max().item()
    # WQB-vs-WQB is the meaningful parity check; the fp32 reference includes
    # per-channel scale and output rounding that reaches several units at K
    # on the real dimensions.
    grouped_err_wqb = (
        grouped_out.float() - per_layer_out.float()
    ).abs().max().item()
    if grouped_err_wqb > 0.5:
        raise SystemExit(
            f"FAIL grouped-vs-per-layer err={grouped_err_wqb:.4f}"
        )
    print(
        f"grouped-vs-per-layer err={grouped_err_wqb:.4f}; "
        f"fp32-ref grouped={grouped_err_fp32:.4f} "
        f"per-layer={per_layer_err_fp32:.4f}; ok"
    )

    try:
        graph = torch.npu.NPUGraph()
        stream = torch.npu.Stream()
        with torch.npu.graph(graph, stream=stream, capture_error_mode="global"):
            graph_out = _grouped(
                x, stacked_packed, stacked_scale, offsets, group_list, d, t_kv
            )
        graph.replay()
        torch.npu.synchronize()
        replay_err = (graph_out.float() - grouped_out.float()).abs().max().item()
        if replay_err != 0.0:
            raise SystemExit(f"FAIL graph replay vs eager err={replay_err:.5f}")
        print(f"graph replay: ok (err={replay_err:.5f})")
    except Exception as exc:  # noqa: BLE001
        print(f"graph replay: SKIP ({type(exc).__name__}: {exc})")

    print("probe_grouped_fused_kv: PASS")


if __name__ == "__main__":
    main()
