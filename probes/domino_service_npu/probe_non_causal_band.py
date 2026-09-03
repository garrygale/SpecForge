#!/usr/bin/env python3
"""NPU smoke probe for Domino's non-causal sliding-window FIA band mode.

The service uses ``sparse_mode=4`` with ``pre_tokens == next_tokens == W``
for the Domino draft's trained block-bidirectional sliding layers.  This
probe issues one small TND call so a missing/unsupported band-mode shape
reports early.
"""

from __future__ import annotations

import argparse

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=256)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    args = parser.parse_args()

    try:
        import torch_npu  # noqa: F401
    except ImportError as exc:
        raise SystemExit("SKIP: torch_npu is not available") from exc
    if not torch.npu.is_available():
        raise SystemExit("SKIP: NPU is not available")

    torch.npu.config.allow_internal_format = True
    q = torch.randn(
        args.batch * args.tokens, args.heads, args.head_dim, dtype=torch.bfloat16
    ).npu()
    k = q.clone().contiguous()
    v = q.clone().contiguous()
    attn_mask = torch.zeros((2048, 2048), dtype=torch.int8).npu()
    # PrefillNoCache service path uses contiguous TND K/V and no block table;
    # this is the minimal shape that exercises the sparse-mode=4 band setting.
    block_table = None
    actual_q = torch.tensor([args.tokens] * args.batch, dtype=torch.int32).npu()
    actual_kv = actual_q.clone()

    try:
        torch_npu.npu_fused_infer_attention_score(
            query=q,
            key=k,
            value=v,
            atten_mask=attn_mask,
            input_layout="TND",
            block_size=128,
            actual_seq_lengths=actual_q,
            actual_seq_lengths_kv=actual_kv,
            num_key_value_heads=args.heads,
            num_heads=args.heads,
            scale=args.head_dim**-0.5,
            pre_tokens=args.window,
            next_tokens=args.window,
            sparse_mode=4,
        )
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"FAIL: {type(exc).__name__}: {exc}") from exc
    torch.npu.synchronize()
    print("non-causal sliding band FIA call: ok")


if __name__ == "__main__":
    main()
