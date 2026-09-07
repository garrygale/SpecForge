#!/usr/bin/env python3
"""NPU probe for Domino's non-causal sliding-window FIA band mode.

The service uses ``sparse_mode=4`` with ``pre_tokens == next_tokens == W``
for the Domino draft's trained block-bidirectional sliding layers.  This
probe runs that call at one or two window sizes so a missing/unsupported
band-mode shape reports early, and it checks that the output stays finite.

Use --compare to sweep W=2048 vs W=3072 (the two values that separated
healthy from corrupted service runs), with token lengths well past 2048 so
the band has real space on both sides.
"""

from __future__ import annotations

import argparse

import torch


def run_case(
    *,
    window: int,
    tokens: int,
    batch: int,
    heads: int,
    head_dim: int,
    iterations: int,
) -> None:
    import torch_npu  # noqa: F401

    torch.npu.config.allow_internal_format = True
    q = torch.randn(
        batch * tokens, heads, head_dim, dtype=torch.bfloat16
    ).npu()
    k = q.clone().contiguous()
    v = q.clone().contiguous()
    attn_mask = torch.zeros((2048, 2048), dtype=torch.int8).npu()
    # PrefillNoCache service path uses contiguous TND K/V and no block table;
    # this is the minimal shape that exercises the sparse-mode=4 band setting.
    block_table = None
    actual_q = torch.tensor([tokens] * batch, dtype=torch.int32).npu()
    actual_kv = actual_q.clone()

    attn_out = None
    for _ in range(iterations):
        attn_out, _ = torch_npu.npu_fused_infer_attention_score(
            query=q,
            key=k,
            value=v,
            atten_mask=attn_mask,
            input_layout="TND",
            block_size=128,
            actual_seq_lengths=actual_q,
            actual_seq_lengths_kv=actual_kv,
            num_key_value_heads=heads,
            num_heads=heads,
            scale=head_dim**-0.5,
            pre_tokens=window,
            next_tokens=window,
            sparse_mode=4,
        )
    torch.npu.synchronize()

    finite = bool(torch.isfinite(attn_out).all().item())
    max_abs = float(attn_out.abs().max().item())
    print(
        f"window={window}: ok finite={finite} "
        f"max_abs={max_abs:.6g}"
    )
    if not finite:
        raise SystemExit(f"FAIL: non-finite FIA output at window={window}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=256)
    parser.add_argument("--window2", type=int, default=3072)
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args()

    windows = [args.window]
    if args.compare:
        windows.append(args.window2)
    if any(args.tokens <= w for w in windows):
        raise SystemExit(
            f"FAIL: --tokens {args.tokens} must be greater than every "
            f"--window value {windows} so the FIA band has space on both sides"
        )

    try:
        import torch_npu  # noqa: F401
    except ImportError as exc:
        raise SystemExit("SKIP: torch_npu is not available") from exc
    if not torch.npu.is_available():
        raise SystemExit("SKIP: NPU is not available")

    for window in windows:
        try:
            run_case(
                window=window,
                tokens=args.tokens,
                batch=args.batch,
                heads=args.heads,
                head_dim=args.head_dim,
                iterations=args.iterations,
            )
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"FAIL: {type(exc).__name__}: {exc}") from exc
    print("non-causal sliding band FIA call: ok")


if __name__ == "__main__":
    main()
