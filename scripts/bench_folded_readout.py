"""Wall-clock check for the folded softmax readout against a dense down_proj.

FLOPs are not time.  The folded readout replaces one big GEMM (M -> N) with a
tiny normalized mixture plus a smaller GEMM (M/c -> N), so the saving only
materializes if the mixture and the smaller GEMM together beat the dense GEMM
on the target device.  At small token counts both are weight-bandwidth bound,
which is what a drafting step looks like; large token counts check the
compute-bound regime (prefill / training chunks).

Run it on the machine that matters, e.g.

    python scripts/bench_folded_readout.py --device cuda
    python scripts/bench_folded_readout.py --device npu --tokens 1,16,128

Dims default to configs/qwen3.6-35b-a3b-domino-dflare-verifiedBase.json
(hidden 2560, intermediate 9728 -> 4 branches, folded width 2432).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn

from specforge.modeling.draft.dflash_kernels import FoldedSoftmaxReadout

DEFAULT_CONFIG = "configs/qwen3.6-35b-a3b-domino-dflare-verifiedBase.json"


def _dims(config_path: str) -> tuple:
    raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
    return int(raw["hidden_size"]), int(raw["intermediate_size"])


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "npu":  # torch_npu uses the same API surface
        torch.npu.synchronize()


def _time(
    module: nn.Module,
    tokens: int,
    input_size: int,
    dtype: torch.dtype,
    device: torch.device,
    iters: int,
    warmup: int,
    backward: bool,
) -> float:
    inputs = torch.randn(
        tokens, input_size, device=device, dtype=dtype, requires_grad=backward
    )

    def step() -> None:
        out = module(inputs)
        if backward:
            out.sum().backward()
            module.zero_grad(set_to_none=True)

    for _ in range(warmup):
        step()
    _sync(device)
    start = time.perf_counter()
    for _ in range(iters):
        step()
    _sync(device)
    return (time.perf_counter() - start) / max(1, iters)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--intermediate-size", type=int, default=None)
    parser.add_argument("--branches", type=int, default=4)
    parser.add_argument("--granularity", type=int, default=16)
    parser.add_argument("--tokens", default="1,16,128,1024")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"]
    )
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--backward", action="store_true")
    args = parser.parse_args()

    hidden_size, intermediate_size = _dims(args.config)
    hidden_size = args.hidden_size or hidden_size
    intermediate_size = args.intermediate_size or intermediate_size

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    dense = nn.Linear(intermediate_size, hidden_size, bias=False).to(
        device=device, dtype=dtype
    )
    folded = FoldedSoftmaxReadout(
        hidden_size,
        intermediate_size,
        branches=args.branches,
        granularity=args.granularity,
    ).to(device=device, dtype=dtype)

    print(
        f"device={device} dtype={args.dtype} hidden={hidden_size} "
        f"intermediate={intermediate_size} branches={args.branches} "
        f"granularity={args.granularity} folded_width={folded.folded_size}"
    )
    print(
        "readout params: "
        f"dense={hidden_size * intermediate_size / 1e6:.2f}M "
        f"folded={hidden_size * folded.folded_size / 1e6:.2f}M "
        f"(+{args.branches * args.granularity} mixture weights)"
    )
    print(
        "note: gate/up projections are identical in both arms and are not "
        "measured here"
    )

    for tokens in (int(item) for item in args.tokens.split(",")):
        dense_seconds = _time(
            dense,
            tokens,
            intermediate_size,
            dtype,
            device,
            args.iters,
            args.warmup,
            args.backward,
        )
        folded_seconds = _time(
            folded,
            tokens,
            intermediate_size,
            dtype,
            device,
            args.iters,
            args.warmup,
            args.backward,
        )
        print(
            f"tokens={tokens:>6} "
            f"dense={dense_seconds * 1e6:9.1f}us "
            f"folded={folded_seconds * 1e6:9.1f}us "
            f"speedup={dense_seconds / folded_seconds:5.2f}x"
        )


if __name__ == "__main__":
    main()
