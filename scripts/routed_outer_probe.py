# coding=utf-8
"""Inference-speed probe for the routed outer draft FFN.

Times the routed_outer blend (both router granularities) against the
reference points it must beat, per batch size, and prints an interpretable
table plus a summary.  Self-contained: only needs torch, so it runs on the
training box, the A3 node or any vLLM container (mirror of SpecForge's
``RoutedOuterMLP``/``SharedGLUMLP`` in ``dflash_kernels.py`` — the parity
tests lock the real modules to this math).

Rows per batch size:

  dense        full SwiGLU at intermediate I = E*G*U   (accuracy/cost ceiling)
  lattice GxU  single outer lattice, E=1 equivalent    (the e1 sweep point)
  lattice ExGxU  one lattice spending the whole I      (routing's alternative)
  routed expert / gate_slot                            (the new path)

Usage (d=1536, E=4 48x48 experts, domino batch = batch x block_size 16)::

    python scripts/routed_outer_probe.py --hidden-size 1536 --experts 4 \
        --gate-groups 48 --up-groups 48 --batch-sizes 16,128,512,2048 \
        --iters 50 --warmup 10 --device auto --dtype bf16
"""

from __future__ import annotations

import argparse
import math
import time

import torch
from torch import nn
import torch.nn.functional as F


class DenseMLP(nn.Module):
    """Reference full SwiGLU (Qwen3MLP) at ``intermediate`` channels."""

    def __init__(self, d, intermediate):
        super().__init__()
        self.gate_up = nn.Linear(d, 2 * intermediate, bias=False)
        self.down = nn.Linear(intermediate, d, bias=False)

    def forward(self, x):
        gate_up = self.gate_up(x)
        return self.down(F.silu(gate_up[..., : gate_up.shape[-1] // 2]) * gate_up[..., gate_up.shape[-1] // 2:])


class LatticeMLP(nn.Module):
    """Single outer lattice (SpecForge SharedGLUMLP, pairing='outer')."""

    def __init__(self, d, gate_groups, up_groups):
        super().__init__()
        self.gate_groups, self.up_groups = gate_groups, up_groups
        self.intermediate = gate_groups * up_groups
        self.gate = nn.Linear(d, gate_groups, bias=False)
        self.up = nn.Linear(d, up_groups, bias=False)
        self.down = nn.Linear(self.intermediate, d, bias=False)
        channel = torch.arange(self.intermediate)
        self.register_buffer("gate_idx", channel % gate_groups, persistent=False)
        self.register_buffer("up_idx", channel // gate_groups, persistent=False)

    def forward(self, x):
        gate = self.gate(x)
        up = self.up(x)
        hidden = F.silu(gate[..., self.gate_idx]) * up[..., self.up_idx]
        return self.down(hidden)


class RoutedMLP(nn.Module):
    """Routed outer experts (mirror of SpecForge RoutedOuterMLP)."""

    def __init__(self, d, experts, gate_groups, up_groups, router):
        super().__init__()
        self.E, self.G, self.U = experts, gate_groups, up_groups
        self.router = router
        self.router_width = experts if router == "expert" else experts * gate_groups
        self.gate_up = nn.Linear(d, experts * (gate_groups + up_groups) + self.router_width, bias=False)
        self.down = nn.Linear(gate_groups * up_groups, d, bias=False)

    def forward(self, x):
        fused = self.gate_up(x)
        lead = fused.shape[:-1]
        E, G, U = self.E, self.G, self.U
        feats = fused[..., : E * (G + U)].reshape(*lead, E, G + U)
        gate = F.silu(feats[..., :G])
        up = feats[..., G:]
        route = fused[..., E * (G + U):]
        if self.router == "expert":
            gate = gate * torch.softmax(route, dim=-1).unsqueeze(-1)
        else:
            delta = torch.softmax(route.reshape(*lead, G, E), dim=-1)
            gate = gate * delta.transpose(-1, -2)
        hidden = (gate.unsqueeze(-1) * up.unsqueeze(-2)).sum(dim=-3)
        return self.down(hidden.reshape(*lead, G * U))


def _pick_device(name):
    if name != "auto":
        return torch.device(name)
    for attr, dev in (("npu", "npu"), ("cuda", "cuda")):
        try:
            if getattr(torch, attr) is not None and getattr(torch, attr).is_available():
                return torch.device(dev)
        except Exception:
            pass
    return torch.device("cpu")


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "npu":
        torch.npu.synchronize()


def _device_label(device):
    try:
        if device.type == "cuda":
            return f"{device} ({torch.cuda.get_device_name(0)})"
        if device.type == "npu":
            return f"{device} ({torch.npu.get_device_name(0)})"
    except Exception:
        pass
    return str(device)


def _time(fn, iters, warmup, device):
    for _ in range(warmup):
        fn()
    _sync(device)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        _sync(device)
        times.append(time.perf_counter() - t0)
    return min(times) * 1e3  # ms/call, min is the robust estimator


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hidden-size", type=int, default=1536)
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--gate-groups", type=int, default=48)
    parser.add_argument("--up-groups", type=int, default=48)
    parser.add_argument("--routers", default="expert,gate_slot")
    parser.add_argument("--batch-sizes", default="16,128,512,2048",
                        help="tokens per forward; for domino, batch x block_size(16)")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    args = parser.parse_args()

    d, E, G, U = args.hidden_size, args.experts, args.gate_groups, args.up_groups
    if min(d, E, G, U) < 1:
        raise SystemExit("hidden-size/experts/gate-groups/up-groups must be >= 1")
    routers = [r.strip() for r in args.routers.split(",") if r.strip()]
    for r in routers:
        if r not in ("expert", "gate_slot"):
            raise SystemExit(f"unknown router {r!r} (expert | gate_slot)")
    if E == 1 and "gate_slot" in routers:
        print("note: gate_slot with experts=1 is the identity blend; skipping it")
        routers = [r for r in routers if r != "gate_slot"]
    batches = [int(b) for b in args.batch_sizes.split(",")]
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    device = _pick_device(args.device)
    I = E * G * U

    variants = [
        ("dense", DenseMLP(d, I), 3 * d * I),
        ("lattice %dx%d (e1)" % (G, U), LatticeMLP(d, G, U), d * (G + U) + d * G * U),
        ("lattice %dx%d (full I)" % (E * G, U), LatticeMLP(d, E * G, U), d * (E * G + U) + d * I),
    ]
    for router in routers:
        mlp = RoutedMLP(d, E, G, U, router)
        macs = d * (E * (G + U) + mlp.router_width) + d * G * U + E * G * U
        variants.append((f"routed {router}", mlp, macs))

    print("== routed-outer inference probe ==")
    print(f"device={_device_label(device)}  dtype={args.dtype}  torch={torch.__version__}")
    print(f"d={d}  experts={E}  lattice {G}x{U}  I={I}  iters={args.iters} (+{args.warmup} warmup)")
    bytes_per = torch.empty(1, dtype=dtype).element_size()
    print(f"\n{'variant':<28}{'params':>12}{'MACs/tok':>12}{'weights':>12}")
    for name, mlp, macs in variants:
        params = sum(p.numel() for p in mlp.parameters())
        print(f"{name:<28}{params:>12,}{macs:>12,}{params * bytes_per / 1e6:>10.1f}MB")
    mlp = variants[0][1].to(device=device, dtype=dtype)

    print()
    speedups = {name: [] for name, _, _ in variants[1:]}
    for batch in batches:
        x = torch.randn(batch, d, device=device, dtype=dtype)
        rows = []
        with torch.inference_mode():
            for name, mlp, _ in variants:
                mlp = mlp.to(device=device, dtype=dtype)
                ms = _time(lambda mlp=mlp: mlp(x), args.iters, args.warmup, device)
                rows.append((name, ms))
        dense_ms = rows[0][1]
        print(f"batch={batch:>6} tok  ({batch * d * bytes_per / 1e6:.1f}MB activations)")
        header = f"  {'variant':<28}{'ms/call':>10}{'vs dense':>10}{'us/tok':>10}{'GB/s(w)':>10}"
        print(header)
        params_by_name = {name: sum(p.numel() for p in mlp.parameters()) for name, mlp, _ in variants}
        for name, ms in rows:
            gb_s = (
                params_by_name[name] * bytes_per / 1e9 / (ms / 1e3)
            )
            tag = "" if name == "dense" else f"{dense_ms / ms:>9.2f}x"
            print(f"  {name:<28}{ms:>10.3f}{tag:>10}{ms * 1e3 / batch:>10.2f}{gb_s:>10.1f}")
            if name != "dense":
                speedups[name].append(dense_ms / ms)
        print()

    print("== summary (geometric-mean speedup vs dense across batches) ==")
    for name, vals in speedups.items():
        if vals:
            geo = math.exp(sum(math.log(v) for v in vals) / len(vals))
            spread = f"  range {min(vals):.2f}x-{max(vals):.2f}x"
            print(f"  {name:<28}{geo:>6.2f}x{spread}")
    print(
        "\nreading guide: at small batches the call is weight-bandwidth-bound "
        "(compare GB/s(w) with the device HBM bandwidth); at large batches it "
        "turns GEMM-bound and speedups converge toward the MACs/tok ratio. "
        "Routed-vs-e1-lattice isolates what the extra experts cost; "
        "routed-vs-full-I-lattice is what the rank-<=E readout saves."
    )


if __name__ == "__main__":
    main()
