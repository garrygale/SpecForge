# coding=utf-8
"""Inference-speed probe for the routed outer draft FFN (ref vs fast paths).

Times the routed_outer blend (both router granularities) and the single
lattice against the dense reference, per batch size, with both a naive
"ref" implementation (mirrors the module math literally: index gathers,
materialized outer products) and a launch-count-optimized "fast" path:

  lattice fast   no index buffers at all — the outer-product broadcast
                 replaces gather+gather+mul (4 real ops vs 7)
  routed fast    the per-expert outer products and the sum over experts
                 fold into one einsum bmm (6 real ops vs ~8)

Every fast path is numerically checked against its ref twin before any
timing (the SSIM-probe discipline).  Self-contained: torch only, so it
runs on the training box, CUDA or any NPU container.

Usage (d=1536, E=4 48x48 experts; domino batch = batch x block_size 16)::

    python scripts/routed_outer_probe.py --hidden-size 1536 --experts 4 \
        --gate-groups 48 --up-groups 48 --batch-sizes 16,128,512,2048 \
        --iters 50 --warmup 10 --device auto --dtype bf16 \
        --dense-intermediate 9216
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
        gate, up = gate_up.chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class LatticeRefMLP(nn.Module):
    """Single lattice via literal index gathers (mirrors SharedGLUMLP's op
    mix; the channel order here is the row-major outer layout so the fast
    twin shares the same readout weights)."""

    def __init__(self, d, gate_groups, up_groups):
        super().__init__()
        self.G, self.U = gate_groups, up_groups
        self.intermediate = gate_groups * up_groups
        self.gate = nn.Linear(d, gate_groups, bias=False)
        self.up = nn.Linear(d, up_groups, bias=False)
        self.down = nn.Linear(self.intermediate, d, bias=False)
        channel = torch.arange(self.intermediate)
        self.register_buffer("gate_idx", channel // up_groups, persistent=False)
        self.register_buffer("up_idx", channel % up_groups, persistent=False)

    def forward(self, x):
        hidden = F.silu(self.gate(x)[..., self.gate_idx]) * self.up(x)[..., self.up_idx]
        return self.down(hidden)


class LatticeFastMLP(LatticeRefMLP):
    """Same math, no gathers: the gated hidden is the broadcast outer
    product silu(gate) (x) up, flattened in the same row-major order."""

    def forward(self, x):
        hidden = F.silu(self.gate(x)).unsqueeze(-1) * self.up(x).unsqueeze(-2)
        return self.down(hidden.reshape(x.shape[:-1] + (self.intermediate,)))


class RoutedRefMLP(nn.Module):
    """Routed outer experts, literal form (mirrors RoutedOuterMLP)."""

    def __init__(self, d, experts, gate_groups, up_groups, router):
        super().__init__()
        self.E, self.G, self.U = experts, gate_groups, up_groups
        self.router = router
        self.router_width = experts if router == "expert" else experts * gate_groups
        self.gate_up = nn.Linear(d, experts * (gate_groups + up_groups) + self.router_width, bias=False)
        self.down = nn.Linear(gate_groups * up_groups, d, bias=False)

    def blend(self, gate, up, route):
        lead = gate.shape[:-2]
        if self.router == "expert":
            gate = gate * torch.softmax(route, dim=-1).unsqueeze(-1)
        else:
            delta = torch.softmax(route.reshape(*lead, self.G, self.E), dim=-1)
            gate = gate * delta.transpose(-1, -2)
        return gate, up

    def forward(self, x):
        fused = self.gate_up(x)
        lead = fused.shape[:-1]
        E, G, U = self.E, self.G, self.U
        feats = fused[..., : E * (G + U)].reshape(*lead, E, G + U)
        gate = F.silu(feats[..., :G])
        up = feats[..., G:]
        route = fused[..., E * (G + U):]
        gate, up = self.blend(gate, up, route)
        hidden = (gate.unsqueeze(-1) * up.unsqueeze(-2)).sum(dim=-3)
        return self.down(hidden.reshape(*lead, G * U))


class RoutedFastMLP(RoutedRefMLP):
    """Same math: the per-expert outer products and the expert sum fold
    into a single einsum (one batched GEMM instead of mul+reduce), and
    nothing of width E*G*U is ever materialized."""

    def forward(self, x):
        fused = self.gate_up(x)
        lead = fused.shape[:-1]
        E, G, U = self.E, self.G, self.U
        feats = fused[..., : E * (G + U)].reshape(*lead, E, G + U)
        gate = F.silu(feats[..., :G])
        up = feats[..., G:]
        route = fused[..., E * (G + U):]
        gate, _ = self.blend(gate, up, route)
        # (..., G, E) @ (..., E, U) -> (..., G, U): the expert sum rides a
        # single batched matmul; nothing of width E*G*U is materialized.
        hidden = torch.matmul(gate.transpose(-1, -2), up)
        return self.down(hidden.reshape(*lead, G * U))


try:
    from numba import cuda as _ncuda

    @_ncuda.jit(fastmath=True)
    def _fused_expert_kernel(fused, out, total, E, G, U):
        """h[t, g*U+u] = sum_e softmax(route)_e * silu(gate) * up, one thread
        per output element; the whole elementwise chain in a single kernel
        (the SSIM-probe fusion pattern)."""
        idx = _ncuda.blockIdx.x * _ncuda.blockDim.x + _ncuda.threadIdx.x
        GU = G * U
        T = out.shape[0]
        if idx < T * GU:
            t = idx // GU
            gu = idx - t * GU
            g = gu // U
            u = gu - g * U
            base = t * total
            rbase = base + E * (G + U)
            m = fused[rbase]
            for e in range(E):
                v = fused[rbase + e]
                if v > m:
                    m = v
            s = 0.0
            for e in range(E):
                s += math.exp(fused[rbase + e] - m)
            acc = 0.0
            for e in range(E):
                w = math.exp(fused[rbase + e] - m) / s
                a = fused[base + e * (G + U) + g]
                a = a / (1.0 + math.exp(-a))
                acc += w * a * fused[base + e * (G + U) + G + u]
            out[idx] = acc

    @_ncuda.jit(fastmath=True)
    def _fused_gateslot_kernel(fused, out, total, E, G, U):
        """Same, per-gate-slot routing: softmax over E at rbase + g*E."""
        idx = _ncuda.blockIdx.x * _ncuda.blockDim.x + _ncuda.threadIdx.x
        GU = G * U
        T = out.shape[0]
        if idx < T * GU:
            t = idx // GU
            gu = idx - t * GU
            g = gu // U
            u = gu - g * U
            base = t * total
            rb = base + E * (G + U) + g * E
            m = fused[rb]
            for e in range(E):
                v = fused[rb + e]
                if v > m:
                    m = v
            s = 0.0
            for e in range(E):
                s += math.exp(fused[rb + e] - m)
            acc = 0.0
            for e in range(E):
                w = math.exp(fused[rb + e] - m) / s
                a = fused[base + e * (G + U) + g]
                a = a / (1.0 + math.exp(-a))
                acc += w * a * fused[base + e * (G + U) + G + u]
            out[idx] = acc

    _NUMBA_CUDA = True
except ImportError:
    _NUMBA_CUDA = False


class RoutedFusedMLP(RoutedRefMLP):
    """3-launch forward: gate_up GEMM, one fused numba kernel, down GEMM.
    fp32 only (numba); the production NPU version is a triton-ascend kernel
    with the same fusion and native bf16."""

    def forward(self, x):
        fused = self.gate_up(x)
        T = x.shape[0]
        # numba takes flat 1-D views: a 2-D out[idx] = scalar is a row
        # broadcast-fill in numba, not a flat write (the SSIM kernels passed
        # 1-D arrays for exactly this reason).
        out = torch.empty(T * self.G * self.U, device=x.device, dtype=x.dtype)
        n = out.shape[0]
        threads = 256
        blocks = (n + threads - 1) // threads
        flat = fused.reshape(-1)
        if self.router == "expert":
            _fused_expert_kernel[blocks, threads](flat, out, fused.shape[-1], self.E, self.G, self.U)
        else:
            _fused_gateslot_kernel[blocks, threads](flat, out, fused.shape[-1], self.E, self.G, self.U)
        return self.down(out.view(T, self.G * self.U))


try:
    import triton
    import triton.language as tl

    @triton.jit
    def _routed_combine_triton(
        fused_ptr, out_ptr, T, G, U, TOTAL, num_pid_n,
        E: tl.constexpr, GATE_SLOT: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        """h[(t*G+g)*U+u] = sum_e w_e * silu(gate[t,e,g]) * up[t,e,u].

        One 1-D grid program per (BLOCK_M rows of t*G+g, BLOCK_N of U).
        bf16 loads, fp32 math, bf16 store so the down GEMM consumes the
        output directly; ``care_padding=False`` on masked loads follows the
        domino_gru house style (lanes are elementwise-independent)."""
        pid = tl.program_id(0)
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rows = T * G
        mask_m = offs_m < rows
        mask_n = offs_n < U
        t = offs_m // G
        g = offs_m % G
        route_off = t * TOTAL + E * (G + U)
        if GATE_SLOT:
            route_off += g * E

        # softmax over E (unrolled; E is constexpr), two passes for max/denom
        m = tl.full((BLOCK_M,), float("-inf"), tl.float32)
        for e in tl.static_range(E):
            r = tl.load(
                fused_ptr + route_off + e, mask=mask_m, other=float("-inf"),
                care_padding=False,
            ).to(tl.float32)
            m = tl.maximum(m, r)
        s = tl.zeros((BLOCK_M,), tl.float32)
        for e in tl.static_range(E):
            r = tl.load(
                fused_ptr + route_off + e, mask=mask_m, other=0.0,
                care_padding=False,
            ).to(tl.float32)
            s += tl.exp(r - m)

        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for e in tl.static_range(E):
            r = tl.load(
                fused_ptr + route_off + e, mask=mask_m, other=0.0,
                care_padding=False,
            ).to(tl.float32)
            w = tl.exp(r - m) / s
            gate = tl.load(
                fused_ptr + t * TOTAL + e * (G + U) + g,
                mask=mask_m, other=0.0, care_padding=False,
            ).to(tl.float32)
            a = gate / (1.0 + tl.exp(-gate)) * w
            up = tl.load(
                fused_ptr + t[:, None] * TOTAL + e * (G + U) + G + offs_n[None, :],
                mask=mask_m[:, None] & mask_n[None, :], other=0.0,
                care_padding=False,
            ).to(tl.float32)
            acc += a[:, None] * up
        tl.store(
            out_ptr + offs_m[:, None] * U + offs_n[None, :],
            acc.to(out_ptr.dtype.element_ty),
            mask=mask_m[:, None] & mask_n[None, :],
        )

    _TRITON = True
except ImportError:
    _TRITON = False


class RoutedTritonMLP(RoutedRefMLP):
    """Triton fused combine (bf16 in/out, fp32 math): GEMM + kernel + GEMM.
    Written for triton-ascend on NPU; also runs on CUDA triton where
    available.  Kernel math is verified against the ref twin by the probe's
    correctness gate before any timing."""

    def __init__(self, d, experts, gate_groups, up_groups, router,
                 block_m=64, block_n=32, num_warps=4):
        super().__init__(d, experts, gate_groups, up_groups, router)
        self.block_m, self.block_n, self.num_warps = block_m, block_n, num_warps

    def forward(self, x):
        fused = self.gate_up(x)
        T = x.shape[0]
        G, U, E = self.G, self.U, self.E
        out = torch.empty(T * G * U, device=x.device, dtype=fused.dtype)
        num_pid_m = triton.cdiv(T * G, self.block_m)
        num_pid_n = triton.cdiv(U, self.block_n)
        _routed_combine_triton[(num_pid_m * num_pid_n,)](
            fused,
            out,
            T,
            G,
            U,
            fused.shape[-1],
            num_pid_n,
            E=E,
            GATE_SLOT=1 if self.router == "gate_slot" else 0,
            BLOCK_M=self.block_m,
            BLOCK_N=self.block_n,
            num_warps=self.num_warps,
        )
        return self.down(out.view(T, G * U))


def _pick_device(name):
    if name != "auto":
        return torch.device(name)
    for attr in ("npu", "cuda"):
        try:
            if getattr(torch, attr) is not None and getattr(torch, attr).is_available():
                return torch.device(attr)
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


# ---------------------------------------------------------------------------
# Edit here instead of typing flags: every CLI default below reads from this
# block (command-line flags still win when given).
# ---------------------------------------------------------------------------
CONFIG = dict(
    hidden_size=1536,
    experts=8,
    gate_groups=48,
    up_groups=48,
    routers="expert,gate_slot",
    batch_sizes="8,32,128,256",
    dense_intermediate=6144,      # None -> experts*gate*up
    no_lattice=True,
    fused=False,                  # numba single-kernel fusion (fp32 + cuda)
    fused_triton=True,            # triton-ascend fused combine (NPU)
    block_m=64,
    block_n=32,
    iters=50,
    warmup=10,
    device="auto",
    dtype="bf16",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hidden-size", type=int, default=CONFIG["hidden_size"])
    parser.add_argument("--experts", type=int, default=CONFIG["experts"])
    parser.add_argument("--gate-groups", type=int, default=CONFIG["gate_groups"])
    parser.add_argument("--up-groups", type=int, default=CONFIG["up_groups"])
    parser.add_argument("--routers", default=CONFIG["routers"])
    parser.add_argument("--batch-sizes", default=CONFIG["batch_sizes"],
                        help="tokens per forward; for domino, batch x block_size(16)")
    parser.add_argument("--dense-intermediate", type=int, default=CONFIG["dense_intermediate"],
                        help="dense baseline width (default experts*G*U; pass the "
                             "old draft's width, e.g. 6144, for a vs-production row)")
    parser.add_argument("--no-lattice", action=argparse.BooleanOptionalAction, default=CONFIG["no_lattice"],
                        help="drop the single-lattice rows (routed vs dense only)")
    parser.add_argument("--fused", action=argparse.BooleanOptionalAction, default=CONFIG["fused"],
                        help="add the single-kernel numba fusion rows (fp32 + cuda only)")
    parser.add_argument("--fused-triton", action=argparse.BooleanOptionalAction, default=CONFIG["fused_triton"],
                        help="add the triton fused-combine rows (triton-ascend on NPU; "
                             "same dtype as the run)")
    parser.add_argument("--block-m", type=int, default=CONFIG["block_m"])
    parser.add_argument("--block-n", type=int, default=CONFIG["block_n"])
    parser.add_argument("--iters", type=int, default=CONFIG["iters"])
    parser.add_argument("--warmup", type=int, default=CONFIG["warmup"])
    parser.add_argument("--device", default=CONFIG["device"])
    parser.add_argument("--dtype", default=CONFIG["dtype"], choices=["bf16", "fp16", "fp32"])
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
    dense_I = args.dense_intermediate or I

    # (name, module, macs_per_token, forward_op_count, ref_twin_for_the_gate)
    variants = [
        (f"dense (I={dense_I})", DenseMLP(d, dense_I), 3 * d * dense_I, 4, None),
    ]
    if not args.no_lattice:
        lat_ref = LatticeRefMLP(d, G, U)
        lat_fast = LatticeFastMLP(d, G, U)
        lat_fast.load_state_dict(lat_ref.state_dict())
        variants.append(("lattice ref (gather)", lat_ref, d * (G + U) + d * G * U, 7, None))
        variants.append(("lattice fast (outer)", lat_fast, d * (G + U) + d * G * U, 4, "lattice ref (gather)"))
    for router in routers:
        ref = RoutedRefMLP(d, E, G, U, router)
        fast = RoutedFastMLP(d, E, G, U, router)
        fast.load_state_dict(ref.state_dict())
        macs = d * (E * (G + U) + ref.router_width) + d * G * U + E * G * U
        variants.append((f"routed {router} ref", ref, macs, 8, None))
        variants.append((f"routed {router} fast", fast, macs, 6, f"routed {router} ref"))
        if args.fused:
            if dtype != torch.float32:
                raise SystemExit("--fused needs --dtype fp32 (numba kernel)")
            if device.type != "cuda" or not _NUMBA_CUDA:
                raise SystemExit("--fused needs a cuda device with numba installed")
            fused_mlp = RoutedFusedMLP(d, E, G, U, router)
            fused_mlp.load_state_dict(ref.state_dict())
            variants.append((f"routed {router} fused", fused_mlp, macs, 3, f"routed {router} ref"))
        if args.fused_triton:
            if not _TRITON:
                raise SystemExit("--fused-triton needs triton installed")
            tri = RoutedTritonMLP(d, E, G, U, router, args.block_m, args.block_n)
            tri.load_state_dict(ref.state_dict())
            variants.append((f"routed {router} triton", tri, macs, 3, f"routed {router} ref"))

    # ---- correctness gate: every fast path must match its ref twin ----
    torch.manual_seed(0)
    x_check = torch.randn(max(batches), d)
    tol = 1e-4 if dtype == torch.float32 else 2e-2
    by_name = {name: mlp for name, mlp, _, _, _ in variants}
    with torch.inference_mode():
        x_dev = x_check.to(device=device, dtype=dtype)
        for name, mlp, _, _, ref_name in variants:
            if ref_name is None:
                continue
            fast_out = mlp.to(device=device, dtype=dtype)(x_dev)
            ref_out = by_name[ref_name].to(device=device, dtype=dtype)(x_dev)
            ok = torch.allclose(fast_out.float(), ref_out.float(), atol=tol, rtol=tol)
            status = "OK" if ok else "MISMATCH"
            print(f"correctness: {name:<24} vs {ref_name:<24} {status}")
            if not ok:
                raise SystemExit("fast path diverges from its reference; aborting")
    print()

    print("== routed-outer inference probe ==")
    print(f"device={_device_label(device)}  dtype={args.dtype}  torch={torch.__version__}")
    print(f"d={d}  experts={E}  lattice {G}x{U}  I={I}  iters={args.iters} (+{args.warmup} warmup)")
    bytes_per = torch.empty(1, dtype=dtype).element_size()
    print(f"\n{'variant':<28}{'params':>12}{'MACs/tok':>12}{'ops':>5}{'weights':>12}")
    for name, mlp, macs, ops, _ in variants:
        params = sum(p.numel() for p in mlp.parameters())
        print(f"{name:<28}{params:>12,}{macs:>12,}{ops:>5}{params * bytes_per / 1e6:>10.1f}MB")

    print()
    speedups = {name: [] for name, _, _, _, _ in variants[1:]}
    fast_gains = []
    for batch in batches:
        x = torch.randn(batch, d, device=device, dtype=dtype)
        rows = []
        with torch.inference_mode():
            for name, mlp, _, _, _ in variants:
                mlp = mlp.to(device=device, dtype=dtype)
                ms = _time(lambda mlp=mlp: mlp(x), args.iters, args.warmup, device)
                rows.append((name, ms))
        dense_ms = rows[0][1]
        dense_name = rows[0][0]
        print(f"batch={batch:>6} tok")
        print(f"  {'variant':<28}{'ms/call':>10}{'vs dense':>10}{'us/tok':>10}{'GB/s(w)':>10}")
        params_by_name = {
            name: sum(p.numel() for p in mlp.parameters()) for name, mlp, _, _, _ in variants
        }
        ms_by_name = dict(rows)
        for name, ms in rows:
            gb_s = params_by_name[name] * bytes_per / 1e9 / (ms / 1e3)
            tag = "" if name == dense_name else f"{dense_ms / ms:>9.2f}x"
            print(f"  {name:<28}{ms:>10.3f}{tag:>10}{ms * 1e3 / batch:>10.2f}{gb_s:>10.1f}")
            if name != dense_name:
                speedups[name].append(dense_ms / ms)
        for router in routers:
            gain = ms_by_name[f"routed {router} ref"] / ms_by_name[f"routed {router} fast"]
            fast_gains.append((batch, router, gain))
        if not args.no_lattice:
            lat_gain = ms_by_name["lattice ref (gather)"] / ms_by_name["lattice fast (outer)"]
            fast_gains.append((batch, "lattice", lat_gain))
        print()

    print("== summary (geometric-mean speedup vs dense across batches) ==")
    for name, vals in speedups.items():
        if vals:
            geo = math.exp(sum(math.log(v) for v in vals) / len(vals))
            print(f"  {name:<28}{geo:>6.2f}x  range {min(vals):.2f}x-{max(vals):.2f}x")
    print("\n== fast-path gain over its own ref twin (the optimization delta) ==")
    by_kind = {}
    for batch, kind, gain in fast_gains:
        by_kind.setdefault(kind, []).append(gain)
    for kind, vals in by_kind.items():
        geo = math.exp(sum(math.log(v) for v in vals) / len(vals))
        print(f"  {kind:<12}{geo:>6.2f}x  range {min(vals):.2f}x-{max(vals):.2f}x")
    print(
        "\nreading guide: at small batches the call is weight-bandwidth-bound "
        "(compare GB/s(w) with the device HBM bandwidth); at large batches it "
        "turns GEMM-bound and speedups converge toward the MACs/tok ratio. "
        "The ref/fast delta is the launch-count effect — expect it to grow "
        "on NPU/CUDA where per-kernel launch overhead is larger than on CPU."
    )


if __name__ == "__main__":
    main()
