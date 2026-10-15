"""Wall-clock comparison of einsum-free formulations for the VHA ops.

Training-time VHA adds two small ops per attention layer: the per-group
premix (physical->virtual query head expansion) and the low-rank postmix
(head-axis residual before o_proj).  einsum lowers them to permute+bmm
paths that measured catastrophically slow on the dflare fusion work, so
this bench times the candidate einsum-free formulations end-to-end
(reshape/stack/add included), forward and backward, on the device that
matters:

    python scripts/bench_vha_ops.py --device cuda --dtype float16
    python scripts/bench_vha_ops.py --device npu --dtype bfloat16

Shapes default to the dflare draft: 32 virtual heads, head_dim 128,
physical 16, kv groups 2, postmix rank 8; prefill B12/S2048 and a decode
block B12/S16.  fp32 CPU runs work too for correctness cross-checks
(--device cpu --check prints max pairwise deviation).
"""

from __future__ import annotations

import argparse
import time

import torch

PREMIX_FORMS = ("einsum", "matmul_groups", "matmul_broadcast", "cat_groups", "blockdiag_gemm")
POSTMIX_FORMS = ("einsum2", "gemm_transpose", "bmm_expand")


def premix_einsum(q, premix):
    b, s, p, d = q.shape
    return torch.einsum("bspd,kde->bskpe", q, premix).reshape(b, s, p * premix.shape[0], d)


def premix_matmul_groups(q, premix):
    b, s, p, d = q.shape
    return torch.stack([q @ pm for pm in premix.unbind(0)], dim=2).reshape(
        b, s, p * premix.shape[0], d
    )


def premix_cat_groups(q, premix):
    b, s, p, d = q.shape
    mixed = torch.cat(
        [(q @ pm).reshape(b, s, p * d) for pm in premix.unbind(0)], dim=2
    )
    return mixed.reshape(b, s, p * premix.shape[0], d)


def premix_blockdiag_gemm(q, premix):
    # out[b,s,(k,p),d] = q[b,s,p,:] @ premix[k] == q_flat @ blockdiag(premix)
    b, s, p, d = q.shape
    k = premix.shape[0]
    eye = torch.eye(p, device=q.device, dtype=q.dtype).view(1, p, 1, p, 1)
    weight = (premix.transpose(1, 2).view(k, 1, d, 1, d) * eye).reshape(k * p * d, p * d)
    return (q.reshape(b * s, p * d) @ weight.t()).reshape(b, s, k * p, d)


def premix_matmul_broadcast(q, premix):
    b, s, p, d = q.shape
    return torch.matmul(q.unsqueeze(2), premix.unsqueeze(0)).reshape(
        b, s, p * premix.shape[0], d
    )


def postmix_einsum2(y, u, v):
    b, t, flat = y.shape
    heads = y.view(b, t, u.shape[0], -1)
    z = torch.einsum("bthd,hr->btrd", heads, u)
    delta = torch.einsum("btrd,hr->bthd", z, v)
    return (heads + delta).reshape(b, t, flat)


def postmix_gemm_transpose(y, u, v):
    b, t, flat = y.shape
    heads = y.view(b, t, u.shape[0], -1)
    m = u @ v.t()
    delta = (heads.transpose(-1, -2) @ m).transpose(-1, -2)
    return (heads + delta).reshape(b, t, flat)


def postmix_bmm_expand(y, u, v):
    b, t, flat = y.shape
    h = u.shape[0]
    heads = y.view(b, t, h, -1)
    m = u @ v.t()
    delta = torch.bmm(
        m.t().expand(b * t, h, h), heads.reshape(b * t, h, flat // h)
    ).view(b, t, h, -1)
    return (heads + delta).reshape(b, t, flat)


PREMIX = {
    "einsum": premix_einsum,
    "matmul_groups": premix_matmul_groups,
    "matmul_broadcast": premix_matmul_broadcast,
    "cat_groups": premix_cat_groups,
    "blockdiag_gemm": premix_blockdiag_gemm,
}
POSTMIX = {
    "einsum2": postmix_einsum2,
    "gemm_transpose": postmix_gemm_transpose,
    "bmm_expand": postmix_bmm_expand,
}


def bench(fn, args, params, iters, warmup=5):
    for _ in range(warmup):
        out = fn(*args)
        (out.sum() + sum(p.sum() for p in params)).backward()
        for p in params:
            p.grad = None
    if torch.cuda.is_available() and args[0].is_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(iters):
        out = fn(*args)
        (out.sum() + sum(p.sum() for p in params)).backward()
        for p in params:
            p.grad = None
    if torch.cuda.is_available() and args[0].is_cuda:
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 2**20
    else:
        peak = float("nan")
    return (time.perf_counter() - start) / iters * 1e3, peak


def reference_premix(q, premix):
    b, s, p, d = q.shape
    out = torch.zeros(b, s, premix.shape[0], p, d, dtype=q.dtype)
    for k, pm in enumerate(premix.unbind(0)):
        for i in range(p):
            out[:, :, k, i, :] = q[:, :, i, :] @ pm
    return out.reshape(b, s, p * premix.shape[0], d)


def reference_postmix(y, u, v):
    b, t, flat = y.shape
    h, r = u.shape
    d = flat // h
    heads = y.view(b, t, h, d)
    out = heads.clone()
    for hb in range(h):
        for rr in range(r):
            z = (heads * u[:, rr].view(1, 1, h, 1)).sum(dim=2)
            out[:, :, hb, :] += v[hb, rr] * z
    return out.reshape(b, t, flat)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--shapes",
        default="12x2048,12x16",
        help="batch x seq list, comma separated",
    )
    parser.add_argument(
        "--heads", type=int, default=32, help="virtual heads (postmix axis)"
    )
    parser.add_argument("--physical", type=int, default=16)
    parser.add_argument("--kv-groups", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--check", action="store_true", help="print correctness refs")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(0)

    def run_premix(bs, s):
        q = torch.randn(
            bs, s, args.physical, args.head_dim, device=device, dtype=dtype, requires_grad=True
        )
        premix = torch.randn(
            args.kv_groups, args.head_dim, args.head_dim, device=device, dtype=dtype, requires_grad=True
        )
        rows = []
        for name, fn in PREMIX.items():
            ms, peak = bench(fn, (q, premix), [premix], args.iters)
            rows.append((name, ms, peak))
        return rows

    def run_postmix(bs, s):
        y = torch.randn(
            bs, s, args.heads * args.head_dim, device=device, dtype=dtype, requires_grad=True
        )
        u = torch.randn(args.heads, args.rank, device=device, dtype=dtype, requires_grad=True)
        v = torch.randn(args.heads, args.rank, device=device, dtype=dtype, requires_grad=True)
        rows = []
        for name, fn in POSTMIX.items():
            ms, peak = bench(fn, (y, u, v), [u, v], args.iters)
            rows.append((name, ms, peak))
        return rows

    print(f"device={device} dtype={dtype} iters={args.iters} "
          f"H={args.heads} P={args.physical} K={args.kv_groups} D={args.head_dim} R={args.rank}")
    for spec in args.shapes.split(","):
        b, s = (int(x) for x in spec.split("x"))
        print(f"\n== premix  B{b} S{s} (fwd+bwd, ms/iter | peak MB)")
        for name, ms, peak in run_premix(b, s):
            print(f"   {name:<18} {ms:8.3f} | {peak:8.0f}")
        print(f"== postmix B{b} S{s} (fwd+bwd, ms/iter | peak MB)")
        for name, ms, peak in run_postmix(b, s):
            print(f"   {name:<18} {ms:8.3f} | {peak:8.0f}")

    if args.check:
        cpu = dict(device=torch.device("cpu"), dtype=torch.float32)
        torch.manual_seed(1)
        q = torch.randn(2, 5, args.physical, args.head_dim, **cpu)
        premix = torch.randn(args.kv_groups, args.head_dim, args.head_dim, **cpu)
        ref = reference_premix(q, premix)
        for name, fn in PREMIX.items():
            gap = (fn(q, premix) - ref).abs().max().item()
            print(f"premix {name:<18} max|diff| {gap:.2e}")
        y = torch.randn(2, 5, args.heads * args.head_dim, **cpu)
        u = torch.randn(args.heads, args.rank, **cpu)
        v = torch.randn(args.heads, args.rank, **cpu)
        ref = reference_postmix(y, u, v)
        for name, fn in POSTMIX.items():
            gap = (fn(y, u, v) - ref).abs().max().item()
            print(f"postmix {name:<17} max|diff| {gap:.2e}")


if __name__ == "__main__":
    main()
