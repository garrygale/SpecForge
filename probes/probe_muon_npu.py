# coding=utf-8
"""A3 probe: torch.optim.Muon + the SpecForge muon-mode shell on Ascend NPU.

Answers, in one self-analyzing run (no manual follow-up needed):

1. OPS  — does a full torch.optim.Muon step run on this device at all
          (addmm/lerp_/bf16-cast coverage of the Newton-Schulz path)?
2. SHELL— does BF16Optimizer(optimizer_type="muon") construct, split, step and
          copy back through the fp32-master shell on this device?
3. NS   — Newton-Schulz orthogonality error in bf16 (Muon's internal compute
          dtype) vs fp32, on draft-scale shapes.
4. TIME — Muon step wall time per matrix shape vs an AdamW step, so the
          per-step overhead of the optimizer can be estimated.
5. FIT  — tiny least-squares race, Muon vs AdamW, loss curve printed.

Run on the A3 box:  python probes/probe_muon_npu.py
Optional env:       SPECFORGE_PROBE_DEVICE=npu|cuda|cpu  (default: auto)
"""

import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

REPORT = []


def log(line=""):
    print(line)
    REPORT.append(line)


def pick_device():
    forced = os.environ.get("SPECFORGE_PROBE_DEVICE")
    if forced:
        return torch.device(forced)
    try:
        import torch_npu  # noqa: F401

        if torch.npu.is_available():
            return torch.device("npu")
    except ImportError:
        pass
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def verdict(ok, label, detail):
    log(f"[{'PASS' if ok else 'FAIL'}] {label}: {detail}")


def newton_schulz(update, steps=5, dtype=torch.bfloat16):
    """Reference replication of torch's quintic NS for precision diagnostics."""
    a, b, c = 3.4445, -4.775, 2.0315
    x = update.to(dtype)
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.T
    x = x / x.norm().clamp(min=1e-7)
    for _ in range(steps):
        g = x @ x.T
        g = b * g + c * (g @ g)
        x = a * x + g @ x
    if transposed:
        x = x.T
    return x.to(torch.float32)


def ns_stats(update, ortho):
    m, n = update.shape
    gram = (ortho.T @ ortho) if m >= n else (ortho @ ortho.T)
    off_diag = gram - torch.diag(torch.diag(gram))
    mean_cos = float(off_diag.abs().mean())
    sv = torch.linalg.svdvals(ortho).to(torch.float32)
    return mean_cos, float(sv.median()), float(sv.max())


def main():
    device = pick_device()
    log(f"# Muon NPU probe — device={device} torch={torch.__version__}")
    if not hasattr(torch.optim, "Muon"):
        verdict(False, "OPS", "torch.optim.Muon missing (need torch>=2.9)")
        return
    log()

    # ---- 1. OPS: one full Muon step over fp32 masters (shell-style) ----
    torch.manual_seed(0)
    shapes = [(2048, 2048), (2048, 6144), (6144, 2048), (2048, 14336)]
    masters = [
        torch.randn(s, device=device, dtype=torch.float32) * 0.02 for s in shapes
    ]
    for mp in masters:
        mp.requires_grad_(True)
        mp.grad = torch.randn_like(mp) * 0.1
    muon = torch.optim.Muon(masters, lr=0.02, adjust_lr_fn="match_rms_adamw")
    before = [mp.detach().clone() for mp in masters]
    try:
        muon.step()
        moved = [float((b - a).abs().max()) for b, a in zip(masters, before)]
        finite = all(math.isfinite(v) for v in moved)
        verdict(finite and max(moved) > 0, "OPS", f"Muon step ran, max|delta| per matrix={['%.2e' % v for v in moved]}")
    except Exception as exc:  # noqa: BLE001
        verdict(False, "OPS", f"Muon step raised {type(exc).__name__}: {exc}")
        log("\nMuon is unusable on this device — stop here.")
        return
    log()

    # ---- 2. SHELL: full BF16Optimizer muon-mode step on device ----
    from types import SimpleNamespace

    from torch import nn

    from specforge.optimizer import BF16Optimizer

    class TinyDraft(nn.Module):
        def __init__(self, vocab=96, hidden=32):
            super().__init__()
            self.fc1 = nn.Linear(hidden, 64, bias=False)
            self.norm_w = nn.Parameter(torch.ones(64))
            self.head = nn.Linear(64, vocab, bias=False)
            self.config = SimpleNamespace(vocab_size=vocab)

        def forward(self, x):
            return self.head(torch.nn.functional.gelu(self.fc1(x)))

    try:
        torch.manual_seed(0)
        model = TinyDraft().to(device)
        opt = BF16Optimizer(
            model,
            lr=5e-2,
            total_steps=100,
            lr_scheduler="constant",
            optimizer_type="muon",
        )
        torch.manual_seed(1)
        inputs = torch.randn(128, 32, device=device)
        targets = inputs @ torch.randn(32, 96, device=device)
        losses = []
        for _ in range(30):
            loss = torch.nn.functional.mse_loss(model(inputs), targets)
            model.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss))
        ok = losses[-1] < losses[0] * 0.5
        verdict(ok, "SHELL", f"loss {losses[0]:.3f} -> {losses[-1]:.3f} over 30 steps (muon lr 5e-2, match_rms_adamw)")
    except Exception as exc:  # noqa: BLE001
        verdict(False, "SHELL", f"raised {type(exc).__name__}: {exc}")
    log()

    # ---- 3. NS: quintic behaviour bf16 (Muon's internal dtype) vs fp32 ----
    # The iteration is approximate by design: sigma converges toward ~[0.5,
    # 1.5] and mean inter-column |cos| is a few 1e-2 after 5 steps.
    for shape in shapes:
        g = torch.randn(shape, device=device, dtype=torch.float32)
        o_bf16 = newton_schulz(g, dtype=torch.bfloat16)
        o_fp32 = newton_schulz(g, dtype=torch.float32)
        cos16, med16, max16 = ns_stats(g, o_bf16)
        cos32, med32, max32 = ns_stats(g, o_fp32)
        drift = float((o_bf16 - o_fp32).norm() / o_fp32.norm())
        ok = cos16 < 0.05 and med16 > 0.4 and max16 < 1.7 and drift < 0.05
        verdict(
            ok,
            "NS",
            f"{shape}: |cos| bf16={cos16:.4f} fp32={cos32:.4f}, "
            f"sigma_med/max bf16={med16:.2f}/{max16:.2f} fp32={med32:.2f}/{max32:.2f}, "
            f"bf16-vs-fp32 drift={drift:.2e}",
        )
    log()

    # ---- 4. TIME: Muon vs AdamW per-step cost ----
    log("[TIME] per-step wall time, median of 10 (fp32 masters, grads prefilled):")
    for shape in shapes:
        m = (torch.randn(shape, device=device) * 0.02).requires_grad_(True)
        m.grad = torch.randn_like(m)
        mu = torch.optim.Muon([m], lr=0.02, adjust_lr_fn="match_rms_adamw")
        ad = torch.optim.AdamW([m], lr=1e-3)

        def bench(opt, runs=10):
            for _ in range(2):
                opt.step()
            if device.type != "cpu":
                getattr(torch, device.type).synchronize()
            t0 = time.perf_counter()
            for _ in range(runs):
                opt.step()
            if device.type != "cpu":
                getattr(torch, device.type).synchronize()
            return (time.perf_counter() - t0) / runs

        t_muon, t_adamw = bench(mu), bench(ad)
        log(f"    {shape}: Muon {t_muon * 1e3:8.2f} ms   AdamW {t_adamw * 1e3:8.2f} ms   ratio {t_muon / max(t_adamw, 1e-12):6.1f}x")
    log()

    # ---- 5. FIT: Muon vs AdamW race on the shell toy ----
    log("[FIT] 80-step least-squares race (same shell, same lr where meaningful):")
    for name, kwargs in (
        ("muon", dict(optimizer_type="muon", lr=5e-2)),
        ("adamw", dict(lr=5e-2)),
    ):
        torch.manual_seed(0)
        model = TinyDraft().to(device)
        opt = BF16Optimizer(model, total_steps=200, lr_scheduler="constant", **kwargs)
        torch.manual_seed(1)
        x = torch.randn(128, 32, device=device)
        y = x @ torch.randn(32, 96, device=device)
        first = last = None
        for _ in range(80):
            loss = torch.nn.functional.mse_loss(model(x), y)
            model.zero_grad()
            loss.backward()
            opt.step()
            last = float(loss)
            if first is None:
                first = last
        log(f"    {name:5s}: loss {first:8.3f} -> {last:8.3f} (ratio {last / first:.3f})")
    log()

    log("# copyable report below " + "-" * 40)
    log("\n".join(REPORT))


if __name__ == "__main__":
    main()
