# coding=utf-8
"""CP energy spectrum of a staggered (outer-pairing) draft FFN readout.

Pre-experiment for replacing the down projection of an ``ffn_sharing``
outer-layout draft with a CP factorization (a bilinear/FM-style readout
``y = sum_k w_k (v_k.u)(g_k.a)``).  The question this answers *before any
training*: how much of the deployed readout's energy does a rank-R CP
approximation capture?

The deployed bilinear map is reconstructed exactly from the checkpoint:

* dense staggered:  ``W_eff = down_proj.weight``            (d x I)
* staggered+folded: ``W_eff = down_proj.proj.weight @ A``   (A = the tied
  softmax chunk-mixture matrix rebuilt from ``down_proj.fold_logits``)

then reshaped under the outer pairing (channel j = q*(G_u*G_g) + c*G_g + r,
repetitions summed over q) into ``T in R^{d x G_u x G_g}``, and

* a CP-ALS sweep reports the captured-energy curve over R,
* a matrix-SVD sweep on ``W_eff`` reports the plain low-rank fallback curve,
* the 95%-at-R=1024 rule prints a GO / REFINED-GO / PARK verdict.

Nested pairing is refused: only the outer lattice makes the intermediate a
Kronecker product, so the tensor view (and this whole avenue) is exclusive to
``pairing: "outer"``.

Run where the checkpoint lives (the sweep prints a Tflops estimate, one
progress line per completed (layer, rank) step with live ETA, and — under
--verbose — per-iteration ALS heartbeats; the R=2048 sweeps take ~minutes on
GPU, too slow on laptop CPUs.  --iters 15 gives a quicker first pass: the
HOSVD init converges fast and the energy curve moves little)::

    python scripts/staggered_cp_spectrum.py --checkpoint exported-draft/ \
        [--ranks 128,256,512,1024,2048] [--iters 25] [--device cuda] [--layers all]

A synthetic self-test (no checkpoint needed) validates the mixture-matrix
reconstruction against SpecForge's module and the ALS against a planted rank::

    python scripts/staggered_cp_spectrum.py --synthetic
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

GO_RULE_RANK = 1024
GO_RULE_ENERGY = 0.95


# ---------------------------------------------------------------------------
# checkpoint loading
# ---------------------------------------------------------------------------


def _load_state_dict(checkpoint: str) -> dict:
    path = pathlib.Path(checkpoint)
    if path.is_dir():
        shards = sorted(path.glob("*.safetensors")) or [path / "model.safetensors"]
        if shards and shards[0].is_file():
            from safetensors.torch import load_file

            state = {}
            for shard in shards:
                state.update(load_file(str(shard)))
            return state
        candidates = sorted(path.glob("*.bin")) + sorted(path.glob("*.pt"))
        if not candidates:
            raise FileNotFoundError(f"no weight file under {path}")
        path = candidates[0]
    blob = torch.load(str(path), map_location="cpu", weights_only=True)
    if isinstance(blob, dict) and "draft_state_dict" in blob:
        blob = blob["draft_state_dict"]
    return blob


def _normalize_keys(state: dict) -> dict:
    out = {}
    for name, tensor in state.items():
        for prefix in ("model.",):
            if name.startswith(prefix):
                name = name[len(prefix):]
        out[name] = tensor
    return out


# ---------------------------------------------------------------------------
# deployed effective readout
# ---------------------------------------------------------------------------


def build_fold_mixture_matrix(fold_logits: torch.Tensor, intermediate_size: int):
    """The tied softmax chunk mixture ``A`` (s x I) used by FoldedSoftmaxReadout.

    mixed[q] = sum_b softmax(logits)[b, q % K] * h[b * s + q],  s = I / c.
    """

    weights = torch.softmax(fold_logits.float(), dim=0)  # (c, K)
    branches, granularity = weights.shape
    folded_size = intermediate_size // branches
    A = torch.zeros(
        folded_size, intermediate_size, dtype=weights.dtype, device=weights.device
    )
    q = torch.arange(folded_size, device=weights.device)
    for branch in range(branches):
        A[q, branch * folded_size + q] = weights[branch, q % granularity]
    return A


def effective_readout(layer_prefix: str, state: dict, intermediate_size: int):
    """Return ``(W_eff [d, I], kind)`` for one layer's down projection."""

    dense_key = f"{layer_prefix}.down_proj.weight"
    proj_key = f"{layer_prefix}.down_proj.proj.weight"
    logits_key = f"{layer_prefix}.down_proj.fold_logits"
    if torch.is_tensor(state.get(dense_key)):
        return state[dense_key].float(), "dense"
    if torch.is_tensor(state.get(proj_key)):
        if not torch.is_tensor(state.get(logits_key)):
            raise KeyError(f"folded readout without {logits_key}")
        A = build_fold_mixture_matrix(state[logits_key], intermediate_size)
        return state[proj_key].float() @ A, "folded"
    raise KeyError(f"no down projection under {layer_prefix} (tried {dense_key} and {proj_key})")


def outer_tensor(W_eff: torch.Tensor, gate_groups: int, up_groups: int):
    """Reshape [d, I] into T [d, G_u, G_g] under the outer pairing.

    Channel j = q*(G_u*G_g) + c*G_g + r; the repetition factor q is summed
    away (exact: the deployed map only ever sees the sum over repetitions).
    """

    d, I = W_eff.shape
    if I % (gate_groups * up_groups):
        raise ValueError(
            f"gate_groups*up_groups={gate_groups * up_groups} must divide I={I}"
        )
    repeats = I // (gate_groups * up_groups)
    return (
        W_eff.view(d, repeats, up_groups, gate_groups)
        .sum(dim=1)
        .reshape(d, up_groups, gate_groups)
        .contiguous()
    )


# ---------------------------------------------------------------------------
# spectra
# ---------------------------------------------------------------------------


def svd_capture_curve(W: torch.Tensor, ranks) -> list:
    singular = torch.linalg.svdvals(W) ** 2
    total = singular.sum().item()
    return [float(singular[: min(r, singular.numel())].sum().item() / total) for r in ranks]


def _khatri_rao(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # rows: A's row index slow, B's row index fast  (matches reshape(d, G_u, G_g))
    return (A.unsqueeze(1) * B.unsqueeze(0)).reshape(A.shape[0] * B.shape[0], -1)


def _left_vectors(M: torch.Tensor, count: int, generator) -> torch.Tensor:
    U, _, _ = torch.linalg.svd(M, full_matrices=False)
    if U.shape[1] >= count:
        return U[:, :count]
    # R can exceed a mode's dimension (e.g. R=2048 > G_g=76), where an
    # orthonormal padding is impossible; normalized random columns suffice
    # as an ALS init.  The CPU generator keeps the seed reproducible across
    # devices; move the sample to M's device before mixing it in.
    extra = torch.randn(
        M.shape[0], count - U.shape[1], generator=generator, dtype=M.dtype
    ).to(M.device)
    extra = extra / extra.norm(dim=0, keepdim=True)
    return torch.cat([U, extra], dim=1)


def init_factors(T: torch.Tensor, max_rank: int, seed: int = 0):
    """HOSVD left bases (with normalized random padding up to ``max_rank``).

    Built once per layer: the rank sweep truncates columns for each R instead
    of re-running the three SVDs for every (layer, R) combination.
    """

    d, Gu, Gg = T.shape
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return (
        _left_vectors(T.reshape(d, -1), max_rank, generator),
        _left_vectors(T.permute(1, 0, 2).reshape(Gu, -1), max_rank, generator),
        _left_vectors(T.permute(2, 0, 1).reshape(Gg, -1), max_rank, generator),
    )


def cp_als(
    T: torch.Tensor,
    R: int,
    iters: int,
    bases=None,
    seed: int = 0,
    heartbeat=None,
):
    """CP-ALS with HOSVD init; returns factors (W, V, G).

    ``bases`` reuses :func:`init_factors` output (truncated to R); pass a
    ``heartbeat(iteration)`` callback for intra-run progress.
    """

    d, Gu, Gg = T.shape
    if bases is None:
        bases = init_factors(T, R, seed=seed)
    W, V, G = (basis[:, :R].contiguous() for basis in bases)
    T1 = T.reshape(d, -1)
    T2 = T.permute(1, 0, 2).reshape(Gu, -1)
    T3 = T.permute(2, 0, 1).reshape(Gg, -1)

    def solve(unfolding, factors_slow, factors_fast):
        # factor = unfolding @ Z @ (Z^T Z + ridge I)^{-1}
        Z = _khatri_rao(factors_slow, factors_fast)
        ZtZ = Z.T @ Z
        ridge = 1e-9 * ZtZ.diagonal().mean().clamp(min=1e-12)
        ZtZ = ZtZ + ridge * torch.eye(ZtZ.shape[0], dtype=Z.dtype, device=Z.device)
        return unfolding @ Z @ torch.linalg.inv(ZtZ)

    for iteration in range(iters):
        W = solve(T1, V, G)
        V = solve(T2, W, G)
        G = solve(T3, W, V)
        if heartbeat is not None:
            heartbeat(iteration)
    return W, V, G


def cp_captured_energy(T: torch.Tensor, factors) -> float:
    """1 - ||T - T_R||_F^2 / ||T||_F^2 without materializing T_R."""

    W, V, G = factors
    total = T.reshape(-1) @ T.reshape(-1)
    cross = ((T.reshape(T.shape[0], -1) @ _khatri_rao(V, G)) * W).sum()
    Gw, Gv, Gg = W.T @ W, V.T @ V, G.T @ G
    approx = (Gw * Gv * Gg).sum()
    residual = (total - 2 * cross + approx).clamp(min=0)
    return float(1 - residual / total)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def _load_config(args) -> dict:
    if args.config:
        return json.loads(pathlib.Path(args.config).read_text(encoding="utf-8"))
    inline = pathlib.Path(args.checkpoint) / "config.json"
    if inline.is_file():
        return json.loads(inline.read_text(encoding="utf-8"))
    raise SystemExit("need --config (no config.json next to the checkpoint)")


def run_checkpoint(args) -> None:
    config = _load_config(args)
    dflash = config.get("dflash_config") or {}
    sharing = dflash.get("ffn_sharing") or {}
    if not sharing:
        raise SystemExit("config has no dflash_config.ffn_sharing entry")
    pairing = sharing.get("pairing", "nested")
    if pairing != "outer":
        raise SystemExit(
            "the CP avenue needs pairing='outer' (the tensor view exists only "
            "for the lattice layout); this checkpoint is "
            f"pairing={pairing!r}"
        )
    gate_groups = int(sharing["gate_groups"])
    up_groups = int(sharing["up_groups"])
    intermediate = int(config["intermediate_size"])
    ranks = [int(r) for r in args.ranks.split(",")]

    state = _normalize_keys(_load_state_dict(args.checkpoint))
    layers = sorted(
        {
            int(key.split(".")[1])
            for key in state
            if key.startswith("layers.") and ".mlp.down_proj" in key
        }
    )
    if args.layers != "all":
        keep = {int(x) for x in args.layers.split(",")}
        layers = [i for i in layers if i in keep]
    if not layers:
        raise SystemExit("no layers.*.mlp.down_proj weights found in the checkpoint")

    device = torch.device(args.device)
    print(
        f"outer pairing {gate_groups}x{up_groups} (m={intermediate // (gate_groups * up_groups)}), "
        f"I={intermediate}, layers={layers}, device={device}, ranks={ranks}"
    )
    header = ["layer", "kind"] + [f"CP@{r}" for r in ranks]
    rows = [header]
    verdicts = []
    def _fmt(seconds: float) -> str:
        seconds = int(seconds)
        if seconds < 60:
            return f"{seconds}s"
        return f"{seconds // 60}m{seconds % 60:02d}s"

    total_steps = len(layers) * len(ranks)
    hidden = int(config["hidden_size"])
    macs_per_rank_iter = (
        hidden * intermediate  # mode-1 solve: d x I against I x R
        + 2 * hidden * up_groups * gate_groups  # mode-2/3 solves (small)
    )
    total_macs = (
        len(layers) * args.iters * sum(ranks) * macs_per_rank_iter
    )
    print(
        f"sweep: {len(layers)} layers x {len(ranks)} ranks x {args.iters} ALS "
        f"iters = {total_steps} steps, ~{2 * total_macs / 1e12:.0f} Tflops of "
        "GEMM work (plus one SVD-basis build per layer)"
    )
    t_sweep = time.perf_counter()
    done = 0
    for layer in layers:
        prefix = f"layers.{layer}.mlp"
        W_eff, kind = effective_readout(prefix, state, intermediate)
        W_eff = W_eff.to(device)
        T = outer_tensor(W_eff, gate_groups, up_groups)
        t0 = time.perf_counter()
        bases = init_factors(T, max(ranks), seed=layer * 1000 + max(ranks))
        print(
            f"[layer {layer}] {kind} readout reconstructed, HOSVD bases ready "
            f"({_fmt(time.perf_counter() - t0)})",
            flush=True,
        )
        captures = []
        for R in ranks:
            t0 = time.perf_counter()

            def heartbeat(iteration, _t0=t0, _R=R):
                stride = max(1, args.iters // 5)
                if (iteration + 1) % stride == 0:
                    print(
                        f"    layer {layer} R={_R}: ALS iter {iteration + 1}/"
                        f"{args.iters} ({time.perf_counter() - _t0:.1f}s)",
                        flush=True,
                    )

            factors = cp_als(
                T, R, args.iters, bases=bases, heartbeat=heartbeat if args.verbose else None
            )
            captures.append(cp_captured_energy(T, factors))
            done += 1
            elapsed = time.perf_counter() - t_sweep
            eta = elapsed / done * (total_steps - done)
            print(
                f"[{done}/{total_steps}] layer {layer} R={R}: captured "
                f"{captures[-1]:.4f} ({time.perf_counter() - t0:.1f}s, elapsed "
                f"{_fmt(elapsed)}, eta {_fmt(eta)})",
                flush=True,
            )
        svd_curve = svd_capture_curve(W_eff, ranks)
        rows.append(
            [str(layer), kind] + [f"{c:.3f}" for c in captures]
        )
        verdicts.append((layer, dict(zip(ranks, captures)), svd_curve))
    print(f"sweep done in {_fmt(time.perf_counter() - t_sweep)}\n", flush=True)

    width = max(len(cell) for row in rows for cell in row)
    for row in rows:
        print("  ".join(cell.rjust(width) for cell in row))

    print("\nper-layer matrix-SVD fallback curve (plain low-rank on W_eff):")
    for layer, _, svd_curve in verdicts:
        print(
            f"  layer {layer}: "
            + "  ".join(f"@{r}:{c:.3f}" for r, c in zip(ranks, svd_curve))
        )

    print(f"\nverdict ({GO_RULE_ENERGY:.0%} CP energy rule):")
    rule_rank = GO_RULE_RANK if GO_RULE_RANK in ranks else max(ranks)
    top_rank = max(ranks)
    worst = min(curve.get(rule_rank, 0.0) for _, curve, _ in verdicts)
    at_top = min(curve.get(top_rank, 0.0) for _, curve, _ in verdicts)
    if worst >= GO_RULE_ENERGY:
        print(
            f"  GO: every layer keeps >= {GO_RULE_ENERGY:.0%} energy at "
            f"R={rule_rank} (worst {worst:.1%}) — CP is a viable "
            "replacement readout; warm-start from the checkpoint's factors."
        )
    elif at_top >= GO_RULE_ENERGY:
        print(
            f"  REFINED GO: R={rule_rank} is short (worst {worst:.1%}) but "
            f"R={top_rank} holds ({at_top:.1%}) — target the larger rank or "
            "accept a short distill."
        )
    else:
        print(
            f"  PARK: even R={top_rank} loses too much (worst "
            f"{at_top:.1%}) — the bilinear form is not CP-concentrated; keep "
            "the current readout and write this line up as future work."
        )


def run_synthetic(args) -> None:
    """Validate the pipeline: exact mixture reconstruction + planted CP rank."""

    torch.manual_seed(0)
    try:
        from specforge.modeling.draft.dflash_kernels import FoldedSoftmaxReadout
    except ImportError:
        print("fold mixture check skipped (folded readout retired; tag-era "
              "checkpoints still readable)")
        FoldedSoftmaxReadout = None

    d, I = 32, 128
    readout = FoldedSoftmaxReadout(d, I, branches=4, granularity=8)
    with torch.no_grad():
        readout.fold_logits.normal_(std=1.5)
        readout.proj.weight.normal_()
    h = torch.randn(3, 7, I)
    A = build_fold_mixture_matrix(readout.fold_logits.data, I)
    # mixed = A @ h, then out = proj(mixed): compare against the module.
    via_A = (A @ h.reshape(-1, I).T).T @ readout.proj.weight.T
    direct = readout(h).reshape(-1, d)
    err = (via_A - direct).abs().max().item()
    print(f"fold mixture reconstruction: max |A-path - module| = {err:.2e}")
    assert err < 1e-4, "mixture matrix does not match FoldedSoftmaxReadout"

    # Planted CP rank 4 + 1% noise on a staggered-shaped tensor.
    Gu, Gg = 16, 8
    rank = 4
    Wp = torch.randn(d, rank)
    Vp = torch.randn(Gu, rank)
    Gp = torch.randn(Gg, rank)
    T = torch.einsum("qr,cr,gr->qcg", Wp, Vp, Gp)
    noise = 0.01 * T.norm() / math.sqrt(T.numel())
    T = T + noise * torch.randn_like(T)
    W_full = T.reshape(d, Gu * Gg)
    captures = {}
    for R in (2, 4, 8):
        captures[R] = cp_captured_energy(T, cp_als(T, R, iters=200))
    print("planted-rank recovery:", {k: f"{v:.4f}" for k, v in captures.items()})
    assert captures[2] < 0.95, "rank-2 should not capture a rank-4 tensor"
    assert captures[4] > 0.99, "rank-4 should recover the planted tensor"
    print("synthetic self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", help="HF export dir / .safetensors / SpecForge training checkpoint")
    parser.add_argument("--config", help="draft config.json (default: <checkpoint>/config.json)")
    parser.add_argument("--ranks", default="128,256,512,1024,2048")
    parser.add_argument("--iters", type=int, default=25)
    parser.add_argument("--device", default=None)
    parser.add_argument("--layers", default="all", help="'all' or comma-separated indices")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        if getattr(torch, "npu", None) is not None:
            try:
                if torch.npu.is_available():
                    args.device = "npu"
            except Exception:
                pass

    if args.synthetic:
        run_synthetic(args)
    else:
        if not args.checkpoint:
            parser.error("--checkpoint is required (or use --synthetic)")
        run_checkpoint(args)


if __name__ == "__main__":
    main()
