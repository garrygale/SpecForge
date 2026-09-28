# coding=utf-8
"""Probe (v0): SQNR budget of the target-hidden fusion -- flare (convex) vs fc (matmul).

Companion to ``SQNR_ANALYSIS_V0.md`` (the lens) and ``SQNR_PLAN.md`` (the plan).
It answers one question with numbers: *where* does the per-token activation
quantizer sit relative to the multi-layer mixing, and how many dB does that
placement cost?

Two modes
---------
``--laws``
    Validates the laws the analysis rests on, on synthetic data with known
    ground truth. No config / checkpoint needed. Run this first: it shows which
    assumptions hold exactly, which hold approximately, and -- importantly --
    where the SQNR metric itself stops being informative.

budget mode (default; needs ``--config``)
    Applies the same laws to the real fusion operators of
    ``specforge/modeling/draft/dflash.py``:

      fc    : target_hidden = hidden_norm(fc(concat_j h_j))       # Linear(T*H -> 2560)
      flare : layer_target  = hidden_norm(sum_j w_ij h_j)         # w = softmax(param)

    and produces a per-block ledger (scale disparity + effective bits), the
    measured SQNR at every quantized point on both paths, and the predicted
    SQNR from the laws. The quantizer is mirrored bit-for-bit from
    ``specforge/layers/wxay.py`` (per-token symmetric, qmax = 2^(b-1)-1).

Usage
-----
    # law checks (seconds, CPU, synthetic)
    python probes/probe_sqnr_budget.py --laws

    # budget with random-init weights + synthetic hidden states
    python probes/probe_sqnr_budget.py --config configs/qwen3-8b-domino-dflare-verifiedBase.json --random

    # budget on real hidden states (+ a trained ckpt for the fc W / the fusion w)
    python probes/probe_sqnr_budget.py --config configs/qwen3-8b-domino-dflare-verifiedBase.json \
        --target-model-path <Qwen3-8B> --draft-ckpt <ckpt> --B 32 --S 256

This probe never runs a draft forward pass: it touches only the fusion
operators, so it is fast and CPU-friendly. Reports JSON to --out
(default probes/reports/sqnr_budget.json).
"""

import argparse
import math
import os

import _blas_env  # noqa: F401  (must precede torch: OpenBLAS thread workaround)

import torch
import torch.nn.functional as F

from common import (
    add_device_arg,
    dump_json,
    extract_target_hidden_from_hf,
    load_checkpoint_state,
    load_trained_draft_model,
    pick_device,
    print_report,
    resolve_dtype,
    summarize_tensor,
    synthetic_target_hidden,
)


# ---------------------------------------------------------------------------
# Quantizer mirror + SQNR definitions
# ---------------------------------------------------------------------------

def qmax_of(bits: int) -> int:
    return 2 ** (bits - 1) - 1


def fake_quant_activation(x: torch.Tensor, bits: int):
    """Bit-for-bit mirror of wxay.quantize_activation (per-token symmetric)."""
    qmax = qmax_of(bits)
    scale = x.abs().amax(dim=-1, keepdim=True) / qmax
    scale = scale.clamp(min=1e-6)
    x_int = torch.round(x / scale).clamp(-qmax, qmax)
    return x_int * scale, scale


def energy_sqnr_db(x: torch.Tensor, x_q: torch.Tensor) -> float:
    """SQNR aggregated by ENERGY (total signal power / total noise power).

    This is the aggregate that governs the error energy a downstream matmul
    sees; it is NOT the mean of the per-token dB values (Law 3 note)."""
    x = x.detach().float()
    x_q = x_q.detach().float()
    sig = x.pow(2).sum()
    noise = (x - x_q).pow(2).sum()
    return float(10.0 * torch.log10(sig / noise.clamp(min=1e-30)))


def pertoken_sqnr_mean_db(x: torch.Tensor, x_q: torch.Tensor) -> float:
    """Mean over tokens of the per-token SQNR in dB (the convention already used
    by probes/common.py:summarize_tensor -> sqnr_<b>bit_mean). Reported next to
    the energy SQNR so the two aggregates can be compared."""
    x = x.detach().float()
    x_q = x_q.detach().float()
    sig = x.pow(2).mean(dim=-1)
    noise = (x - x_q).pow(2).mean(dim=-1)
    return float((10.0 * torch.log10(sig / noise.clamp(min=1e-30))).mean())


def rel_distortion(sqnr_db: float) -> float:
    """Relative RMS error 1/sqrt(SQNR) implied by an SQNR in dB."""
    return float(10.0 ** (-sqnr_db / 20.0))


def sqnr_at(x: torch.Tensor, bits_list):
    """Energy / per-token-mean SQNR of a tensor at several bit widths."""
    out = {}
    for b in bits_list:
        x_q, _ = fake_quant_activation(x, b)
        out["sqnr_energy_" + str(b) + "bit_db"] = energy_sqnr_db(x, x_q)
        out["sqnr_pertokenmean_" + str(b) + "bit_db"] = pertoken_sqnr_mean_db(x, x_q)
        out["rel_distortion_" + str(b) + "bit"] = rel_distortion(
            out["sqnr_energy_" + str(b) + "bit_db"])
    return out


def block_effective_bits(x_blocks: torch.Tensor, bits: int):
    """Per-token, per-block effective bits.

    x_blocks: (N, T, H) float. With the per-token step s_t = max_t / qmax:

        b_eff,j(t) = log2( 2 * r_{t,j} / s_t ) = bits - log2( max_t / r_{t,j} )

    decomposed as

        b_eff,j = [bits - log2(max_t / r_max,t)] - log2(r_max,t / r_{t,j})
                  \\______ spikiness tax ______/   \\__ block-disparity tax __/

    A block with b_eff <= 1 has at most one reliable bit: its content is
    destroyed by the shared per-token scale (the "annihilation" regime).
    Returns (beff (N,T), spikiness_tax (N,), disparity_tax (N,T)).
    """
    qmax = qmax_of(bits)
    max_t = x_blocks.abs().amax(dim=(1, 2))                  # (N,)
    r = x_blocks.pow(2).mean(dim=2).sqrt()                   # (N,T)
    s = (max_t / qmax).clamp(min=1e-30)                      # (N,)
    beff = torch.log2((2.0 * r / s.unsqueeze(-1)).clamp(min=1e-30))
    rmax = r.max(dim=1).values
    spikiness_tax = torch.log2((max_t / rmax.clamp(min=1e-30)).clamp(min=1e-30))
    disparity_tax = torch.log2((rmax.unsqueeze(-1) / r.clamp(min=1e-30)).clamp(min=1e-30))
    return beff, spikiness_tax, disparity_tax


def partial_survival_sqnr(x_blocks: torch.Tensor, bits: int, pi: torch.Tensor):
    """The predictive law, applied to any block-structured tensor.

    x_blocks: (N, T, H); pi: (T,) the downstream column-energy share of each
    block (||W_j||_F^2 / ||W||_F^2; pi = [1.0] for a single-block tensor).

    The per-token step is s_t = max_t/qmax, so a coordinate is quantized to a
    nonzero value only if |x| > s_t/2; everything inside that dead zone is
    rounded to zero. For each (token, block) measure empirically

        p_{t,j} = fraction of coordinates surviving       (|x| > s_t/2)
        z_{t,j} = fraction of the block ENERGY inside the dead zone

    then the block-diagonal + uniform-rounding model gives exactly

        signal = sum_{t,j} pi_j * E_{t,j} * (1 - z_{t,j})       retained content
        noise  = sum_{t,j} pi_j * ( p_{t,j} * H * s_t^2/12
                                    + z_{t,j} * E_{t,j} )       rounding + zeroing

    with E_{t,j} the block's energy. This interpolates smoothly between the two
    regimes of Law 1 (isotropic rounding when s_t << r_{t,j}; total annihilation
    when s_t >> r_{t,j}) and needs no Gaussian assumption -- p and z are
    measured. The two things it does NOT model are the off-diagonal (cross-block)
    part of the input covariance and the exact phase of the values against the
    quantizer grid; ``ratio_signal``/``ratio_noise`` and ``pred_minus_meas_db``
    expose exactly that residual, so the law is auditable.

    Returns (sqnr_db, dict). ``destroyed_signal_fraction`` is the share of the
    downstream-weighted content that is not merely noisy but GONE -- the
    quantity the SQNR number alone cannot express.
    """
    N, T, H = x_blocks.shape
    qmax = qmax_of(bits)
    x_blocks = x_blocks.detach().float()
    max_t = x_blocks.abs().amax(dim=(1, 2))                  # (N,)
    s = max_t / qmax                                        # (N,)
    alive = x_blocks.abs() > (s / 2.0).view(N, 1, 1)
    e_tot = x_blocks.pow(2).sum(dim=2)                      # (N,T)
    e_dead = (x_blocks.pow(2) * (~alive).float()).sum(dim=2)
    z = e_dead / e_tot.clamp(min=1e-30)
    p = alive.float().mean(dim=2)                           # (N,T)
    pi_v = pi.view(1, -1)
    signal = (pi_v * e_tot * (1.0 - z)).sum()
    noise = (pi_v * (p * H * (s.pow(2) / 12.0).unsqueeze(-1) + e_dead)).sum()
    signal_full = (pi_v * e_tot).sum()
    sqnr = float(10.0 * torch.log10((signal / noise.clamp(min=1e-30)).clamp(min=1e-30)))
    beff, spikiness_tax, disparity_tax = block_effective_bits(x_blocks, bits)
    return sqnr, {
        "bits": bits,
        "surviving_coord_fraction_mean": float(p.mean()),
        "dead_zone_energy_fraction_mean": float(z.mean()),
        "destroyed_signal_fraction": float((pi_v * e_dead).sum() / signal_full.clamp(min=1e-30)),
        "blocks_with_le_1_eff_bit_mean": float((beff <= 1.0).float().sum(dim=1).mean()),
        "spikiness_tax_bits_mean": float(spikiness_tax.mean()),
        "disparity_tax_bits_mean_per_block": [float(v) for v in disparity_tax.mean(dim=0)],
        "beff_mean_per_block": [float(v) for v in beff.mean(dim=0)],
        "beff_p10_per_block": [float(v) for v in beff.quantile(0.1, dim=0)],
        "signal_retained_rel": float(signal / signal_full.clamp(min=1e-30)),
    }


def measure_and_predict(x_blocks: torch.Tensor, bits: int, pi: torch.Tensor):
    """Measured energy/pertoken SQNR plus the partial-survival law prediction."""
    N, T, H = x_blocks.shape
    flat = x_blocks.reshape(N, T * H)
    x_q, _ = fake_quant_activation(flat, bits)
    meas_e = energy_sqnr_db(flat, x_q)
    meas_p = pertoken_sqnr_mean_db(flat, x_q)
    pred, regime = partial_survival_sqnr(x_blocks, bits, pi)
    regime["sqnr_meas_energy_db"] = meas_e
    regime["sqnr_meas_pertokenmean_db"] = meas_p
    regime["sqnr_pred_db"] = pred
    regime["pred_minus_meas_db"] = float(pred - meas_e)
    regime["rel_distortion_meas"] = rel_distortion(meas_e)
    return regime


def output_sqnr_and_law(x_blocks: torch.Tensor, W: torch.Tensor, bits: int):
    """The fc case: the quantizer sits on the INPUT concat, but the quantity that
    matters downstream is the SQNR at the OUTPUT y = W x (it is what feeds
    hidden_norm -> k_proj_target/v_proj_target).

    Compares the measured output SQNR against the partial-survival law, whose pi
    weights are the column-block energy shares of W. The law's remaining
    approximation is the block-diagonal part of the input covariance; the two
    ratios below expose it.
    """
    N, T, H = x_blocks.shape
    x_blocks = x_blocks.detach().float()
    W = W.detach().float()
    flat = x_blocks.reshape(N, T * H)
    qmax = qmax_of(bits)
    x_q, _ = fake_quant_activation(flat, bits)
    y = F.linear(flat, W)
    y_q = F.linear(x_q, W)
    pi = torch.tensor([float(W[:, j * H:(j + 1) * H].pow(2).sum() / W.pow(2).sum())
                       for j in range(T)])
    pred, stat = partial_survival_sqnr(x_blocks, bits, pi)
    # the two ratios: measured output signal/noise against the model's
    # ||W||_F^2 * (block-weighted signal/noise). Both should be ~1 when the
    # block-diagonal (in x) and column-energy-share (in W) approximations hold.
    max_t = x_blocks.abs().amax(dim=(1, 2))
    s = max_t / qmax
    alive = x_blocks.abs() > (s / 2.0).view(N, 1, 1)
    e_tot = x_blocks.pow(2).sum(dim=2)
    e_dead = (x_blocks.pow(2) * (~alive).float()).sum(dim=2)
    p = alive.float().mean(dim=2)
    pi_v = pi.view(1, -1)
    # per-coordinate (not per-block) energies, so the model is comparable with
    # the measured ||W x||^2 / ||W e||^2: E||W x_t||^2 = sum_j (E_{t,j}/H) ||W_j||_F^2
    sig_model = (pi_v * e_tot * (1.0 - e_dead / e_tot.clamp(min=1e-30))).sum() / H
    noise_model = (pi_v * (p * H * (s.pow(2) / 12.0).unsqueeze(-1) + e_dead)).sum() / H
    w_fro2 = W.pow(2).sum()
    stat["sqnr_out_meas_db"] = energy_sqnr_db(y, y_q)
    stat["sqnr_out_pertokenmean_db"] = pertoken_sqnr_mean_db(y, y_q)
    stat["sqnr_out_pred_db"] = pred
    stat["pred_minus_meas_db"] = float(pred - stat["sqnr_out_meas_db"])
    stat["ratio_signal_meas_over_blockdiag"] = float(y.pow(2).sum() / (w_fro2 * sig_model))
    stat["ratio_noise_meas_over_model"] = float((y - y_q).pow(2).sum() / (w_fro2 * noise_model))
    stat["rel_distortion_out"] = rel_distortion(stat["sqnr_out_meas_db"])
    stat["pi_j_column_energy_share"] = [float(v) for v in pi]
    return stat


# ---------------------------------------------------------------------------
# Law checks (synthetic ground truth)
# ---------------------------------------------------------------------------

def law_1_noise_floor(seed: int = 0):
    """Law 1 -- per-token quantizer noise floor, and where it stops applying.

    For a per-token symmetric quantizer with step s_t = max_t/qmax, a coordinate
    whose magnitude exceeds s_t/2 carries a rounding error that is statistically
    uniform on (-s_t/2, s_t/2), so

        E[(x-Q(x))_i^2 | token t]  <=  s_t^2/12

    with equality up to (a) the fraction of coordinates inside the dead zone
    (|x_i| < s_t/2) and (b) the phase of the values against the grid. Reported:
    measured noise energy per coordinate / (s_t^2/12).

    The third tensor shows the regime the SQNR metric cannot describe: when a
    token is dominated by one spike, every other coordinate is annihilated
    (rounded to zero). The measured SQNR then looks healthy because the residual
    equals the (now destroyed) signal, which is exactly why an effective-bits
    ledger is needed in addition to SQNR.
    """
    g = torch.Generator().manual_seed(seed)
    N, M = 4096, 4096
    rows = []
    for label in ("gaussian", "token-localized spikes", "one spike + annihilated rest"):
        x = torch.randn(N, M, generator=g)
        if label == "token-localized spikes":
            idx = torch.randint(0, M, (N,), generator=g)
            x[torch.arange(N), idx] += 200.0
        elif label == "one spike + annihilated rest":
            x[:, 0] *= 500.0
        for bits in (4, 8):
            x_q, scale = fake_quant_activation(x, bits)
            noise = (x - x_q).pow(2).sum(dim=-1) / M              # per-coordinate
            bound = scale.squeeze(-1).pow(2) / 12.0
            dead = (x.abs() < scale / 2).float().mean()
            rows.append({
                "tensor": label,
                "bits": bits,
                "noise_over_s2_12": float((noise / bound.clamp(min=1e-30)).mean()),
                "coord_zeroed_fraction": float(dead),
                "sqnr_energy_db": energy_sqnr_db(x, x_q),
                "spikiness_mean": float((x.abs().amax(-1) / x.pow(2).mean(-1).sqrt()).mean()),
            })
    return {"law": "1 quantizer noise floor E[e^2] <= s_t^2/12, and its validity domain",
            "rows": rows}


def law_2_matmul_transfer(seed: int = 0):
    """Law 2 -- SQNR transfer through a matmul (the fc path).

    For y = W x with the per-token error e_t, the output error and signal
    energies are sum_t ||W e_t||^2 and sum_t ||W x_t||^2, so under a
    block-diagonal input model and an isotropic noise model

        SQNR_out = sum_j pi_j * sigma_j^2 / [ (s^2/12) + annihilation terms ]

    with pi_j = ||W_j||_F^2 / ||W||_F^2 the column-BLOCK energy share. A matmul
    therefore performs an energy-weighted average of per-block SQNRs: it cannot
    repair a destroyed block and it dilutes good blocks by bad ones. The
    function reports the measured-vs-predicted gap and the correction ratios
    that absorb the residual (off-diagonal input correlations, noise
    anisotropy), so the identity is auditable rather than assumed.
    """
    g = torch.Generator().manual_seed(seed)
    T, H, N = 3, 1024, 2048
    M = T * H
    block_rms = torch.tensor([1.0, 4.0, 16.0])
    x = torch.cat([torch.randn(N, H, generator=g) * block_rms[j] for j in range(T)], dim=-1)
    x_blocks = x.view(N, T, H)
    concat_ref = {}
    for bits in (4, 8):
        x_q, _ = fake_quant_activation(x, bits)
        concat_ref[bits] = energy_sqnr_db(x, x_q)
    rows = []
    for label in ("random uniform W", "W energy on block 0", "W energy on block 2"):
        W = torch.randn(256, M, generator=g) * (1.0 / math.sqrt(M))
        if label == "W energy on block 0":
            W[:, :H] *= 8.0
        elif label == "W energy on block 2":
            W[:, 2 * H:] *= 8.0
        for bits in (4, 8):
            # the quantizer is on the INPUT concat; the metric is the OUTPUT SQNR
            entry = output_sqnr_and_law(x_blocks, W, bits)
            entry["W"] = label
            entry["sqnr_in_concat_meas_db"] = concat_ref[bits]
            rows.append(entry)
    return {"law": "2 matmul SQNR transfer = column-energy-weighted, partial-survival "
                   "block SQNR",
            "rows": rows}


def law_3_concat_penalty(seed: int = 0):
    """Law 3 -- the concat block-scale penalty, and the two SQNR aggregates.

    A per-token quantizer gives every coordinate the same absolute step
    s_t = max_t/qmax, with max_t taken over the WHOLE token vector. Hence

        SQNR_E(concat) = 12 qmax^2 * mean_j(sigma_j^2) / mean_t(max_t^2)

    which degrades as the block scale disparity grows: the quantizer's dynamic
    range is spent on the loudest block and the quiet blocks lose
    log2(r_max/r_j) bits (Law 1's ledger). The same table also shows that the
    per-token-mean SQNR and the energy SQNR are NOT interchangeable: per-token
    spikiness is heavy tailed, so E[1/spik^2] != 1/E[spik]^2.
    """
    g = torch.Generator().manual_seed(seed)
    rows = []
    for disparity in (1.0, 10.0, 100.0):
        T, H, N = 9, 512, 2048
        rms = 4.0 * torch.logspace(0, math.log10(disparity), T)
        x = torch.cat([torch.randn(N, H, generator=g) * rms[j] for j in range(T)], dim=-1)
        x_blocks = x.view(N, T, H)
        pi = torch.full((T,), 1.0 / T)
        for bits in (4, 8):
            qmax = qmax_of(bits)
            # the analytic form uses the per-token mean of the per-coordinate signal
            max_t = x.abs().amax(dim=-1)
            sig_per_coord = x_blocks.pow(2).mean()          # global mean of x_i^2
            sqnr_analytic = float(10.0 * torch.log10(
                (12.0 * qmax ** 2 * sig_per_coord / max_t.pow(2).mean()).clamp(min=1e-30)))
            entry = measure_and_predict(x_blocks, bits, pi)
            entry["disparity_max_over_min_block_rms"] = disparity
            entry["qmax"] = qmax
            entry["sqnr_pred_analytic_concat_db"] = sqnr_analytic
            entry["sqnr_pertokenmean_meas_db"] = entry["sqnr_meas_pertokenmean_db"]
            entry["spikiness_mean"] = float((max_t / x.pow(2).mean(-1).sqrt()).mean())
            rows.append(entry)
    return {"law": "3 concat block-scale penalty + two SQNR aggregates", "rows": rows}


def law_4_convex_dilution(seed: int = 0):
    """Law 4 -- convex (simplex) fusion: norm bound and WHEN dilution exists.

    For y = sum_j w_j h_j with w on the simplex:
      (a) BOUND: max_i |y_i| <= max_j max_i |h_{j,i}| -- a contraction, so the
          convex mix can never amplify a source spike (an fc row can).
      (b) DILUTION: if each layer spikes at an INDEPENDENT (token, dim) pair, the
          fused max is w_max*A while the fused RMS is ||w||_2*RMS, so
              gain = 20 log10(||w||_2 / w_max)  in [0, 10 log10 T].
      (c) COHERENCE LIMIT: if all layers spike at the SAME (token, dim), the
          spikes add coherently: fused max = A*sum_j w_j = A and the fused RMS
          picks up (A sum_j w_j)^2/H, so BOTH scale with sum_j w_j = 1 and the
          gain collapses to ~0 dB.
    Residual streams across neighbouring layers are strongly correlated (the
    budget mode measures the cosine), so real layers sit between (b) and (c).
    """
    g = torch.Generator().manual_seed(seed)
    N, H, T = 2048, 4096, 9
    rows = []

    def one(weights_label, w, spike_label, rho=None):
        h = torch.randn(T, N, H, generator=g)
        if rho is not None:               # layer j = correlated copy of layer 0
            base = h[0].clone()
            for j in range(1, T):
                h[j] = rho * base + math.sqrt(max(1e-12, 1.0 - rho ** 2)) * h[j]
        idx = torch.randint(0, H, (N, T), generator=g)          # per-layer spike dims
        if spike_label == "coherent spikes":
            idx = idx[:, :1].expand(N, T)
        for j in range(T):
            h[j, torch.arange(N), idx[:, j]] += 300.0
        fused = torch.einsum("t,tnh->nh", w, h)
        sp_fused = float((fused.abs().amax(-1) / fused.pow(2).mean(-1).sqrt()).mean())
        dom = int(w.argmax())
        sp_dom = float((h[dom].abs().amax(-1) / h[dom].pow(2).mean(-1).sqrt()).mean())
        # a convex mix cannot exceed its largest source in any norm (Law 4a)
        max_ratio = float(fused.abs().max() / h.abs().amax(dim=(0, 2)).max())
        rows.append({
            "weights": weights_label,
            "spikes": spike_label,
            "layer_correlation_rho": rho,
            "w_max_over_norm": float(w.max() / w.norm()),
            "dilution_gain_pred_db": float(20.0 * math.log10(float(w.norm() / w.max()))),
            "spikiness_dominant_layer": sp_dom,
            "spikiness_fused": sp_fused,
            "dilution_gain_meas_db": float(20.0 * math.log10(sp_dom / max(sp_fused, 1e-30))),
            "max_amplification_meas": max_ratio,
        })

    uniform = torch.full((T,), 1.0 / T)
    onehot = torch.eye(T)[T - 1]
    peaked = torch.full((T,), 0.15 / (T - 1))
    peaked[T - 1] = 0.85
    for wl, w in (("one-hot", onehot), ("uniform", uniform), ("peaked 0.85", peaked)):
        for sl in ("independent spikes", "coherent spikes"):
            one(wl, w, sl)
    for rho in (0.0, 0.5, 0.9, 0.99):
        one("uniform (rho sweep)", uniform, "independent spikes", rho=rho)
    return {"law": "4 convex fusion: contraction bound + dilution 20log10(||w||2/w_max) "
                   "only for incoherent spikes", "rows": rows}


def law_5_rmsnorm_invariance(seed: int = 0):
    """Law 5 -- RMSNorm is per-token SQNR-invariant.

    The quantizer step is max_i|x_i|/qmax and RMSNorm divides each token by its
    own RMS, so the two quantized tensors differ by a positive per-token scalar:
    each token's SQNR is identical. Consequence: what matters is WHERE the norm
    sits relative to WHERE the quantizer sits, not the norm itself. The energy
    aggregate can still drift by ~1 dB when per-token RMS varies, because
    E[max^2] != E[rms^2]*E[spikiness^2].
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(512, 4096, generator=g)
    x[:, 0] *= 50.0
    x_n = x / x.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
    rows = []
    for bits in (4, 8):
        a, _ = fake_quant_activation(x, bits)
        b, _ = fake_quant_activation(x_n, bits)
        rows.append({
            "bits": bits,
            "sqnr_energy_raw_db": energy_sqnr_db(x, a),
            "sqnr_energy_normed_db": energy_sqnr_db(x_n, b),
            "sqnr_pertokenmean_raw_db": pertoken_sqnr_mean_db(x, a),
            "sqnr_pertokenmean_normed_db": pertoken_sqnr_mean_db(x_n, b),
        })
    return {"law": "5 RMSNorm is per-token SQNR-invariant", "rows": rows}


def run_laws(seed: int = 0):
    return {
        "mode": "laws",
        "law_1_noise_floor": law_1_noise_floor(seed),
        "law_2_matmul_transfer": law_2_matmul_transfer(seed),
        "law_3_concat_penalty": law_3_concat_penalty(seed),
        "law_4_convex_dilution": law_4_convex_dilution(seed),
        "law_5_rmsnorm_invariance": law_5_rmsnorm_invariance(seed),
    }


# ---------------------------------------------------------------------------
# Budget mode
# ---------------------------------------------------------------------------

def block_ledger(th_reshaped: torch.Tensor, bits_list):
    """Per-block scale statistics and effective bits at each bit width.

    th_reshaped: (B, S, T, H) float.
    """
    B, S, T, H = th_reshaped.shape
    flat = th_reshaped.reshape(B * S, T, H)
    max_t = flat.abs().amax(dim=(1, 2))
    r = flat.pow(2).mean(dim=2).sqrt()
    global_rms = flat.pow(2).mean(dim=(1, 2)).sqrt()
    out = {
        "tokens": int(B * S),
        "T": int(T),
        "H": int(H),
        "block_rms_mean": [float(v) for v in r.mean(dim=0)],
        "block_rms_min_over_tokens": [float(v) for v in r.min(dim=0).values],
        "block_max_abs": [float(v) for v in flat.abs().amax(dim=(0, 2))],
        "rms_disparity_max_over_min": float(r.mean(dim=0).max() / r.mean(dim=0).min()),
        "max_over_rms_mean": float((max_t / global_rms.clamp(min=1e-12)).mean()),
        "per_bits": {},
    }
    for bits in bits_list:
        beff, spikiness_tax, disparity_tax = block_effective_bits(flat, bits)
        annihilated = (beff <= 1.0).float().sum(dim=1)
        out["per_bits"][str(bits)] = {
            "beff_mean_per_block": [float(v) for v in beff.mean(dim=0)],
            "beff_min_per_block": [float(v) for v in beff.min(dim=0).values],
            "spikiness_tax_bits_mean": float(spikiness_tax.mean()),
            "disparity_tax_bits_mean_per_block": [float(v) for v in disparity_tax.mean(dim=0)],
            "blocks_with_le_1_bit_mean": float(annihilated.mean()),
            "blocks_with_le_1_bit_p90": float(annihilated.quantile(0.9)),
            "blocks_with_le_0_bits_mean": float((beff <= 0.0).float().sum(dim=1).mean()),
        }
    return out


def run_budget(args):
    import json as _json

    device = pick_device(args.device)
    dtype = resolve_dtype(args.dtype)
    torch.manual_seed(args.seed)
    # All SQNR arithmetic runs on CPU: it is elementwise plus one matmul, and
    # pinning the models and the statistics to one device avoids dtype/device
    # churn (same choice as probe_fusion_stats). ``device`` is still used for the
    # heavy target-model forward that produces the hidden states.
    stat_device = "cpu"
    report = {"mode": "budget", "args": vars(args), "device": device,
              "stat_device": stat_device, "dtype": args.dtype}
    cfg = _json.load(open(args.config, encoding="utf-8"))
    dc = cfg["dflash_config"]

    # ---------------- target hidden states ----------------
    if args.target_model_path:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print("[target] loading " + args.target_model_path)
        tok = AutoTokenizer.from_pretrained(args.target_model_path)
        tgt = AutoModelForCausalLM.from_pretrained(
            args.target_model_path, torch_dtype=dtype, device_map=None
        ).eval().to(device)
        texts = []
        if args.texts_file and os.path.exists(args.texts_file):
            with open(args.texts_file, "r", encoding="utf-8") as f:
                texts = [line.strip() for line in f if line.strip()][: args.num_texts]
        if not texts:
            texts = [
                "The theory of speculative decoding dates back to",
                "Quantization of neural networks requires",
                "In a large language model, the residual stream",
                "Block diffusion models predict all tokens at once.",
            ] * max(1, args.num_texts // 4)
        layer_ids = dc["target_layer_ids"]
        th, _ = extract_target_hidden_from_hf(tgt, tok, texts, layer_ids, device, dtype, args.max_len)
        th = th.cpu()
        S = min(th.shape[1], args.S)
        target_hidden = th[:, :S].contiguous()
        report["target"] = {"model": args.target_model_path, "layer_ids": layer_ids,
                            "B": int(target_hidden.shape[0]), "S": int(target_hidden.shape[1])}
    else:
        T0 = len(dc["target_layer_ids"])
        H0 = dc.get("target_hidden_size", cfg.get("hidden_size"))
        target_hidden = synthetic_target_hidden(
            args.B, args.S, T0, H0, seed=args.seed, device=stat_device, dtype=dtype)
        report["target"] = {"synthetic": True, "T": T0, "H": H0, "B": args.B, "S": args.S}

    B, S, TH = target_hidden.shape
    T = len(dc["target_layer_ids"])
    H = TH // T
    bits_list = [int(b) for b in args.bits.split(",")]
    report["config"] = {
        "path": args.config, "T": T, "H": H, "hidden_size": cfg["hidden_size"],
        "num_draft_layers": cfg["num_hidden_layers"],
        "target_layer_ids": dc["target_layer_ids"],
        "fusion_mode_in_config": dc.get("fusion_mode"),
        "heterogeneous_kv": dc.get("heterogeneous_kv"),
        "qat_w_bit": dc.get("qat_w_bit"), "qat_a_bit": dc.get("qat_a_bit"),
        "qat_w4a4_layers": dc.get("qat_w4a4_layers"),
    }

    N = B * S
    x = target_hidden.float().reshape(B, S, T, H)
    x_blocks = x.reshape(N, T, H)
    flat_concat = x_blocks.reshape(N, T * H)

    # ---------------- the block ledger ----------------
    report["block_ledger"] = block_ledger(x, bits_list)
    report["concat_raw_stats"] = summarize_tensor(flat_concat, "concat_fc_input",
                                                  bits_iter=bits_list)

    # ---------------- concat = the fc input ----------------
    fc_model, _ = load_trained_draft_model(
        args.config, args.draft_ckpt, fusion_mode="fc", dtype=dtype, device=stat_device)
    W = fc_model.fc.weight.detach().float()
    pi = torch.tensor([float(W[:, j * H:(j + 1) * H].pow(2).sum() / W.pow(2).sum())
                       for j in range(T)])
    fc_from_ckpt = None
    if args.draft_ckpt:
        sd = load_checkpoint_state(args.draft_ckpt, device="cpu")
        fc_from_ckpt = "fc.weight" in sd
        if not fc_from_ckpt:
            print("[warn] checkpoint has no 'fc.weight' -> fc W is RANDOM INIT. "
                  "Point --draft-ckpt at an fc-mode checkpoint for trained weights.")
    report["fc_path"] = {
        "fc_weight_shape": list(W.shape),
        "fc_weight_from_checkpoint": fc_from_ckpt,
        "column_energy_share_pi_j": [float(v) for v in pi],
        "fc_row_norm_over_concat_norm": float(
            W.norm() / math.sqrt(W.shape[0]) / math.sqrt(T * H)),
        "quantized_points": {},
    }
    for bits in bits_list:
        key = "fc_input_" + str(bits) + "bit"
        # the quantized point is the concat; the metric that matters is the OUTPUT
        in_pt = measure_and_predict(x_blocks, bits, pi)
        entry = output_sqnr_and_law(x_blocks, W, bits)
        entry["sqnr_at_quantized_point_concat_db"] = in_pt["sqnr_meas_energy_db"]
        entry["sqnr_at_quantized_point_concat_pertokenmean_db"] = \
            in_pt["sqnr_meas_pertokenmean_db"]
        report["fc_path"]["quantized_points"][key] = entry

    # ---------------- flare path ----------------
    model, _ = load_trained_draft_model(
        args.config, args.draft_ckpt, fusion_mode="flare", dtype=dtype, device=stat_device)
    if args.random:
        model._init_fusion_weights()
    w = F.softmax(model.layer_fusion_weights.detach().float(), dim=1)   # (D, T)
    D = w.shape[0]
    report["flare_path"] = {
        "fusion_weights_softmax": [[float(v) for v in w[i]] for i in range(D)],
        "entropy_per_layer_nats": [float(-(w[i] * w[i].clamp(min=1e-12).log()).sum())
                                   for i in range(D)],
        "entropy_uniform_nats": float(math.log(T)),
        "w_max_over_norm_per_layer": [float(w[i].max() / w[i].norm()) for i in range(D)],
        "dilution_gain_pred_db_per_layer": [
            float(20.0 * math.log10(float(w[i].norm() / w[i].max()))) for i in range(D)],
        "per_layer": {},
    }
    # Inter-layer similarity decides whether Law 4(b) dilution is available: if
    # two layers are near-collinear, blending them is a rescaling of the same
    # signal (Law 4(c)), not an average of independent spikes.
    v = x_blocks - x_blocks.mean(dim=2, keepdim=True)          # center per token
    u = v / v.pow(2).sum(dim=2, keepdim=True).sqrt().clamp(min=1e-30)
    pair = torch.einsum("nth,nsh->nts", u, u)                  # (N,T,T) per token
    off_diag = pair.sum() - pair.diagonal(dim1=1, dim2=2).sum()
    report["flare_path"]["inter_layer_cosine_mean"] = float(off_diag / (N * T * (T - 1)))
    report["flare_path"]["inter_layer_cosine_matrix"] = [
        [float(v) for v in pair.mean(dim=0)[t]] for t in range(T)]

    one = torch.ones(1)
    for i in range(D):
        raw = (x * w[i].view(1, 1, -1, 1)).sum(dim=2)             # (B,S,H)
        normed = model.hidden_norm(raw.to(dtype=dtype)).float()
        flat_fused = normed.reshape(N, 1, H)
        dom = int(w[i].argmax())
        src = x[:, :, dom, :].reshape(N, 1, H)
        entry = {
            "dominant_source_layer_index": dom,
            "dominant_weight": float(w[i][dom]),
            "fused_raw": sqnr_at(raw.reshape(N, H), bits_list),
            "dominant_source_alone": {},
            "dilution_gain_pred_db": float(20.0 * math.log10(float(w[i].norm() / w[i].max()))),
            "per_bits": {},
        }
        for b in bits_list:
            fused_m = measure_and_predict(flat_fused, b, one)
            src_m = measure_and_predict(src, b, one)
            entry["per_bits"][str(b)] = {
                "fused_measured": fused_m,
                "dominant_source_alone": src_m,
                "dilution_gain_meas_db": float(
                    fused_m["sqnr_meas_energy_db"] - src_m["sqnr_meas_energy_db"]),
                "fused_blocks_with_le_1_eff_bit_mean": fused_m["blocks_with_le_1_eff_bit_mean"],
                "fused_surviving_coord_fraction_mean": fused_m["surviving_coord_fraction_mean"],
            }
            entry["dominant_source_alone"][str(b)] = src_m
        report["flare_path"]["per_layer"]["draft_layer_" + str(i)] = entry

    # ---------------- the headline comparison ----------------
    for b in bits_list:
        per_layer = report["flare_path"]["per_layer"]
        fused_mean = float(sum(
            per_layer["draft_layer_" + str(i)]["per_bits"][str(b)]["fused_measured"]
            ["sqnr_meas_energy_db"] for i in range(D)) / D)
        fused_destroyed = float(sum(
            per_layer["draft_layer_" + str(i)]["per_bits"][str(b)]["fused_measured"]
            ["destroyed_signal_fraction"] for i in range(D)) / D)
        cat = report["fc_path"]["quantized_points"]["fc_input_" + str(b) + "bit"]
        report["headline_" + str(b) + "bit"] = {
            "flare_fused_sqnr_energy_db_mean_over_layers": fused_mean,
            "flare_fused_rel_distortion": rel_distortion(fused_mean),
            "flare_fused_destroyed_signal_fraction_mean": fused_destroyed,
            "fc_concat_sqnr_at_quantized_point_db": cat["sqnr_at_quantized_point_concat_db"],
            "fc_concat_sqnr_pertokenmean_db": cat["sqnr_at_quantized_point_concat_pertokenmean_db"],
            "fc_concat_spikiness_mean": report["concat_raw_stats"]["spikiness_mean"],
            "fc_concat_destroyed_signal_fraction": cat["destroyed_signal_fraction"],
            "fc_concat_blocks_with_le_1_eff_bit_mean": cat["blocks_with_le_1_eff_bit_mean"],
            "fc_output_sqnr_energy_db_measured": cat["sqnr_out_meas_db"],
            "fc_output_sqnr_energy_db_predicted": cat["sqnr_out_pred_db"],
            "fc_output_rel_distortion": cat["rel_distortion_out"],
            "fc_output_pred_minus_meas_db": cat["pred_minus_meas_db"],
            "fc_pi_j_column_energy_share": cat["pi_j_column_energy_share"],
            "sqnr_gap_flare_minus_fc_output_db": float(fused_mean - cat["sqnr_out_meas_db"]),
            "note": ("the quantized point differs: fc quantizes the pre-mixing concat "
                     "(9 blocks, one shared per-token scale), flare quantizes the "
                     "post-mixing single-block fused tensor. The gap is the "
                     "block-disparity tax plus the dead-zone content loss."),
        }

    return report


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--laws", action="store_true",
                    help="run the law checks on synthetic ground truth (no config needed)")
    ap.add_argument("--config", default=None, help="draft config JSON path")
    ap.add_argument("--draft-ckpt", default=None, help="trained draft checkpoint (dir or file)")
    ap.add_argument("--random", action="store_true",
                    help="re-init the fusion weights to the schedule (deterministic)")
    ap.add_argument("--target-model-path", default=None, help="HF target model for real hidden states")
    ap.add_argument("--texts-file", default=None)
    ap.add_argument("--num-texts", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--S", type=int, default=128)
    ap.add_argument("--bits", default="4,8", help="activation bit widths to sweep")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="JSON output path")
    add_device_arg(ap)
    args = ap.parse_args()

    if args.laws:
        report = run_laws(args.seed)
        out_path = args.out or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "reports", "sqnr_laws.json")
    else:
        if not args.config:
            ap.error("--config is required unless --laws is given")
        report = run_budget(args)
        out_path = args.out or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "reports", "sqnr_budget.json")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    dump_json(out_path, report)
    print_report(report)


if __name__ == "__main__":
    main()
