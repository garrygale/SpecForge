# coding=utf-8
"""S1a: ledger of the OFFICIAL trained fusion weights (no target model needed).

Reads the tensors fetched by ``fetch_hf_tensors.py`` and answers the two
weight-side questions the SQNR lens raises. It needs neither the Qwen3-8B target
nor a GPU: everything here is a property of the fusion parameters.

1. DFlare (``AngelSlim/Qwen3-8b-dflare``, T=9, ``layer_fusion_weights`` [7,9]):
   how much mixing did training actually learn, and therefore how much of Law 4's
   dilution is available at all? Per draft layer: softmax entropy, ``w_max/||w||_2``,
   ``20 log10(||w||_2/w_max)`` dB of available dilution, dominant tap.

2. DFlash (``shanjiaz/dflash-qwen3-8b``, T=3, ``fc.weight`` [4096,12288]):
   where does the BF16 task optimum put its column energy (``pi_j``), and what does
   Law 2 then predict for the fc OUTPUT SQNR once the concat is per-token
   quantized? The block RMS profile ``u_j = sigma_j/sigma_max`` is NOT in the
   checkpoint, so the script reports SQNR as a function of the assumed per-token
   max-to-RMS ratio ``rho`` and depth ramp, i.e. a bracket that S1b pins down with
   real hidden states.

Usage:
    python probes/sqnr/analyze_fusion_weights.py --dir <fetched tensor dir>
    python probes/sqnr/analyze_fusion_weights.py --dir ... --out probes/sqnr/reports/fusion_weights.json

Tensor files are matched by name, so the directory may hold a partial fetch.
"""

import argparse
import glob
import json
import math
import os

import torch


def find_one(directory: str, *patterns):
    for pat in patterns:
        hits = sorted(glob.glob(os.path.join(directory, pat)))
        if hits:
            return hits[0]
    return None


def dflare_fusion_ledger(path: str, target_layer_ids):
    raw = torch.load(path).float()
    soft = torch.softmax(raw, dim=1)
    rows = []
    for i in range(soft.shape[0]):
        w = soft[i]
        ent = float(-(w * w.clamp(min=1e-12).log()).sum())
        r = float(w.max() / w.norm())
        dom = int(w.argmax())
        rows.append({
            "draft_layer": i,
            "entropy_nats": ent,
            "entropy_uniform_nats": math.log(soft.shape[1]),
            "w_max_over_norm": r,
            "available_dilution_db": float(20.0 * math.log10(1.0 / r)),
            "dominant_tap": dom,
            "dominant_target_layer": int(target_layer_ids[dom]),
            "softmax_weights": [float(v) for v in w],
            "raw_logit_spread": float(raw[i].max() - raw[i].min()),
            "nonzero_taps_1pct": [int(j) for j in range(soft.shape[1]) if float(w[j]) >= 0.01],
        })
    used = sorted({r["dominant_tap"] for r in rows})
    return {
        "checkpoint": os.path.basename(path),
        "shape": list(raw.shape),
        "target_layer_ids": list(target_layer_ids),
        "per_draft_layer": rows,
        "mean_entropy_nats": float(sum(r["entropy_nats"] for r in rows) / len(rows)),
        "mean_available_dilution_db": float(
            sum(r["available_dilution_db"] for r in rows) / len(rows)),
        "max_available_dilution_db": float(max(r["available_dilution_db"] for r in rows)),
        "dominant_taps_used": used,
        "dominant_target_layers_used": [int(target_layer_ids[t]) for t in used],
        "taps_available": int(raw.shape[1]),
        "note": ("available dilution is an UPPER bound: Law 4 shows it is only "
                 "collected when the fused layers spike incoherently, and the "
                 "measured realised gain on the trained model was ~0 dB"),
    }


def spectral_norm_power(W: torch.Tensor, iters: int = 12) -> float:
    """Spectral norm by power iteration.

    ``torch.linalg.svdvals`` on the 4096x12288 fc weight is minutes of CPU per
    call; 12 matvecs give the leading singular value to ~1e-6 relative error for
    a matrix this dense and are ~100x cheaper.
    """
    v = torch.randn(W.shape[1], generator=torch.Generator().manual_seed(0))
    v = v / v.norm()
    for _ in range(iters):
        u = W @ v
        v = W.t() @ u
        v = v / v.norm().clamp(min=1e-30)
    return float((W @ v).norm())


def dflash_fc_ledger(path: str, target_layer_ids, full_svd: bool = False):
    W = torch.load(path).float()
    out_features, in_features = W.shape
    T = len(target_layer_ids)
    H = in_features // T
    total = W.pow(2).sum()
    blocks = []
    for j in range(T):
        Wj = W[:, j * H:(j + 1) * H]
        blocks.append({
            "block": j,
            "target_layer": int(target_layer_ids[j]),
            "pi_column_energy_share": float(Wj.pow(2).sum() / total),
            "fro_norm": float(Wj.norm()),
            "spectral_norm": spectral_norm_power(Wj),
            "mean_abs": float(Wj.abs().mean()),
            "max_abs": float(Wj.abs().max()),
        })
    row_norms = W.norm(dim=1)
    out = {
        "checkpoint": os.path.basename(path),
        "shape": [out_features, in_features],
        "target_layer_ids": list(target_layer_ids),
        "H_per_block": H,
        "per_block": blocks,
        "pi_uniform_reference": 1.0 / T,
        "row_norm_mean": float(row_norms.mean()),
        "row_norm_min": float(row_norms.min()),
        "row_norm_max": float(row_norms.max()),
        "row_norm_spread": float(row_norms.max() / row_norms.min()),
        "spectral_norm": spectral_norm_power(W),
        "fro_norm": float(W.norm()),
    }
    out["fro_over_spectral"] = out["fro_norm"] / out["spectral_norm"]
    if full_svd:
        sv = torch.linalg.svdvals(W)
        out.update({
            "sv_max": float(sv[0]),
            "sv_median": float(sv[len(sv) // 2]),
            "sv_min": float(sv[-1]),
            "condition_number": float(sv[0] / sv[-1]),
        })
    return out


def fc_output_sqnr(pi, u, rho, bits):
    """Law 2 (two-regime, analytic) applied to the trained fc.

    pi: column-block energy shares; u_j = sigma_j / sigma_max (the block RMS
    profile, unknown from weights alone); rho = max_t / sigma_j for the loudest
    block; bits -> qmax.

    Per-token step s = max_t/qmax, so block j survives iff
    2 sigma_j >= s  <=>  u_j >= rho/qmax. In units of sigma_max^2:

        signal = sum_{surv} pi_j u_j^2
        noise  = (rho^2 / (12 qmax^2)) * sum_{surv} pi_j  +  sum_{annih} pi_j u_j^2

    (surviving blocks carry a rounding error of density s^2/12; annihilated
    blocks are zeroed, so their residual is their own energy). Validated against
    measured fake-quant SQNR to <0.1 dB outside the annihilation regime by
    ``probe_sqnr_budget.py --laws`` (Law 2).
    """
    qmax = 2 ** (bits - 1) - 1
    surv = [j for j in range(len(u)) if u[j] >= rho / qmax]
    annih = [j for j in range(len(u)) if j not in surv]
    signal = sum(pi[j] * u[j] ** 2 for j in surv)
    noise = (rho ** 2 / (12.0 * qmax ** 2)) * sum(pi[j] for j in surv) \
        + sum(pi[j] * u[j] ** 2 for j in annih)
    destroyed = sum(pi[j] * u[j] ** 2 for j in annih) / \
        max(sum(pi[j] * u[j] ** 2 for j in range(len(u))), 1e-30)
    # PER-PATHWAY metrics. The aggregate above is an ENERGY average and is
    # structurally blind to the annihilation mechanism: as a quiet block gets
    # quieter, its destroyed energy shrinks, so the aggregate SQNR and the
    # energy-weighted destroyed fraction both IMPROVE while that pathway carries
    # less and less information. What does not improve is the block's own
    # functional SNR and its effective bit count.
    per_block = []
    for j in range(len(u)):
        beff = math.log2(max(2.0 * u[j] * qmax / rho, 1e-30))
        sqnr_j = 10.0 * math.log10(max(12.0 * qmax ** 2 * u[j] ** 2 / rho ** 2, 1e-30))
        per_block.append({
            "block": j,
            "u_sigma_over_sigma_max": u[j],
            "pi_column_energy_share": pi[j],
            "effective_bits": beff,
            "functional_snr_db": sqnr_j,
            "surviving": bool(j in surv),
        })
    return {
        "bits": bits, "rho": rho, "u": list(u),
        "blocks_surviving": len(surv), "blocks_annihilated": len(annih),
        "sqnr_out_db": float(10.0 * math.log10(max(signal / max(noise, 1e-30), 1e-30))),
        "destroyed_signal_fraction": float(destroyed),
        "per_block": per_block,
        "min_pi_weighted_effective_bits": float(min(
            b["effective_bits"] for b in per_block if b["pi_column_energy_share"] > 0.1)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".", help="directory holding the fetched .pt tensors")
    ap.add_argument("--dflare-fusion-weights", default=None, help="explicit path")
    ap.add_argument("--dflash-fc-weight", default=None, help="explicit path")
    ap.add_argument("--dflare-layers", default="1,5,9,13,17,21,25,29,33")
    ap.add_argument("--dflash-layers", default="1,17,33")
    ap.add_argument("--bits", default="4,8")
    ap.add_argument("--full-svd", action="store_true",
                    help="also run a full SVD of the fc weight (minutes on CPU)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dflare_layers = [int(v) for v in args.dflare_layers.split(",")]
    dflash_layers = [int(v) for v in args.dflash_layers.split(",")]
    bits_list = [int(v) for v in args.bits.split(",")]

    report = {"mode": "official_fusion_weights", "dir": os.path.abspath(args.dir)}
    fw = args.dflare_fusion_weights or find_one(
        args.dir, "*dflare__layer_fusion_weights.pt", "*layer_fusion_weights.pt")
    fc = args.dflash_fc_weight or find_one(
        args.dir, "*dflash*__fc_weight.pt", "*fc_weight.pt")

    # ---- DFlare: how much mixing exists at all? ----
    if fw:
        led = dflare_fusion_ledger(fw, dflare_layers)
        report["dflare_fusion"] = led
        print("=" * 88)
        print("DFlare %s  T=%d  target layers %s" % (led["shape"][1], led["shape"][1], dflare_layers))
        print("  L  entropy(nats)  w_max/||w||2  avail.dilution(dB)  dominant tap -> target  softmax")
        for r in led["per_draft_layer"]:
            print("  %d   %6.3f        %6.3f        %+7.2f          tap %d -> L%-3d      %s" % (
                r["draft_layer"], r["entropy_nats"], r["w_max_over_norm"],
                r["available_dilution_db"], r["dominant_tap"], r["dominant_target_layer"],
                " ".join("%.3f" % v for v in r["softmax_weights"])))
        print("  mean entropy %.3f nats (uniform %.3f) | mean available dilution %+.2f dB "
              "(max %+.2f dB)" % (led["mean_entropy_nats"], math.log(len(dflare_layers)),
                                  led["mean_available_dilution_db"],
                                  led["max_available_dilution_db"]))
        print("  dominant target layers actually used: %s of %d available taps"
              % (led["dominant_target_layers_used"], led["taps_available"]))
    else:
        print("[warn] no DFlare layer_fusion_weights found in " + args.dir)

    # ---- DFlash: where does the trained fc put its column energy? ----
    if fc:
        led = dflash_fc_ledger(fc, dflash_layers, full_svd=args.full_svd)
        report["dflash_fc"] = led
        print()
        print("=" * 88)
        print("DFlash fc.weight %s  T=%d  (uniform pi = %.4f)"
              % (led["shape"], len(dflash_layers), led["pi_uniform_reference"]))
        for b in led["per_block"]:
            print("  block %d (target L%-3d): pi=%.4f  ||Wj||_F=%8.3f  s1=%8.2f  mean|w|=%.5f"
                  % (b["block"], b["target_layer"], b["pi_column_energy_share"],
                     b["fro_norm"], b["spectral_norm"], b["mean_abs"]))
        print("  row norms: mean %.2f  spread %.2fx | ||W||_F %.1f  ||W||_2 %.1f  F/2 %.2f"
              % (led["row_norm_mean"], led["row_norm_spread"], led["fro_norm"],
                 led["spectral_norm"], led["fro_over_spectral"]))
        if "condition_number" in led:
            print("  full SVD: sv max %.1f median %.2f min %.3f cond %.0f"
                  % (led["sv_max"], led["sv_median"], led["sv_min"], led["condition_number"]))

        # ---- Law 2 bracket over the unknown block-RMS profile ----
        # target layers are in depth order, so assume a monotone depth ramp:
        # u = [q, ..., 1] with a geometric ramp; q is the quietest/loudest ratio.
        pi = [b["pi_column_energy_share"] for b in led["per_block"]]
        T = len(pi)
        sweep = []
        for q in (0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.002, 1e-3, 1e-4):
            u = [q ** ((T - 1 - j) / (T - 1)) for j in range(T)]
            for rho in (4.0, 8.0):
                for bits in bits_list:
                    row = fc_output_sqnr(pi, u, rho, bits)
                    row["quiet_over_loud_rms"] = q
                    sweep.append(row)
        report["dflash_fc"]["law2_bracket"] = sweep
        print()
        print("  Law 2 bracket for the fc OUTPUT once the concat is per-token quantized")
        print("  (block RMS profile assumed: geometric depth ramp, quiet/loud ratio q;")
        print("   rho = max_t/sigma_loudest; real q/rho come from S1b hidden states)")
        print("      q     rho  bits  surv/annih  SQNR_out(dB)  destroyed_signal_frac")
        for q in (0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.002, 1e-3, 1e-4):
            for rho in (4.0, 8.0):
                for bits in bits_list:
                    for row in sweep:
                        if row["quiet_over_loud_rms"] == q and row["rho"] == rho and row["bits"] == bits:
                            print("   %7.4f %5.1f  %2d      %d/%d        %+7.2f        %.4f" % (
                                q, rho, bits, row["blocks_surviving"], row["blocks_annihilated"],
                                row["sqnr_out_db"], row["destroyed_signal_fraction"]))
    else:
        print("[warn] no DFlash fc_weight found in " + args.dir)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1)
        print("\n[report] wrote " + args.out)


if __name__ == "__main__":
    main()
