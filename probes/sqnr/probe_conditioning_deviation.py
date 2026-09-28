# coding=utf-8
"""PTQ sensitivity of the target-CONDITIONING path: DFlash (fc) vs DFlare (convex).

Motivated by the reported PTQ observation (w8 on every Linear, no QAT):
DFlare's acceptance barely moves, DFlash's drops hard. This probe measures the
relative deviation that PTQ injects into the target-conditioning path, which is
the path the draft exists to read, and decomposes it into its two quantized
points.

DFlash conditioning path (per draft layer i):

    concat -> [ fc ] -> hidden_norm -> [ k_proj , v_proj ] -> attention
                 ^^^^^^                    ^^^^^^^^^^^^^^^^
                 point 1 (fc)              point 2 (SHARED with the draft stream)

DFlare conditioning path:

    h_j -> sum_j softmax(w_ij) h_j -> hidden_norm -> [ k_proj_target, v_proj_target ] -> attention
         ^^^^^^^^^^^^^^^^^^^^^^^^                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
         NO nn.Linear here at all                    point 2 (DEDICATED to the target path)

So under "quantize every Linear weight" the two models differ twice, not once:
DFlare's fusion is arithmetic on a parameter vector (softmax weights) and cannot
be touched by weight quantization at all -- its fused feature is bit-identical to
BF16 -- while DFlash's fusion is a genuine `nn.Linear` and is corrupted at the
source, feeding a single shared feature to all 5 layers.

Metrics per variant and per draft layer: relative L2 error and cosine of the
conditioning feature, and of the K/V that attention actually reads.

Usage:
    python probes/sqnr/probe_conditioning_deviation.py --dir <fetched tensor dir> \
        [--bits 8,4] [--fetch] [--out probes/sqnr/reports/conditioning_deviation.json]

--fetch downloads the (small) K/V projection tensors on demand; the forecast
tensors must already be present (see fetch_hf_tensors.py).
"""

import argparse
import json
import math
import os
import sys

import _blas_env  # noqa: F401

import torch
import torch.nn.functional as F

from common import synthetic_target_hidden
import fetch_hf_tensors as fh

WXAY = sys.modules["_wxay_probe_mod"]

DFLASH_REPO = "z-lab/Qwen3-8B-DFlash-b16"
DFLARE_REPO = "AngelSlim/Qwen3-8b-dflare"
# 9 taps the DFlare draft reads; DFlash's 5 taps are a subset of these
TAPS9 = [1, 5, 9, 13, 17, 21, 25, 29, 33]
DFLASH_TAP_POS = [0, 2, 4, 6, 8]        # taps [1, 9, 17, 25, 33]
N_DFLASH_LAYERS = 5
N_DFLARE_LAYERS = 7


def rms_norm(t: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return t / t.pow(2).mean(dim=-1, keepdim=True).add(eps).sqrt()


def deviation(a: torch.Tensor, ref: torch.Tensor) -> dict:
    a = a.float()
    ref = ref.float()
    sig = ref.pow(2).sum()
    noi = (a - ref).pow(2).sum()
    return {
        "rel_l2": float((a - ref).norm() / ref.norm().clamp(min=1e-30)),
        "cosine": float(F.cosine_similarity(a, ref, dim=-1).mean()),
        "sqnr_db": float(10.0 * math.log10(float(sig / noi.clamp(min=1e-30)))),
    }


def ensure_tensors(directory: str, repo: str, names, do_fetch: bool):
    out = {}
    url = header = data_start = None
    for name in names:
        path = os.path.join(directory, repo.split("/")[-1] + "__" + name.replace(".", "_") + ".pt")
        if os.path.exists(path):
            out[name] = torch.load(path).float()
            continue
        if not do_fetch:
            raise FileNotFoundError(path + "  (pass --fetch)")
        if url is None:
            url, header, data_start = fh.read_header(repo)
        t, _ = fh.fetch_tensor(url, header, name, data_start)
        torch.save(t, path)
        print("[fetch] " + name)
        out[name] = t.float()
    return out


def mean_rel_l2(entry: dict, n_layers: int) -> dict:
    def m(key):
        return sum(entry["per_layer_kv"]["layer_%d" % i][key]["rel_l2"]
                   for i in range(n_layers)) / n_layers
    return {"k": m("k_end_to_end"), "v": m("v_end_to_end")}


def build_inputs(B: int, S: int, H: int, seed: int = 0):
    """One shared synthetic target hidden-state stack; both drafts read from it."""
    th = synthetic_target_hidden(B, S, len(TAPS9), H, seed=seed, device="cpu",
                                dtype=torch.float32)
    return th.view(B * S, len(TAPS9), H)


def dflash_conditioning(xr, fc_w, kv, bits_list):
    """End-to-end deviation of the conditioning path against the BF16 reference.

    IMPORTANT: every variant's K/V is compared against K/V computed from the
    **BF16 feature**, not from that variant's own feature. Using the variant's
    own feature as the reference silently drops the upstream `fc` error and
    understates the total by ~10x (8.5-11% vs 0.86% at w8a8).
    """
    N = xr.shape[0]
    H = xr.shape[2]
    concat = xr[:, DFLASH_TAP_POS, :].reshape(N, len(DFLASH_TAP_POS) * H)
    ref_feat = rms_norm(concat @ fc_w.t())
    kv_ref = {}
    for i in range(N_DFLASH_LAYERS):
        k_w = kv["layers.%d.self_attn.k_proj.weight" % i]
        v_w = kv["layers.%d.self_attn.v_proj.weight" % i]
        kv_ref[i] = (ref_feat @ k_w.t(), ref_feat @ v_w.t())
    out = {}
    for bits in bits_list:
        fc_q, _ = WXAY.quantize_weight(fc_w, bits)
        concat_q, _ = WXAY.quantize_activation(concat, bits)
        feat_wonly = rms_norm(concat @ fc_q.t())          # fc weights quantized, concat exact
        variants = [
            # (tag, feature, use_a8) -- every K/V is compared against the BF16 ref
            ("w%d_weight_only" % bits, feat_wonly, False),
            ("w%da%d_fc_in_exact" % (bits, bits), feat_wonly, True),
            ("w%da%d_fc_in_quant" % (bits, bits), rms_norm(concat_q @ fc_q.t()), True),
        ]
        for tag, feat, use_a8 in variants:
            feat_q, _ = WXAY.quantize_activation(feat, bits)
            qfeat = feat_q if use_a8 else feat
            entry = {"feature": deviation(feat, ref_feat), "per_layer_kv": {}}
            for i in range(N_DFLASH_LAYERS):
                k_w = kv["layers.%d.self_attn.k_proj.weight" % i]
                v_w = kv["layers.%d.self_attn.v_proj.weight" % i]
                k_q, _ = WXAY.quantize_weight(k_w, bits)
                v_q, _ = WXAY.quantize_weight(v_w, bits)
                k_ref, v_ref = kv_ref[i]
                entry["per_layer_kv"]["layer_%d" % i] = {
                    "k_end_to_end": deviation(qfeat @ k_q.t(), k_ref),
                    "v_end_to_end": deviation(qfeat @ v_q.t(), v_ref),
                }
            out[tag] = entry
    return {"reference_feature": {"spikiness": float(
        (ref_feat.abs().amax(-1) / ref_feat.pow(2).mean(-1).sqrt()).mean())}, "variants": out}


def dflare_conditioning(xr, fusion_raw, kv, bits_list):
    """The flare fusion has no Linear: weight quantization cannot touch it."""
    N, T, H = xr.shape
    w = torch.softmax(fusion_raw, dim=1)
    feats = [rms_norm((xr * w[i].view(1, -1, 1)).sum(dim=1)) for i in range(N_DFLARE_LAYERS)]
    kv_ref = {}
    for i in range(N_DFLARE_LAYERS):
        k_w = kv["layers.%d.self_attn.k_proj_target.weight" % i]
        v_w = kv["layers.%d.self_attn.v_proj_target.weight" % i]
        kv_ref[i] = (feats[i] @ k_w.t(), feats[i] @ v_w.t())
    out = {}
    for bits in bits_list:
        for tag, use_a8 in (("w%d_weight_only" % bits, False),
                            ("w%da%d" % (bits, bits), True)):
            entry = {"feature": {"rel_l2": 0.0, "cosine": 1.0, "sqnr_db": None,
                                 "note": "no nn.Linear on this path -> bit-identical"},
                     "per_layer_kv": {}}
            for i in range(N_DFLARE_LAYERS):
                k_w = kv["layers.%d.self_attn.k_proj_target.weight" % i]
                v_w = kv["layers.%d.self_attn.v_proj_target.weight" % i]
                k_q, _ = WXAY.quantize_weight(k_w, bits)
                v_q, _ = WXAY.quantize_weight(v_w, bits)
                feat_q, _ = WXAY.quantize_activation(feats[i], bits)
                qfeat = feat_q if use_a8 else feats[i]
                k_ref, v_ref = kv_ref[i]
                entry["per_layer_kv"]["layer_%d" % i] = {
                    "k_end_to_end": deviation(qfeat @ k_q.t(), k_ref),
                    "v_end_to_end": deviation(qfeat @ v_q.t(), v_ref),
                }
            out[tag] = entry
    return {"note": "dedicated k_proj_target/v_proj_target, not shared with the draft stream",
            "variants": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="directory for fetched tensors")
    ap.add_argument("--bits", default="8,4")
    ap.add_argument("--B", type=int, default=4)
    ap.add_argument("--S", type=int, default=64)
    ap.add_argument("--H", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fetch", action="store_true", help="download missing K/V tensors")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    bits_list = [int(b) for b in args.bits.split(",")]
    os.makedirs(args.dir, exist_ok=True)

    xr = build_inputs(args.B, args.S, args.H, args.seed)
    report = {"mode": "ptq_conditioning_deviation", "taps9": TAPS9,
              "dflash_taps": [TAPS9[p] for p in DFLASH_TAP_POS],
              "tokens": int(xr.shape[0]), "bits": bits_list}

    # ---- DFlash: fusion is an nn.Linear, so it IS quantized ----
    fc_w = ensure_tensors(args.dir, DFLASH_REPO, ["fc.weight"], args.fetch)["fc.weight"]
    dflash_kv_names = []
    for i in range(N_DFLASH_LAYERS):
        dflash_kv_names += ["layers.%d.self_attn.k_proj.weight" % i,
                            "layers.%d.self_attn.v_proj.weight" % i]
    dflash_kv = ensure_tensors(args.dir, DFLASH_REPO, dflash_kv_names, args.fetch)
    report["dflash"] = dflash_conditioning(xr, fc_w, dflash_kv, bits_list)

    print("=" * 92)
    print("DFlash (z-lab official):  fc IS an nn.Linear -> PTQ corrupts the conditioning at the source")
    print("  variant        feature relL2  cos     | mean per-layer end-to-end relL2 vs BF16: K, V")
    for tag, e in report["dflash"]["variants"].items():
        m = mean_rel_l2(e, N_DFLASH_LAYERS)
        print("  %-14s %.4f      %.5f | K %.4f   V %.4f" % (
            tag, e["feature"]["rel_l2"], e["feature"]["cosine"],
            m["k"], m["v"]))

    # ---- DFlare: fusion has no Linear at all ----
    fw = ensure_tensors(args.dir, DFLARE_REPO, ["layer_fusion_weights"], args.fetch)["layer_fusion_weights"]
    dflare_kv_names = []
    for i in range(N_DFLARE_LAYERS):
        dflare_kv_names += ["layers.%d.self_attn.k_proj_target.weight" % i,
                            "layers.%d.self_attn.v_proj_target.weight" % i]
    dflare_kv = ensure_tensors(args.dir, DFLARE_REPO, dflare_kv_names, args.fetch)
    report["dflare"] = dflare_conditioning(xr, fw, dflare_kv, bits_list)

    print("=" * 92)
    print("DFlare (AngelSlim official): fusion has NO nn.Linear -> PTQ cannot touch the feature")
    print("  variant        feature relL2  cos     | mean per-layer end-to-end relL2 vs BF16: K, V")
    for tag, e in report["dflare"]["variants"].items():
        m = mean_rel_l2(e, N_DFLARE_LAYERS)
        print("  %-14s %.4f      %.5f | K %.4f   V %.4f" % (
            tag, e["feature"]["rel_l2"], e["feature"]["cosine"],
            m["k"], m["v"]))

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1)
        print("\n[report] wrote " + args.out)


if __name__ == "__main__":
    main()
