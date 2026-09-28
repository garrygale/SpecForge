# coding=utf-8
"""Qwen3-8B target residual-stream scale profile -> per-tap effective bits.

One command, one path. Everything else defaults to Qwen3-8B + the official
DFlash draft config; nothing needs editing.

    python probes/sqnr/probe_residual_scale_profile.py --target-model /path/to/Qwen3-8B

That is the whole interface. It will:

  1. load the target model (HF id, or a local dir) and run a forward pass with
     ``output_hidden_states=True``;
  2. read ``target_layer_ids`` from the draft config (``--draft-ckpt``, default
     the official ``z-lab/Qwen3-8B-DFlash-b16``) and take ``hidden_states[L+1]``
     for each tap -- exactly what ``extract_context_feature`` concatenates;
  3. report the per-tap RMS and the per-TOKEN across-tap disparity. This is the
     "block disparity" that a single per-token activation scale has to cover;
  4. if the draft has an ``fc.weight`` (i.e. matmul fusion), fetch ONLY that
     tensor and print the effective-bits table: how many of the 8 nominal bits
     each tap actually keeps once the concat shares one scale.

Why this matters: the fc input is ``cat([hidden_states[L] for L in taps])`` with
NO per-tap normalization, and the draft quantizer is per-token symmetric, so one
step ``s = max_t/qmax`` covers the whole concat. Effective bits for a tap of RMS
``r`` are ``b_eff = bits - log2(max_t / r)``; a tap with ``b_eff <= 1`` carries
at most one bit, and ``b_eff < 0`` means it is inside the dead zone.

STATISTICS WARNING: hidden-state norms are extremely heavy tailed (massive
activations), so a MEAN over tokens is dominated by a few outliers -- on
Qwen3-0.6B the mean per-token RMS at the onset layer reads 16.0 while p90 is
0.489. Every number below is a median unless labelled otherwise.

Outputs ``probes/sqnr/reports/residual_scale_profile.json`` alongside the tables.
"""

import argparse
import json
import os

import _blas_env  # noqa: F401  (must precede torch: OpenBLAS thread workaround)

import torch

try:  # Ascend NPU backend (optional; only needed for --device npu)
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None


DEFAULT_TARGET = "Qwen/Qwen3-8B"
DEFAULT_DRAFT = "z-lab/Qwen3-8B-DFlash-b16"
DEFAULT_TAPS = [1, 9, 17, 25, 33]
DEFAULT_TEXT = "The theory of speculative decoding dates back to"

PROMPTS = [
    "The theory of speculative decoding dates back to",
    "def quicksort(x):\n    if len(x) <= 1:\n        return x\n",
    "In a large language model, the residual stream",
    "Solve for x: 3x + 7 = 22. Step by step:",
    "The capital of France is Paris, and the capital of Japan is",
    "Quantization of neural networks requires careful handling of",
]

_DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32, "fp16": torch.float16}
_HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# model / inputs
# ---------------------------------------------------------------------------

def pick_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch_npu is not None and torch.npu.is_available():
        return "npu"
    return "cpu"


def load_target(model_id: str, dtype: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("[target] loading %s (dtype=%s, device=%s)" % (model_id, dtype, device))
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=_DTYPES[dtype], device_map=None).eval().to(device)
    return model, tok


def read_texts(path):
    if not path:
        return list(PROMPTS)
    with open(path, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]
    return texts or list(PROMPTS)


def gather_hidden_states(model, tok, texts, max_len: int, device: str):
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
              max_length=max_len)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                    output_hidden_states=True, use_cache=False)
    return out.hidden_states, enc["attention_mask"].bool()


# ---------------------------------------------------------------------------
# draft checkpoint: taps + fc.weight (only that tensor is ever read)
# ---------------------------------------------------------------------------

def _is_hf_repo(spec: str) -> bool:
    return (not os.path.exists(spec)) and ("/" in spec) and (not os.path.isabs(spec))


def draft_config(draft: str) -> dict:
    """Read dflash_config from a local dir/file/repo without downloading weights."""
    import json as _json
    if os.path.isdir(draft) and os.path.exists(os.path.join(draft, "config.json")):
        with open(os.path.join(draft, "config.json"), encoding="utf-8") as f:
            return _json.load(f)
    if _is_hf_repo(draft):
        try:
            import urllib.request
            url = "https://huggingface.co/%s/resolve/main/config.json" % draft
            with urllib.request.urlopen(url, timeout=60) as r:
                return _json.loads(r.read())
        except Exception as e:
            print("[draft] could not read config from %s (%s)" % (draft, str(e)[:80]))
    return {}


def resolve_taps(draft: str, explicit: str):
    if explicit:
        return [int(v) for v in explicit.split(",")]
    cfg = draft_config(draft)
    taps = (cfg.get("dflash_config") or {}).get("target_layer_ids")
    if taps and not isinstance(taps[0], list):
        print("[draft] target_layer_ids from %s: %s" % (draft, taps))
        return [int(t) for t in taps]
    print("[draft] no target_layer_ids found; using DFlash default %s" % DEFAULT_TAPS)
    return list(DEFAULT_TAPS)


def load_fc_weight(draft: str, cache_dir: str):
    """Return fc.weight (float32) or None. Reads ONLY that tensor.

    - local safetensors file / dir -> safe_open (lazy, reads just this tensor)
    - HF repo id                  -> range-fetch via fetch_hf_tensors (167 MB for
      the 8B DFlash draft, instead of the 5 GB checkpoint)
    Returns None when the draft has no fc.weight (e.g. a flare/router draft).
    """
    import safetensors.torch as st
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, os.path.basename(draft.rstrip("/\\")) + "__fc_weight.pt")
    if os.path.exists(cache):
        print("[draft] fc.weight from cache: " + cache)
        return torch.load(cache).float()

    if os.path.isfile(draft) and draft.endswith(".safetensors"):
        with st.safe_open(draft, framework="pt") as f:
            if "fc.weight" not in f.keys():
                return None
            w = f.get_tensor("fc.weight")
    elif os.path.isdir(draft):
        path = None
        for name in ("model.safetensors", "pytorch_model.bin"):
            if os.path.exists(os.path.join(draft, name)):
                path = os.path.join(draft, name)
                break
        if path is None:
            print("[draft] no weights file in %s" % draft)
            return None
        if path.endswith(".safetensors"):
            with st.safe_open(path, framework="pt") as f:
                if "fc.weight" not in f.keys():
                    return None
                w = f.get_tensor("fc.weight")
        else:
            sd = torch.load(path, map_location="cpu", weights_only=True)
            if "fc.weight" not in sd:
                return None
            w = sd["fc.weight"]
    elif _is_hf_repo(draft):
        import fetch_hf_tensors as fh
        url, header, data_start = fh.read_header(draft)
        if "fc.weight" not in header:
            return None
        print("[draft] range-fetching fc.weight from %s ..." % draft)
        w, _ = fh.fetch_tensor(url, header, "fc.weight", data_start)
    else:
        print("[draft] cannot interpret %s" % draft)
        return None

    torch.save(w.detach().cpu().to(torch.bfloat16), cache)
    print("[draft] fc.weight %s -> cached %s" % (list(w.shape), cache))
    return w.float()


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def per_layer_curve(hs, mask):
    rows = []
    mask_c = mask.cpu()
    for k, h in enumerate(hs):
        # per-layer move to CPU: quantile/median are not guaranteed on NPU, and
        # one layer of statistics is small.
        h = h.float().cpu()[mask_c]
        r = h.pow(2).mean(dim=-1).sqrt()
        mx = h.abs().amax(dim=-1)
        rows.append({
            "hidden_state_index": k,
            "out_of_layer": k - 1,
            "rms_median": float(r.median()),
            "rms_p10": float(r.quantile(0.10)),
            "rms_p90": float(r.quantile(0.90)),
            "rms_p99": float(r.quantile(0.99)),
            "rms_mean": float(r.mean()),
            "rms_max": float(r.max()),
            "spikiness_median": float((mx / r.clamp(min=1e-12)).median()),
        })
    return rows


def tap_stats(hs, mask, taps, map_from_layers=None):
    """Per-tap per-token RMS and the within-token across-tap disparity.

    ``map_from_layers`` is only for using a SMALLER model as a proxy: taps are
    then matched by relative depth. On the real Qwen3-8B leave it None and the
    taps are used as-is.
    """
    n_layers = len(hs) - 1
    if map_from_layers:
        idx = [min(max(int(round(t / map_from_layers * n_layers)), 0), n_layers)
               for t in taps]
    else:
        idx = [min(max(int(t), 0), n_layers) for t in taps]
    mask_c = mask.cpu()
    R = torch.stack([
        hs[L + 1].float().cpu().pow(2).mean(dim=-1).sqrt()[mask_c] for L in idx], dim=1)
    disp = R.max(dim=1).values / R.min(dim=1).values.clamp(min=1e-12)
    med = R.median(dim=0).values
    return {
        "layers_read_out_of_layer": idx,
        "relative_depth_mapped": bool(map_from_layers),
        "per_tap": [{
            "target_layer": int(t),
            "out_of_layer": int(L),
            "rms_median": float(med[j]),
            "rms_p10": float(R[:, j].quantile(0.10)),
            "rms_p90": float(R[:, j].quantile(0.90)),
            "rms_p99": float(R[:, j].quantile(0.99)),
            "rms_mean": float(R[:, j].mean()),
            "rms_max": float(R[:, j].max()),
        } for j, (t, L) in enumerate(zip(taps, idx))],
        "disparity_median": float(disp.median()),
        "disparity_p10": float(disp.quantile(0.10)),
        "disparity_p90": float(disp.quantile(0.90)),
        "disparity_p99": float(disp.quantile(0.99)),
        "ratio_of_tap_medians": float(med.max() / med.min()),
        "ratio_of_tap_means_outlier_dominated": float(
            R.mean(dim=0).max() / R.mean(dim=0).min()),
        "token_fraction_any_tap_gt_10": float((R.max(dim=1).values > 10).float().mean()),
        "token_fraction_any_tap_gt_100": float((R.max(dim=1).values > 100).float().mean()),
    }


def effective_bits_table(tap_rms, pi, bits: int, rho: float):
    """b_eff = bits - log2(max_t / r); max_t ~ rho * rms_loudest."""
    qmax = 2 ** (bits - 1) - 1
    max_t = rho * max(tap_rms)
    s = max_t / qmax
    rows = []
    for r, p in zip(tap_rms, pi):
        rows.append({
            "rms": float(r),
            "pi_column_energy_share": float(p),
            "effective_bits": float(torch.log2(torch.tensor(max(2.0 * r / s, 1e-30)))),
            "tap_rel_error_from_step": float((s ** 2 / 12.0) ** 0.5 / max(r, 1e-30)),
        })
    return {"bits": bits, "rho": rho, "step": float(s), "taps": rows,
            "pi_share_on_taps_le_1_bit": float(
                sum(r["pi_column_energy_share"] for r in rows if r["effective_bits"] <= 1.0))}


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Qwen3-8B residual-stream scale profile and per-tap effective bits. "
                    "Only --target-model normally needs to be given.")
    ap.add_argument("--target-model", default=DEFAULT_TARGET,
                    help="Qwen3-8B path or HF id (default %s)" % DEFAULT_TARGET)
    ap.add_argument("--draft-ckpt", default=DEFAULT_DRAFT,
                    help="DFlash draft: used only for target_layer_ids and fc.weight "
                         "(default %s)" % DEFAULT_DRAFT)
    ap.add_argument("--taps", default=None,
                    help="override target_layer_ids, e.g. 1,9,17,25,33")
    ap.add_argument("--texts-file", default=None,
                    help="one text per line; default built-in prompts")
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32", "fp16"])
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | npu")
    ap.add_argument("--rho", type=float, default=15.0,
                    help="assumed max_t/rms_loudest for the effective-bits table (default 15)")
    ap.add_argument("--bits", default="8,4")
    ap.add_argument("--proxy-map-layers", type=int, default=None,
                    help="ONLY when profiling a smaller proxy model: the tap reference "
                         "layer count (e.g. 36 when running Qwen3-0.6B). Omit on Qwen3-8B.")
    ap.add_argument("--no-fc", action="store_true", help="skip the fc.weight lookup")
    ap.add_argument("--cache-dir", default=os.path.join(_HERE, "reports", "cache"))
    ap.add_argument("--out", default=os.path.join(_HERE, "reports",
                                                 "residual_scale_profile.json"))
    args = ap.parse_args()

    device = pick_device(args.device)
    bits_list = [int(b) for b in args.bits.split(",")]
    texts = read_texts(args.texts_file)
    taps = resolve_taps(args.draft_ckpt, args.taps)

    model, tok = load_target(args.target_model, args.dtype, device)
    n_layers = int(model.config.num_hidden_layers)
    hidden = int(model.config.hidden_size)
    hs, mask = gather_hidden_states(model, tok, texts, args.max_len, device)
    print("[target] %d layers, hidden %d, %d prompts, %d tokens kept"
          % (n_layers, hidden, len(texts), int(mask.sum())))

    report = {
        "mode": "residual_scale_profile",
        "target_model": args.target_model,
        "target_num_layers": n_layers,
        "target_hidden_size": hidden,
        "draft": args.draft_ckpt,
        "taps": taps,
        "dtype": args.dtype, "device": device, "max_len": args.max_len,
        "rho": args.rho,
        "per_hidden_state": per_layer_curve(hs, mask),
        "taps_stats": tap_stats(hs, mask, taps, args.proxy_map_layers),
    }
    ts = report["taps_stats"]

    print()
    print("=" * 96)
    print("PER-TAP RMS (per token; medians -- means are outlier-dominated)")
    print("  target L   read@layer   rms_median      p10       p90        p99        max")
    for p in ts["per_tap"]:
        print("  L%-8d %8d    %10.4f %9.4f %9.4f %10.2f %10.1f" % (
            p["target_layer"], p["out_of_layer"], p["rms_median"],
            p["rms_p10"], p["rms_p90"], p["rms_p99"], p["rms_max"]))
    print()
    print("  per-TOKEN across-tap disparity (max_j rms / min_j rms within one token):")
    print("    median %.2fx    p10 %.2fx    p90 %.2fx    p99 %.2fx" % (
        ts["disparity_median"], ts["disparity_p10"], ts["disparity_p90"], ts["disparity_p99"]))
    print("  ratio of tap medians (max/min) = %.2fx   [ratio of means = %.2fx, do not quote]"
          % (ts["ratio_of_tap_medians"], ts["ratio_of_tap_means_outlier_dominated"]))
    print("  heavy tail: %.2f%% of tokens have some tap rms > 10; %.3f%% > 100" % (
        100 * ts["token_fraction_any_tap_gt_10"], 100 * ts["token_fraction_any_tap_gt_100"]))

    print()
    print("FULL STACK (median per-token RMS; the massive-activation onset is visible)")
    print("  hs_idx  out_of_layer   rms_median   rms_p90    rms_p99     rms_mean")
    for r in report["per_hidden_state"]:
        print("  %5d   %10d   %10.4f %9.4f %10.2f %11.2f" % (
            r["hidden_state_index"], r["out_of_layer"], r["rms_median"],
            r["rms_p90"], r["rms_p99"], r["rms_mean"]))

    fc = None if args.no_fc else load_fc_weight(args.draft_ckpt, args.cache_dir)
    report["fc_weight_used"] = bool(fc is not None)
    if fc is not None:
        n_taps = len(taps)
        if fc.shape[1] % n_taps:
            print("[draft] fc.weight %s does not split into %d taps; skipping table"
                  % (list(fc.shape), n_taps))
        else:
            H = fc.shape[1] // n_taps
            pi = [float(fc[:, j * H:(j + 1) * H].pow(2).sum() / fc.pow(2).sum())
                  for j in range(n_taps)]
            report["pi_column_energy_share"] = pi
            report["effective_bits"] = {}
            print()
            print("EFFECTIVE BITS PER TAP with ONE shared per-token scale over the concat")
            print("  (b_eff = bits - log2(max_t/r), max_t = %.1f x rms_loudest; pi from fc.weight)"
                  % args.rho)
            for bits in bits_list:
                tab = effective_bits_table([p["rms_median"] for p in ts["per_tap"]],
                                          pi, bits, args.rho)
                report["effective_bits"]["%dbit" % bits] = tab
                print("  --- %d-bit, step %.4f ---" % (bits, tab["step"]))
                print("    target L    rms        pi       b_eff    tap rel-err")
                for t, r in zip(taps, tab["taps"]):
                    print("    L%-8d %8.4f   %.4f   %7.2f      %.2f" % (
                        t, r["rms"], r["pi_column_energy_share"],
                        r["effective_bits"], r["tap_rel_error_from_step"]))
                print("    pi share on taps with <= 1 effective bit: %.3f"
                      % tab["pi_share_on_taps_le_1_bit"])
            print()
            print("  >> a tap with b_eff <= 1 carries at most one bit of its content;")
            print("     b_eff < 0 means it is inside the dead zone (rounded to zero).")
    else:
        print()
        print("NOTE: no fc.weight in the draft -> its fusion has no nn.Linear on the")
        print("      conditioning path, so the concat-disparity mechanism cannot apply")

    print()
    print("=" * 96)
    print("SUMMARY (paste this back)")
    print("  target            : %s (%d layers, hidden %d)" % (args.target_model, n_layers, hidden))
    print("  taps              : %s" % taps)
    print("  tap rms medians   : %s" % ["%.3f" % p["rms_median"] for p in ts["per_tap"]])
    print("  disparity median  : %.2fx  (p10 %.2fx / p90 %.2fx)" % (
        ts["disparity_median"], ts["disparity_p10"], ts["disparity_p90"]))
    if fc is not None and "8bit" in report.get("effective_bits", {}):
        print("  b_eff @8-bit      : %s" % ["%.2f" % t["effective_bits"]
                                            for t in report["effective_bits"]["8bit"]["taps"]])
        print("  pi on <=1-bit taps: %.3f" %
              report["effective_bits"]["8bit"]["pi_share_on_taps_le_1_bit"])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print("\n[report] wrote " + args.out)


if __name__ == "__main__":
    main()
