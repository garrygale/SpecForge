# probes/sqnr -- SQNR analysis of the DFlash-vs-DFlare target-hidden fusion

Why the convex (`fusion_mode="flare"`) target-hidden fusion is quantizable
(w4a8 + a w4a4 subset) while the matmul (`fusion_mode="fc"`) fusion loses >10%
acceptance length even at w8a8 QAT.

Start with **`SQNR_ANALYSIS_V0.md`** (the lens, the validated laws, and the S1a
ledger of the official trained weights), then **`SQNR_PLAN.md`** (stages S1-S4,
decision gates, open decisions).

## The one-line mechanism

The quantizer (`specforge/layers/wxay.py:52`) is per-token symmetric, so every
coordinate of a token shares the step `s_t = max_t/qmax`. `replace_linear_with_quantized()`
swaps **every** `nn.Linear` except `qat_exclude`. Therefore the fusion operator
type decides *where the quantizer sits relative to the multi-layer mixing*:

| | fc (`dflash.py:1292`) | flare (`dflash.py:1397`) |
|---|---|---|
| quantized tensor | the **pre-mixing concat**, 9 blocks on one scale | the **post-mixing** fused vector, 1 block |
| pays | spikiness tax + **block-disparity tax** | spikiness tax only |

## Files

| file | what it does |
|---|---|
| `probe_sqnr_budget.py --laws` | validates Laws 1-5 on synthetic ground truth (no config needed) |
| `probe_sqnr_budget.py --config ... [--random]` | the fc-vs-flare SQNR budget: block ledger, both paths in the same units, audit ratios |
| `probe_residual_scale_profile.py --target-model <Qwen3-8B>` | **target residual-stream RMS profile + per-tap effective bits** (see below) |
| `probe_conditioning_deviation.py --dir <dir> --fetch` | PTQ deviation of the target-conditioning path, DFlash vs DFlare, split by quantized point |
| `analyze_fusion_weights.py --dir <dir>` | weight-side ledger of the official trained drafts: DFlare mixing/dilution, DFlash `pi`, Law 2 bracket |
| `fetch_hf_tensors.py` | range-fetch individual safetensors tensors from HuggingFace (drafts are 2.8-5.0 GB; the ledger needs ~100 MB) |
| `common.py`, `_blas_env.py` | standalone loader shim (loads `dflash.py` + `wxay.py` without the full package import chain) |

## The one command to run on the server

```bash
python probes/sqnr/probe_residual_scale_profile.py --target-model /path/to/Qwen3-8B
```

Everything else defaults to Qwen3-8B: `target_layer_ids` come from the official
DFlash draft config (`z-lab/Qwen3-8B-DFlash-b16`, or a local copy via
`--draft-ckpt`), and only that draft's `fc.weight` (167 MB) is fetched -- never
the 5 GB checkpoint. It prints, and writes to
`probes/sqnr/reports/residual_scale_profile.json`:

1. per-tap RMS medians and the **per-token across-tap disparity** (the block
   disparity a single per-token scale has to cover);
2. the full per-layer profile (median per-token RMS, so the massive-activation
   onset is visible);
3. the **effective-bits table**: `b_eff = bits - log2(max_t / r)`, per tap, and
   how much of the trained `fc`'s column energy sits on taps with `b_eff <= 1`.

Useful flags: `--draft-ckpt <path>` (local draft; also picks up its taps and
skips the network), `--no-fc` (skip the fc lookup), `--texts-file <file>`,
`--rho <float>` (max_t/rms_loudest, default 15), `--proxy-map-layers 36` **only**
when profiling a smaller model as a proxy.

Offline runs work as long as `--draft-ckpt` points at a local dir containing
`config.json` + `model.safetensors`.

## Running the rest

```bash
PY=<python with torch + transformers>

# 1. laws (seconds, CPU)
$PY probes/sqnr/probe_sqnr_budget.py --laws

# 2. budget, synthetic hidden states + random-init weights (structural)
$PY probes/sqnr/probe_sqnr_budget.py \
    --config configs/qwen3-8b-domino-dflare-verifiedBase.json --random --B 8 --S 128

# 3. budget on real hidden states (needs the target model)
$PY probes/sqnr/probe_sqnr_budget.py \
    --config configs/qwen3-8b-domino-dflare-verifiedBase.json \
    --target-model-path <Qwen3-8B> --draft-ckpt <ckpt> --B 32 --S 256

# 4. official trained fusion weights (needs network, no target model)
$PY probes/sqnr/fetch_hf_tensors.py --repo AngelSlim/Qwen3-8b-dflare \
    --tensor layer_fusion_weights --tensor hidden_norm.weight --out <dir>
$PY probes/sqnr/fetch_hf_tensors.py --repo z-lab/Qwen3-8B-DFlash-b16 \
    --tensor fc.weight --tensor hidden_norm.weight --out <dir>
$PY probes/sqnr/analyze_fusion_weights.py --dir <dir>

# 5. PTQ deviation of the conditioning path, both drafts (needs network once)
$PY probes/sqnr/probe_conditioning_deviation.py --dir <dir> --fetch
```

Reports land in `probes/sqnr/reports/` (gitignored).

## Environment notes

* `web_fetch`/curl cannot reach huggingface.co here (fake-IP DNS, schannel
  failure); python's TLS stack can, so `fetch_hf_tensors.py` uses `urllib`.
* All SQNR arithmetic is pinned to CPU: it is elementwise plus one matmul, and
  single-device keeps dtype handling simple. `--device` still drives the heavy
  target-model forward.
* `hidden_norm.weight` should read mean ~0.8-1.0. If it reads ~1e37, the
  safetensors `data_offsets` are being interpreted relative to the wrong base
  (they are relative to `8 + header_len`, not 8).
