# coding=utf-8
"""Shared utilities for DFlash/Domino quantization probes.

Runs standalone (without a full specforge install): loads
specforge/modeling/draft/dflash.py and specforge/layers/wxay.py directly from
the repo via a namespace-package shim, so these probes work in any env with
torch + transformers (CPU / CUDA / Ascend NPU).

Main pieces:
  - ensure_standalone(): module shim + loader (call once at import).
  - load_draft_model_from_json(): build DFlashDraftModel from a raw config
    JSON (as in configs/qwen3-8b-domino-dflare-verifiedBase.json).
  - load_checkpoint(): load a draft checkpoint state dict with flexible key
    prefixes.
  - build_dflash_mask(): replicate create_dflash_sdpa_mask (block mode,
    per-layer sliding window) without importing specforge.core.
  - make_inputs(): build a realistic forward input batch (real target hidden
    states if a target model is given, otherwise synthetic spiky ones).
  - activation stats: per-token spikiness, kurtosis, outlier fractions, and
    the per-token symmetric fake-quantizer SQNR that mirrors wxay.py.
  - CaptureHooks: forward_pre_hook collector for matmul input tensors.
"""

import argparse
import importlib.util
import json
import os
import sys
import types
from typing import Dict, Iterable, List, Optional, Tuple

import _blas_env  # noqa: F401  (must precede torch: OpenBLAS thread workaround)

import torch
import torch.nn as nn

# Align torch's own intra-op threads with the BLAS setting above.
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))

try:  # Ascend NPU backend (optional; only needed for --device npu)
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Standalone loader
# ---------------------------------------------------------------------------

def ensure_standalone():
    """Load dflash.py + wxay.py without triggering the full specforge import
    chain (which needs yunchang etc.). Returns the dflash module."""
    if "_dflash_probe_mod" in sys.modules:
        return sys.modules["_dflash_probe_mod"]

    if "specforge" not in sys.modules:
        pkg = types.ModuleType("specforge")
        pkg.__path__ = [os.path.join(REPO_ROOT, "specforge")]
        sys.modules["specforge"] = pkg
    sys.path.insert(0, REPO_ROOT)

    import specforge.utils  # noqa: F401  (get_device_type etc.)

    # dflash.py uses bare sibling imports (``from dflash_kernels import ...``),
    # so its own directory must be importable as well.
    sys.path.insert(0, os.path.join(REPO_ROOT, "specforge", "modeling", "draft"))

    def _load(name, relpath):
        spec = importlib.util.spec_from_file_location(
            name, os.path.join(REPO_ROOT, relpath)
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    dflash = _load("_dflash_probe_mod", os.path.join("specforge", "modeling", "draft", "dflash.py"))
    wxay = _load("_wxay_probe_mod", os.path.join("specforge", "layers", "wxay.py"))
    sys.modules["_wxay_probe_mod"] = wxay
    return dflash


dflash_mod = ensure_standalone()
wxay_mod = sys.modules.get("_wxay_probe_mod")
Qwen3Config = None
try:
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Config
except Exception:  # pragma: no cover
    from transformers import Qwen3Config


# ---------------------------------------------------------------------------
# Config / model construction
# ---------------------------------------------------------------------------

_STRICT_KEYS = ("sliding_window", "use_sliding_window", "layer_types", "dflash_config")


def load_qwen3_config(config_path: str, fusion_mode: Optional[str] = None, attn_impl: str = "sdpa"):
    """Load a raw config JSON (dict-style, with dflash_config) into a
    Qwen3Config with dflash_config attached.

    transformers>=5 dataclass-validates some keys (e.g. list-valued
    sliding_window); on failure we retry with those keys attached via
    object.__setattr__, which keeps dflash code working on both 4.x and 5.x.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if fusion_mode is not None:
        cfg.setdefault("dflash_config", {})["fusion_mode"] = fusion_mode
    if fusion_mode == "fc":
        # DFlash-default pairing: fc output (hidden_size) feeds the shared
        # KV projections; heterogeneous_kv (target_hidden_size input) only
        # pairs with flare (fused feature keeps target_hidden_size).
        if cfg["dflash_config"].get("heterogeneous_kv", False):
            print("[cfg] fc mode: forcing heterogeneous_kv=False (DFlash default pairing)")
        cfg["dflash_config"]["heterogeneous_kv"] = False
    try:
        model_cfg = Qwen3Config(**cfg)
    except Exception:
        # transformers>=5 raises StrictDataclassFieldValidationError (a
        # subclass chain we cannot import portably) for list-valued
        # sliding_window etc.; pop and attach manually.
        strict = {k: cfg.pop(k) for k in list(cfg) if k in _STRICT_KEYS}
        if not strict:
            raise
        model_cfg = Qwen3Config(**cfg)
        for k, v in strict.items():
            object.__setattr__(model_cfg, k, v)
    object.__setattr__(model_cfg, "_attn_implementation", attn_impl)
    return model_cfg


def build_draft_model(
    config_path: str,
    fusion_mode: Optional[str] = None,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cpu",
    attn_impl: str = "sdpa",
):
    """Instantiate DFlashDraftModel from a config JSON (random weights)."""
    model_cfg = load_qwen3_config(config_path, fusion_mode, attn_impl)
    model = dflash_mod.DFlashDraftModel(model_cfg)
    model = model.to(dtype=dtype).to(device)
    model.eval()
    return model, model_cfg


def load_checkpoint_state(path: str, device: str = "cpu") -> Dict[str, torch.Tensor]:
    """Load a draft checkpoint state dict, normalizing key prefixes.

    Accepts: a directory (config.json + model.safetensors / pytorch_model.bin),
    or a single .pt / .bin / .safetensors file.  Strips common prefixes
    ("draft_model.", "model.", "module.") so keys match DFlashDraftModel.
    """
    if os.path.isdir(path):
        for name in ("model.safetensors", "pytorch_model.bin", "pytorch_model.pt"):
            cand = os.path.join(path, name)
            if os.path.exists(cand):
                path = cand
                break
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    if path.endswith(".safetensors"):
        try:
            from safetensors import safe_open
        except ImportError as e:
            raise RuntimeError("safetensors not installed; needed to read " + path) from e
        sd = {}
        with safe_open(path, framework="pt", device=device) as f:
            for k in f.keys():
                sd[k] = f.get_tensor(k)
    else:
        sd = torch.load(path, map_location=device, weights_only=True)

    out = {}
    prefixes = ("draft_model.", "model.", "module.")
    for k, v in sd.items():
        if isinstance(v, torch.Tensor):
            v = v.detach().clone()
        for p in prefixes:
            if k.startswith(p):
                k = k[len(p):]
                break
        out[k] = v
    return out


def load_trained_draft_model(
    config_path: str,
    ckpt_path: Optional[str],
    fusion_mode: Optional[str] = None,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cpu",
) -> Tuple[nn.Module, dict]:
    """Build model from config, then load checkpoint weights if given."""
    model, model_cfg = build_draft_model(config_path, fusion_mode, dtype, device)
    if ckpt_path:
        sd = load_checkpoint_state(ckpt_path, device=device)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print("[load] missing keys (" + str(len(missing)) + "): " + str(missing[:10]))
        if unexpected:
            print("[load] unexpected keys (" + str(len(unexpected)) + "): " + str(unexpected[:10]))
    return model, model_cfg


# ---------------------------------------------------------------------------
# Input construction
# ---------------------------------------------------------------------------

def pick_device(device: str) -> str:
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch_npu is not None and torch.npu.is_available():
            return "npu"
        return "cpu"
    return device


def build_dflash_mask(
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    S: int,
    block_size: int,
    device,
    sliding_window: Optional[int] = None,
) -> torch.Tensor:
    """Replicate create_dflash_sdpa_mask(draft_mask_mode="block")."""
    B, N = anchor_positions.shape
    Q_LEN = N * block_size
    KV_LEN = S + N * block_size
    q_indices = torch.arange(Q_LEN, device=device).view(1, 1, -1, 1)
    kv_indices = torch.arange(KV_LEN, device=device).view(1, 1, 1, -1)
    q_block_ids = q_indices // block_size
    anchor_expanded = anchor_positions.view(B, 1, N, 1).repeat_interleave(block_size, dim=2)
    q_kv_pos = anchor_expanded + (q_indices % block_size)
    mask_context = (kv_indices < S) & (kv_indices < anchor_expanded)
    if sliding_window is not None and sliding_window > 0:
        mask_context = mask_context & (kv_indices >= q_kv_pos - sliding_window)
    is_draft = kv_indices >= S
    kv_block_ids = (kv_indices - S) // block_size
    mask_draft = is_draft & (q_block_ids == kv_block_ids)
    valid_block = block_keep_mask.view(B, 1, N, 1).repeat_interleave(block_size, dim=2)
    return (mask_context | mask_draft) & valid_block


def get_per_layer_sliding_windows(model_cfg) -> List[Optional[int]]:
    out = []
    for i in range(model_cfg.num_hidden_layers):
        out.append(dflash_mod.get_layer_sliding_window(model_cfg, i))
    return out


def synthetic_target_hidden(
    B: int,
    S: int,
    T: int,
    H: int,
    seed: int = 0,
    device: str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    base_rms: Tuple[float, ...] = (3.0, 5.0, 8.0, 12.0, 18.0, 25.0, 35.0, 45.0, 60.0),
    spike_mag: float = 150.0,
    spike_per_token: float = 0.002,
    chan_outlier_scale: float = 6.0,
) -> torch.Tensor:
    """Synthetic target hidden states mimicking real LLM residual-stream
    statistics: per-layer scale disparity, token-localized massive spikes,
    and a few persistent channel outliers."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    states = []
    for t in range(T):
        rms = base_rms[min(t, len(base_rms) - 1)]
        x = torch.randn(B, S, H, generator=g) * (rms / H ** 0.5)
        # channel outliers: persistent across tokens
        n_chan = max(1, H // 512)
        chan_idx = torch.randint(0, H, (n_chan,), generator=g)
        x[:, :, chan_idx] += torch.randn(B, S, n_chan, generator=g) * (chan_outlier_scale * rms / H ** 0.5)
        # token-localized spikes
        n_spike = max(1, int(B * S * spike_per_token))
        flat = torch.randint(0, B * S, (n_spike,), generator=g)
        dims = torch.randint(0, H, (n_spike,), generator=g)
        signs = torch.where(torch.rand(n_spike, generator=g) > 0.5, 1.0, -1.0)
        x.view(B * S, H)[flat, dims] += signs * spike_mag * (rms / 12.0)
        states.append(x)
    out = torch.cat(states, dim=-1)
    return out.to(dtype=dtype).to(device)


def synthetic_noise_embedding(
    B: int,
    num_blocks: int,
    block_size: int,
    H: int,
    seed: int = 1,
    device: str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(B, num_blocks * block_size, H, generator=g) * (4.0 / H ** 0.5)
    return x.to(dtype=dtype).to(device)


def extract_target_hidden_from_hf(
    model,
    tokenizer,
    texts: List[str],
    layer_ids: List[int],
    device: str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    max_len: int = 256,
) -> torch.Tensor:
    """Run a HF causal LM and gather hidden states at layer_ids (+1 offset,
    matching extract_context_feature). Returns (B, S, T*H)."""
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attn,
            output_hidden_states=True,
            use_cache=False,
        )
    hs = out.hidden_states  # [emb, l1, ..., lL]
    sel = [hs[lid + 1].to(dtype=dtype) for lid in layer_ids]
    return torch.cat(sel, dim=-1), input_ids


# ---------------------------------------------------------------------------
# Activation statistics
# ---------------------------------------------------------------------------

def per_token_max(x: torch.Tensor) -> torch.Tensor:
    return x.abs().amax(dim=-1)


def per_token_rms(x: torch.Tensor) -> torch.Tensor:
    return x.pow(2).mean(dim=-1).sqrt()


def per_token_spikiness(x: torch.Tensor) -> torch.Tensor:
    """max|x| / RMS(x) per token - the per-token dynamic range that drives
    per-token uniform-quantizer error (SNR ~ qmax / spikiness)."""
    return per_token_max(x) / per_token_rms(x).clamp(min=1e-12)


def per_token_kurtosis(x: torch.Tensor) -> torch.Tensor:
    """Excess-free kurtosis: E[x^4]/E[x^2]^2 per token (3 = Gaussian)."""
    m2 = x.pow(2).mean(dim=-1)
    m4 = x.pow(4).mean(dim=-1)
    return m4 / m2.clamp(min=1e-12).pow(2)


def outlier_fraction(x: torch.Tensor, thresh_rms: float = 4.0) -> torch.Tensor:
    """Fraction of dims with |x| > thresh_rms * per-token RMS."""
    rms = per_token_rms(x).clamp(min=1e-12).unsqueeze(-1)
    return (x.abs() > thresh_rms * rms).float().mean(dim=-1)


def quant_sqnr_per_token(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Per-token symmetric fake-quantize (mirrors wxay.quantize_activation) and
    return per-token SQNR in dB."""
    qmax = 2 ** (bits - 1) - 1
    scale = x.abs().amax(dim=-1, keepdim=True) / qmax
    scale = scale.clamp(min=1e-12)
    x_q = torch.round(x / scale).clamp(-qmax, qmax) * scale
    sig = x.pow(2).mean(dim=-1)
    noise = (x - x_q).pow(2).mean(dim=-1)
    return 10.0 * torch.log10(sig / noise.clamp(min=1e-12))


def summarize_tensor(x: torch.Tensor, label: str, bits_iter: Iterable[int] = (4, 8)) -> dict:
    """Per-token stats summarized across all tokens (math on CPU for NPU
    compatibility; per-token summary tensors are small)."""
    x = x.detach().float().cpu()
    out = {"label": label}
    out["tokens"] = int(x.shape[0] * x.shape[1])
    out["dims"] = int(x.shape[-1])
    out["global_max_abs"] = float(x.abs().max())
    out["global_rms"] = float(x.pow(2).mean().sqrt())
    sp = per_token_spikiness(x)
    kt = per_token_kurtosis(x)
    of4 = outlier_fraction(x, 4.0)
    of8 = outlier_fraction(x, 8.0)
    out["spikiness_mean"] = float(sp.mean())
    out["spikiness_p99"] = float(sp.quantile(0.99))
    out["spikiness_max"] = float(sp.max())
    out["kurtosis_mean"] = float(kt.mean())
    out["kurtosis_p99"] = float(kt.quantile(0.99))
    out["outlier_frac_4rms_mean"] = float(of4.mean())
    out["outlier_frac_8rms_mean"] = float(of8.mean())
    for b in bits_iter:
        sqnr = quant_sqnr_per_token(x, b)
        out["sqnr_" + str(b) + "bit_mean"] = float(sqnr.mean())
        out["sqnr_" + str(b) + "bit_min"] = float(sqnr.min())
    return out


class CaptureHooks:
    """Attach forward_pre_hooks to a set of modules; aggregate per-token
    summaries on the fly (keeps memory flat)."""

    def __init__(self, model: nn.Module, name_paths: Dict[str, nn.Module]):
        self.model = model
        self.name_paths = name_paths
        self.results: Dict[str, dict] = {}
        self._handles = []
        for name, mod in name_paths.items():
            handle = mod.register_forward_pre_hook(self._make_hook(name))
            self._handles.append(handle)

    def _make_hook(self, name):
        def hook(module, args):
            x = args[0]
            if not isinstance(x, torch.Tensor):
                return
            self.results[name] = summarize_tensor(x, name)
        return hook

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles = []


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------

def print_report(report: dict, indent: int = 0):
    pad = "  " * indent
    for k, v in report.items():
        if isinstance(v, dict):
            print(pad + str(k) + ":")
            print_report(v, indent + 1)
        else:
            print(pad + str(k) + ": " + str(v))


def _sanitize(o):
    if isinstance(o, dict):
        return {k: _sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitize(v) for v in o]
    if isinstance(o, torch.Tensor):
        o = o.item() if o.numel() == 1 else o.tolist()
        return _sanitize(o)
    if isinstance(o, float):
        if o != o or o in (float("inf"), float("-inf")):
            return None
        return o
    return o


def dump_json(path: str, obj: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_sanitize(obj), f, indent=2)
    print("[report] wrote " + path)


def add_device_arg(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--device", default="auto",
        help="auto | cpu | cuda | npu (default auto)",
    )
    parser.add_argument(
        "--dtype", default="bf16", choices=["bf16", "fp16", "fp32"],
        help="model dtype (default bf16)",
    )


def resolve_dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]
