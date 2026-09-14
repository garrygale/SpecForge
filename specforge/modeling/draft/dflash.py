import copy
import math
from typing import Callable, Optional

import torch
from torch import nn
from transformers import DynamicCache
from transformers.cache_utils import Cache
from transformers.integrations.flex_attention import compile_friendly_flex_attention
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    FlashAttentionKwargs,
    GradientCheckpointingLayer,
    Qwen3Config,
    Qwen3PreTrainedModel,
    Qwen3RotaryEmbedding,
    eager_attention_forward,
    rotate_half,
)
from typing_extensions import Tuple, Unpack

try:
    from .dflash_kernels import DEFAULT_DFLASH_KERNELS, DFlashKernels
except ImportError:  # exported top-level remote-code layout
    from dflash_kernels import DEFAULT_DFLASH_KERNELS, DFlashKernels

try:
    from .registry import register_draft
except ImportError:  # exported top-level remote-code layout
    from registry import register_draft

try:
    from specforge.utils import get_device_type
except ImportError:  # exported remote-code fallback
    def get_device_type() -> str:
        return "cpu"

try:
    from .flex_attention_backend import flex_attention_backend
except ImportError:  # exported remote-code fallback
    def flex_attention_backend() -> Optional[str]:
        return None

FULL_ATTENTION = "full_attention"
SLIDING_ATTENTION = "sliding_attention"
_VALID_DFLASH_LAYER_TYPES = {FULL_ATTENTION, SLIDING_ATTENTION}


def _dflash_method_config(config) -> dict:
    return dict(getattr(config, "dflash_config", None) or {})


def get_layer_sliding_window(config, layer_idx: int) -> Optional[int]:
    """Return the sliding-window length for one DFlash layer, if any.

    ``dflash_config.sliding_window`` is preferred because transformers'
    strict ``Qwen3Config`` validation rejects per-layer lists at the top
    level; the top-level scalar remains supported for legacy configs.
    """
    layer_types = getattr(config, "layer_types", None)
    if not layer_types or layer_idx >= len(layer_types):
        return None
    if layer_types[layer_idx] != SLIDING_ATTENTION:
        return None

    method_config = _dflash_method_config(config)
    sliding_window = method_config.get("sliding_window")
    if sliding_window is None:
        sliding_window = getattr(config, "sliding_window", None)
    if sliding_window is None:
        return None
    if isinstance(sliding_window, (list, tuple)):
        if layer_idx >= len(sliding_window):
            return None
        value = sliding_window[layer_idx]
    else:
        value = sliding_window
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def has_sliding_window_attention(config) -> bool:
    """Return whether any instantiated DFlash layer uses SWA."""
    layer_types = getattr(config, "layer_types", None)
    num_layers = getattr(
        config, "num_hidden_layers", len(layer_types) if layer_types else 0
    )
    return any(
        get_layer_sliding_window(config, layer_idx) is not None
        for layer_idx in range(num_layers)
    )


def validate_dflash_sliding_window_config(config) -> None:
    """Validate scalar or per-layer sliding-window settings."""

    layer_types = getattr(config, "layer_types", None)
    num_layers = getattr(
        config, "num_hidden_layers", len(layer_types) if layer_types else 0
    )
    if num_layers is None:
        raise ValueError("DFlash draft config requires num_hidden_layers.")
    if not layer_types or len(layer_types) < num_layers:
        raise ValueError(
            "config.layer_types must describe at least the first "
            f"{num_layers} draft layers."
        )

    method_config = _dflash_method_config(config)
    raw_window = method_config.get("sliding_window")
    if raw_window is None:
        raw_window = getattr(config, "sliding_window", None)

    if isinstance(raw_window, (list, tuple)):
        if len(raw_window) != num_layers:
            raise ValueError(
                "sliding_window must have exactly one entry per draft layer: "
                f"expected {num_layers}, got {len(raw_window)}."
            )
        for layer_idx in range(num_layers):
            layer_type = layer_types[layer_idx]
            window = raw_window[layer_idx]
            if layer_type == SLIDING_ATTENTION:
                if (
                    isinstance(window, bool)
                    or not isinstance(window, int)
                    or window <= 0
                ):
                    raise ValueError(
                        f"Layer {layer_idx} is sliding_attention but "
                        f"sliding_window[{layer_idx}]={window!r}; expected a "
                        "positive integer."
                    )
            elif layer_type == FULL_ATTENTION:
                if window not in (-1, None):
                    raise ValueError(
                        f"Layer {layer_idx} is full_attention but "
                        f"sliding_window[{layer_idx}]={window!r}; expected -1."
                    )
            else:
                raise ValueError(
                    f"Unsupported layer type {layer_type!r} for layer "
                    f"{layer_idx} in a list-valued sliding_window config."
                )
        return

    for layer_idx in range(num_layers):
        if (
            layer_types[layer_idx] == SLIDING_ATTENTION
            and get_layer_sliding_window(config, layer_idx) is None
        ):
            raise ValueError(
                f"Layer {layer_idx} is marked sliding_attention but no "
                "positive sliding_window is configured. Set sliding_window "
                "to a positive integer or a per-layer list, or change "
                f"layer_types[{layer_idx}] to 'full_attention'."
            )


def validate_dflash_attention_backend(config, backend: Optional[str]) -> None:
    """Reject flex attention when sliding-window layers are configured."""
    validate_dflash_sliding_window_config(config)
    if backend == "flex_attention" and has_sliding_window_attention(config):
        raise ValueError(
            "DFlash sliding-window attention cannot be used with "
            "attention_backend='flex_attention'. Use sdpa/eager."
        )


class eagerGRU(nn.Module):
    """From-scratch GRU matching ``torch.nn.GRU`` without NPU bf16 op limits.

    Used during training so Ascend's DynamicGRU never sees bf16 activations.
    The acceptance/export paths keep a lazy ``nn.GRU`` mirror for the optimized
    NPU GRU implementation.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first

        for layer in range(num_layers):
            in_size = input_size if layer == 0 else hidden_size
            self.register_parameter(
                f"weight_ih_l{layer}",
                nn.Parameter(torch.empty(3 * hidden_size, in_size)),
            )
            self.register_parameter(
                f"weight_hh_l{layer}",
                nn.Parameter(torch.empty(3 * hidden_size, hidden_size)),
            )
            if bias:
                self.register_parameter(
                    f"bias_ih_l{layer}",
                    nn.Parameter(torch.empty(3 * hidden_size)),
                )
                self.register_parameter(
                    f"bias_hh_l{layer}",
                    nn.Parameter(torch.empty(3 * hidden_size)),
                )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = (
            1.0 / math.sqrt(self.hidden_size)
            if self.hidden_size > 0
            else 0
        )
        for parameter in self.parameters():
            nn.init.uniform_(parameter, -stdv, stdv)

    def forward(
        self,
        input: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.batch_first:
            layer_input = input.transpose(0, 1)
            batch = input.size(0)
        else:
            layer_input = input
            batch = input.size(1)
        if h0 is None:
            h0 = torch.zeros(
                self.num_layers,
                batch,
                self.hidden_size,
                device=input.device,
                dtype=input.dtype,
            )

        h_n = []
        for layer in range(self.num_layers):
            weight_ih = getattr(self, f"weight_ih_l{layer}")
            weight_hh = getattr(self, f"weight_hh_l{layer}")
            bias_ih = getattr(self, f"bias_ih_l{layer}") if self.bias else None
            bias_hh = getattr(self, f"bias_hh_l{layer}") if self.bias else None
            h = h0[layer]

            seq_len = layer_input.size(0)
            outputs = []
            for step in range(seq_len):
                x_t = layer_input[step]
                gi = x_t @ weight_ih.t()
                gh = h @ weight_hh.t()
                if bias_ih is not None:
                    gi = gi + bias_ih
                if bias_hh is not None:
                    gh = gh + bias_hh
                ri, zi, ni = gi.chunk(3, 1)
                rh, zh, nh = gh.chunk(3, 1)
                r = torch.sigmoid(ri + rh)
                z = torch.sigmoid(zi + zh)
                n = torch.tanh(ni + r * nh)
                h = (1 - z) * n + z * h
                outputs.append(h)
            layer_input = torch.stack(outputs, dim=0)
            h_n.append(h)

        output = layer_input
        if self.batch_first:
            output = output.transpose(0, 1)
        return output, torch.stack(h_n, dim=0)


def initialize_fusion_weights(weights: torch.Tensor) -> None:
    """Initialize flare fusion weights with a near-one-hot layer schedule."""
    nn.init.constant_(weights, 0.0)
    D = weights.shape[0]
    T = weights.shape[1]
    for d in range(D):
        t = min(T - 1, int((d / D) * T))
        weights.data[d, t] = 2.0


def sample(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
    if temperature < 1e-5:
        return torch.argmax(logits, dim=-1)
    bsz, seq_len, vocab_size = logits.shape
    logits = logits.view(-1, vocab_size)
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).view(bsz, seq_len)


def resolve_dflash_attention_layout(
    config: Qwen3Config,
) -> tuple[tuple[str, ...], object]:
    """Validate and return the configured per-layer DFlash attention layout."""

    num_hidden_layers = config.num_hidden_layers
    layer_types = tuple(config.layer_types)

    if len(layer_types) != num_hidden_layers:
        raise ValueError(
            "DFlash config.layer_types must contain exactly "
            f"num_hidden_layers={num_hidden_layers} entries, got "
            f"{len(layer_types)}"
        )
    invalid = set(layer_types) - _VALID_DFLASH_LAYER_TYPES
    if invalid:
        raise ValueError(
            "DFlash config.layer_types supports only full_attention and "
            f"sliding_attention, got {sorted(invalid)}"
        )

    method_config = _dflash_method_config(config)
    sliding_window = method_config.get("sliding_window")
    if sliding_window is None:
        sliding_window = getattr(config, "sliding_window", None)

    if SLIDING_ATTENTION not in layer_types:
        validate_dflash_sliding_window_config(config)
        return layer_types, None

    if sliding_window is None:
        raise ValueError(
            "DFlash sliding_attention layers require use_sliding_window=true "
            "and a positive config.sliding_window"
        )
    if isinstance(sliding_window, (list, tuple)):
        validate_dflash_sliding_window_config(config)
        return layer_types, tuple(sliding_window)
    if isinstance(sliding_window, bool) or not isinstance(sliding_window, int):
        raise ValueError(
            "DFlash sliding_window must be a positive integer or per-layer list"
        )
    if sliding_window <= 0:
        raise ValueError(
            "DFlash sliding_attention layers require a positive sliding_window"
        )
    return layer_types, sliding_window


def resolve_dflash_attention_mode(config: Qwen3Config) -> str:
    """Validate and return the configured draft attention mode.

    ``gqa`` and ``mha`` share :class:`Qwen3DFlashAttention`; ``mla`` swaps in
    the latent parameterization while retaining the family decoder,
    target-context injection, masks, and objectives.
    """

    dflash_config = getattr(config, "dflash_config", None) or {}
    attention_mode = str(dflash_config.get("attention_mode", "gqa")).lower()
    if attention_mode not in _DFLASH_ATTENTION_CLASSES:
        raise ValueError(
            "DFlash dflash_config.attention_mode must be one of "
            f"{sorted(_DFLASH_ATTENTION_CLASSES)}, got {attention_mode!r}"
        )
    return attention_mode


def _require_bool_config(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"DFlash {field} must be a boolean, got {value!r}")
    return value


def _resolve_mla_rope_interleaved(config: Qwen3Config) -> bool:
    """Rotation convention from the standard MLA ``rope_interleave`` field."""

    return _require_bool_config(
        getattr(config, "rope_interleave", True),
        "config.rope_interleave",
    )


def validate_dflash_mla_config(config: Qwen3Config) -> None:
    """Validate the standard MLA dimension fields carried by a draft config."""

    required = (
        "kv_lora_rank",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "v_head_dim",
    )
    missing = [name for name in required if getattr(config, name, None) is None]
    if missing:
        raise ValueError(f"MLA draft config is missing required fields: {missing}")

    q_lora_rank = getattr(config, "q_lora_rank", None)
    if q_lora_rank is not None and int(q_lora_rank) <= 0:
        raise ValueError(f"q_lora_rank must be positive or null, got {q_lora_rank}")

    for name in ("kv_lora_rank", "qk_rope_head_dim", "v_head_dim"):
        value = int(getattr(config, name))
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")

    qk_nope_head_dim = int(config.qk_nope_head_dim)
    if qk_nope_head_dim < 0:
        raise ValueError(
            f"qk_nope_head_dim must be non-negative, got {qk_nope_head_dim}"
        )
    qk_rope_head_dim = int(config.qk_rope_head_dim)
    if qk_rope_head_dim % 2:
        raise ValueError(f"qk_rope_head_dim must be even, got {qk_rope_head_dim}")
    _resolve_mla_rope_interleaved(config)


def validate_dflash_attention_config(config: Qwen3Config) -> str:
    """Validate the selected attention parameterization and return its mode."""

    attention_mode = resolve_dflash_attention_mode(config)
    if attention_mode == "mha" and int(config.num_key_value_heads) != int(
        config.num_attention_heads
    ):
        raise ValueError(
            "attention_mode 'mha' requires num_key_value_heads == "
            f"num_attention_heads, got {config.num_key_value_heads} and "
            f"{config.num_attention_heads}"
        )
    if attention_mode == "mla":
        validate_dflash_mla_config(config)
    return attention_mode


def _rope_config(config: Qwen3Config, attention_mode: str) -> Qwen3Config:
    """Rotary config for the mode: MLA rotates only the partial-RoPE slice."""

    if attention_mode != "mla":
        return config
    rope_config = copy.deepcopy(config)
    rope_config.head_dim = config.qk_rope_head_dim
    return rope_config


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_len = q.size(-2)
    q_embed = (q * cos[..., -q_len:, :]) + (rotate_half(q) * sin[..., -q_len:, :])
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def _rotate_half_interleaved(x: torch.Tensor) -> torch.Tensor:
    """Rotate consecutive pairs, the DeepSeek-style MLA RoPE convention."""

    paired = x.reshape(*x.shape[:-1], -1, 2)
    first, second = paired.unbind(dim=-1)
    return torch.stack((-second, first), dim=-1).flatten(-2)


def apply_mla_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    interleaved: bool,
) -> torch.Tensor:
    if interleaved:
        half = cos.shape[-1] // 2
        cos = cos[..., :half].repeat_interleave(2, dim=-1)
        sin = sin[..., :half].repeat_interleave(2, dim=-1)
        rotated = _rotate_half_interleaved(x)
    else:
        rotated = rotate_half(x)
    return x * cos.unsqueeze(1) + rotated * sin.unsqueeze(1)


def _prepare_dflash_eager_mask(
    attention_mask: Optional[torch.Tensor],
    dtype: torch.dtype,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Convert a boolean allow-mask to eager's additive representation."""

    if attention_mask is None or attention_mask.dtype != torch.bool:
        return attention_mask, None

    valid_queries = attention_mask.any(dim=-1, keepdim=True)
    # A finite minimum keeps eager softmax stable. Fully masked query rows are
    # explicitly zeroed after attention so they cannot average forbidden values.
    additive_mask = torch.zeros_like(attention_mask, dtype=dtype)
    additive_mask.masked_fill_(~attention_mask, torch.finfo(dtype).min)
    return additive_mask, valid_queries


class Qwen3DFlashAttentionBase(nn.Module):
    """Shared scaffold for the DFlash-family attention modes.

    Subclasses own only the projection parameterization: ``_init_projections``
    builds the weights and must define ``scaling``, ``num_key_value_groups``,
    and ``o_proj`` (the shared forward relies on them); ``_compute_qkv``
    returns rotated ``(q, k, v)`` in ``(batch, heads, seq, dim)`` layout with
    keys ordered context-then-draft. Everything the modes must agree on —
    KV-cache updates, backend dispatch, fully-masked-query zeroing, and the
    output projection — lives here so it is maintained in exactly one place.
    """

    def __init__(
        self,
        config: Qwen3Config,
        layer_idx: int,
        kernels: DFlashKernels,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attention_dropout = config.attention_dropout
        if getattr(config, "_attn_implementation", None) == "flex_attention":
            assert (
                config.attention_dropout == 0.0
            ), "DFlash FlexAttention requires attention_dropout=0.0"
        self.is_causal = False
        self.sliding_window = get_layer_sliding_window(config, layer_idx)
        self._init_projections(config, kernels)
        for attribute in ("scaling", "num_key_value_groups", "o_proj"):
            assert hasattr(
                self, attribute
            ), f"_init_projections must define {attribute}"

    def _init_projections(self, config: Qwen3Config, kernels: DFlashKernels) -> None:
        raise NotImplementedError

    def _compute_qkv(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len = hidden_states.shape[:-1]
        q, k, v = self._compute_qkv(hidden_states, target_hidden, position_embeddings)
        if past_key_values is not None:
            cos, sin = position_embeddings
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)
        valid_queries = None
        attn_implementation = getattr(
            self.config, "_attn_implementation", None
        ) or "sdpa"
        if attn_implementation == "flex_attention":
            kernel_options = dict(kwargs.pop("kernel_options", None) or {})
            backend = flex_attention_backend()
            if backend is not None:
                kernel_options["BACKEND"] = backend

            attn_output = compile_friendly_flex_attention(
                q,
                k,
                v,
                block_mask=attention_mask,
                enable_gqa=True,
                scale=self.scaling,
                kernel_options=kernel_options or None,
            ).transpose(1, 2)
            attn_weights = None
        else:
            attn_fn: Callable = eager_attention_forward
            if attn_implementation == "eager":
                attention_mask, valid_queries = _prepare_dflash_eager_mask(
                    attention_mask,
                    q.dtype,
                )
            else:
                attn_fn = ALL_ATTENTION_FUNCTIONS[attn_implementation]
            attn_output, attn_weights = attn_fn(
                self,
                q,
                k,
                v,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                sliding_window=self.sliding_window,
                **kwargs,
            )
            if valid_queries is not None and attn_weights is not None:
                attn_weights = attn_weights.masked_fill(~valid_queries, 0)
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        if valid_queries is not None:
            attn_output = attn_output.masked_fill(
                ~valid_queries.any(dim=1),
                0,
            )
        return attn_output, attn_weights


class Qwen3DFlashAttention(Qwen3DFlashAttentionBase):
    """GQA/MHA projections over the family's context-then-draft KV layout."""

    def _init_projections(self, config: Qwen3Config, kernels: DFlashKernels) -> None:
        method_config = _dflash_method_config(config)
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.heterogeneous_kv = bool(method_config.get("heterogeneous_kv", False))
        self.target_hidden_size = int(
            method_config.get("target_hidden_size", config.hidden_size)
        )
        if self.heterogeneous_kv:
            self.k_proj_target = nn.Linear(
                self.target_hidden_size,
                config.num_key_value_heads * self.head_dim,
                bias=config.attention_bias,
            )
            self.v_proj_target = nn.Linear(
                self.target_hidden_size,
                config.num_key_value_heads * self.head_dim,
                bias=config.attention_bias,
            )
        self.q_norm = kernels.make_rms_norm(self.head_dim, config.rms_norm_eps)
        self.k_norm = kernels.make_rms_norm(self.head_dim, config.rms_norm_eps)

    def _compute_qkv(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_hidden.shape[1]
        q = self.q_proj(hidden_states)
        q = q.view(bsz, q_len, -1, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)
        if self.heterogeneous_kv:
            k_ctx = self.k_proj_target(target_hidden)
            v_ctx = self.v_proj_target(target_hidden)
        else:
            k_ctx = self.k_proj(target_hidden)
            v_ctx = self.v_proj(target_hidden)
        k_noise = self.k_proj(hidden_states)
        v_noise = self.v_proj(hidden_states)
        k = torch.cat([k_ctx, k_noise], dim=1).view(
            bsz, ctx_len + q_len, -1, self.head_dim
        )
        v = torch.cat([v_ctx, v_noise], dim=1).view(
            bsz, ctx_len + q_len, -1, self.head_dim
        )
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        return q, k, v


class Qwen3DFlashMLAAttention(Qwen3DFlashAttentionBase):
    """Multi-head Latent Attention projections for DFlash-family drafts.

    Standard MLA parameterization: an optional low-rank Q path, a shared
    compressed KV latent, and partial RoPE (interleaved or NeoX, from the
    standard ``rope_interleave`` field). K/V are expanded per head for
    training so the mode runs through the same masks and attention backends
    as :class:`Qwen3DFlashAttention`.
    """

    def _init_projections(self, config: Qwen3Config, kernels: DFlashKernels) -> None:
        self.num_heads = int(config.num_attention_heads)
        self.num_key_value_groups = 1
        self.q_lora_rank = (
            None
            if getattr(config, "q_lora_rank", None) is None
            else int(config.q_lora_rank)
        )
        self.kv_lora_rank = int(config.kv_lora_rank)
        self.qk_nope_head_dim = int(config.qk_nope_head_dim)
        self.qk_rope_head_dim = int(config.qk_rope_head_dim)
        self.v_head_dim = int(config.v_head_dim)
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.head_dim = self.qk_head_dim
        self.scaling = self.qk_head_dim**-0.5
        # DeepSeek YaRN applies mscale_all_dim to the full QK logits; the
        # rotary attention factor separately scales only the partial-RoPE slice.
        rope_parameters = config.rope_parameters
        if rope_parameters.get("rope_type", "default") != "default":
            mscale_all_dim = rope_parameters.get("mscale_all_dim", 0)
            if mscale_all_dim:
                factor = rope_parameters["factor"]
                mscale = (
                    1.0
                    if factor <= 1
                    else 0.1 * mscale_all_dim * math.log(factor) + 1.0
                )
                self.scaling *= mscale * mscale
        self.rope_interleaved = _resolve_mla_rope_interleaved(config)

        hidden_size = int(config.hidden_size)
        bias = bool(config.attention_bias)
        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(
                hidden_size,
                self.num_heads * self.qk_head_dim,
                bias=False,
            )
        else:
            self.q_a_proj = nn.Linear(hidden_size, self.q_lora_rank, bias=bias)
            self.q_a_layernorm = kernels.make_rms_norm(
                self.q_lora_rank,
                config.rms_norm_eps,
            )
            self.q_b_proj = nn.Linear(
                self.q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=False,
            )
        self.kv_a_proj_with_mqa = nn.Linear(
            hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=bias,
        )
        self.kv_a_layernorm = kernels.make_rms_norm(
            self.kv_lora_rank,
            config.rms_norm_eps,
        )
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            hidden_size,
            bias=bias,
        )

    def _project_q(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.q_lora_rank is None:
            return self.q_proj(hidden_states)
        return self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))

    def _project_kv(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seq_len = hidden_states.shape[:2]
        kv_compressed, k_rope = self.kv_a_proj_with_mqa(hidden_states).split(
            [self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1,
        )
        kv = self.kv_b_proj(self.kv_a_layernorm(kv_compressed)).view(
            bsz,
            seq_len,
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        k_nope, value = kv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        return k_nope, k_rope, value

    def _compute_qkv(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, q_len = hidden_states.shape[:2]
        query = self._project_q(hidden_states).view(
            bsz,
            q_len,
            self.num_heads,
            self.qk_head_dim,
        )
        q_nope, q_rope = query.split(
            [self.qk_nope_head_dim, self.qk_rope_head_dim],
            dim=-1,
        )
        ctx_k_nope, ctx_k_rope, ctx_value = self._project_kv(target_hidden)
        noise_k_nope, noise_k_rope, noise_value = self._project_kv(hidden_states)
        k_nope = torch.cat((ctx_k_nope, noise_k_nope), dim=1)
        k_rope = torch.cat((ctx_k_rope, noise_k_rope), dim=1)
        v = torch.cat((ctx_value, noise_value), dim=1)
        # The model-level rotary carries qk_rope_head_dim; queries take the
        # trailing positions exactly like apply_rotary_pos_emb.
        cos, sin = position_embeddings
        q_rope = apply_mla_rope(
            q_rope.transpose(1, 2),
            cos[:, -q_len:],
            sin[:, -q_len:],
            interleaved=self.rope_interleaved,
        )
        k_rope = apply_mla_rope(
            k_rope.unsqueeze(1),
            cos,
            sin,
            interleaved=self.rope_interleaved,
        ).expand(-1, self.num_heads, -1, -1)
        q = torch.cat((q_nope.transpose(1, 2), q_rope), dim=-1)
        k = torch.cat((k_nope.transpose(1, 2), k_rope), dim=-1)
        return q, k, v.transpose(1, 2)


_DFLASH_ATTENTION_CLASSES = {
    "gqa": Qwen3DFlashAttention,
    "mha": Qwen3DFlashAttention,
    "mla": Qwen3DFlashMLAAttention,
}


class Qwen3DFlashDecoderLayer(GradientCheckpointingLayer):
    def __init__(
        self,
        config: Qwen3Config,
        layer_idx: int,
        kernels: DFlashKernels,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        attention_cls = _DFLASH_ATTENTION_CLASSES[resolve_dflash_attention_mode(config)]
        self.self_attn = attention_cls(
            config=config,
            layer_idx=layer_idx,
            kernels=kernels,
        )
        self.mlp = kernels.make_mlp(config)
        self.input_layernorm = kernels.make_rms_norm(
            config.hidden_size, config.rms_norm_eps
        )
        self.post_attention_layernorm = kernels.make_rms_norm(
            config.hidden_size, config.rms_norm_eps
        )

    def forward(
        self,
        target_hidden: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )[0]
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


def build_target_layer_ids(num_target_layers: int, num_draft_layers: int):
    if num_draft_layers == 1:
        return [num_target_layers // 2]
    start = 1
    end = num_target_layers - 3
    span = end - start
    target_layer_ids = [
        int(round(start + (i * span) / (num_draft_layers - 1)))
        for i in range(num_draft_layers)
    ]
    return target_layer_ids


def extract_context_feature(
    hidden_states: list[torch.Tensor],
    layer_ids: Optional[list[int]],
) -> torch.Tensor:
    offset = 1
    selected_states = []
    for layer_id in layer_ids:
        selected_states.append(hidden_states[layer_id + offset])
    target_hidden = torch.cat(selected_states, dim=-1)
    return target_hidden


def is_complete_block(start: int, block_size: int, max_length: int) -> bool:
    """Return whether all draft positions in this block fit in max_length."""
    return start + block_size <= max_length


def compute_acceptance_stats(
    acceptance_lengths: list[int],
    block_complete_flags: list[bool],
    block_size: int,
) -> dict:
    """Compute mean acceptance over complete blocks, preserving all lengths."""
    complete_lengths = [
        al
        for al, ok in zip(acceptance_lengths, block_complete_flags)
        if ok
    ]
    num_incomplete = len(block_complete_flags) - len(complete_lengths)
    mean_accept = (
        float(sum(complete_lengths) / len(complete_lengths))
        if complete_lengths
        else None
    )
    return {
        "acceptance_lengths": acceptance_lengths,
        "mean_acceptance_length": mean_accept,
        "num_complete_blocks": len(complete_lengths),
        "num_incomplete_blocks": num_incomplete,
        "block_size": block_size,
    }


def _target_text_model(target: nn.Module) -> nn.Module:
    """Resolve the text-only causal model from a conditional-generation target."""
    get_language_model = getattr(target, "get_language_model", None)
    if callable(get_language_model):
        language_model = get_language_model()
        if language_model is not None:
            return language_model
    language_model = getattr(target, "language_model", None)
    if language_model is not None:
        return language_model
    candidate = getattr(target, "model", None)
    if candidate is not None and getattr(candidate, "language_model", None) is not None:
        return candidate.language_model
    if (
        candidate is not None
        and getattr(candidate, "embed_tokens", None) is not None
        and getattr(candidate, "lm_head", None) is not None
    ):
        return candidate
    return target


def _target_embed_tokens(target: nn.Module) -> nn.Module:
    text_model = _target_text_model(target)
    embed_tokens = getattr(text_model, "embed_tokens", None)
    if embed_tokens is not None:
        return embed_tokens
    return getattr(text_model.model, "embed_tokens")


def _target_lm_head(target: nn.Module) -> nn.Module:
    text_model = _target_text_model(target)
    lm_head = getattr(text_model, "lm_head", None)
    if lm_head is not None:
        return lm_head
    return getattr(target, "lm_head")


def _target_device(target: nn.Module) -> torch.device:
    return next(target.parameters()).device


def normalize_draft_head_checkpoint_keys(
    module,
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
):
    """Map checkpoint-only nested head names onto the direct module layout.

    Early Domino/DSpark checkpoints saved their auxiliary heads beneath a
    ``logit_head`` container. The live architecture no longer owns that wrapper,
    but those tensors remain valid and must not be dropped during warm start or
    full resume.
    """

    del module, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    checkpoint_prefixes = (
        ("logit_head.prefix_gru.", "prefix_gru."),
        ("logit_head.embed_proj.", "embed_proj."),
        ("logit_head.markov_head.", "markov_head."),
        ("logit_head.confidence_head.", "confidence_head."),
    )
    for key in list(state_dict):
        if not key.startswith(prefix):
            continue
        local_key = key[len(prefix) :]
        for checkpoint_prefix, model_prefix in checkpoint_prefixes:
            if not local_key.startswith(checkpoint_prefix):
                continue
            normalized_key = prefix + model_prefix + local_key[len(checkpoint_prefix) :]
            if normalized_key not in state_dict:
                state_dict[normalized_key] = state_dict[key]
            state_dict.pop(key)
            break


@register_draft
class DFlashDraftModel(Qwen3PreTrainedModel):
    config_class = Qwen3Config
    _no_split_modules = ["Qwen3DFlashDecoderLayer"]
    decoder_layer_class = Qwen3DFlashDecoderLayer

    def __init__(
        self,
        config,
        dflash_kernels: Optional[DFlashKernels] = None,
    ) -> None:
        super().__init__(config)
        self.config = config
        self.layer_types, self.sliding_window = resolve_dflash_attention_layout(config)
        if getattr(config, "_attn_implementation", None) is None:
            config._attn_implementation = "sdpa"
        self.attention_mode = validate_dflash_attention_config(config)
        kernels = dflash_kernels or DEFAULT_DFLASH_KERNELS
        dflash_config = getattr(config, "dflash_config", {}) or {}
        block_size = getattr(config, "block_size", None)
        if block_size is None:
            block_size = dflash_config.get("block_size")
        if not isinstance(block_size, int) or isinstance(block_size, bool):
            raise ValueError(
                "DFlash config must define an integer block_size either at "
                "config.block_size or config.dflash_config.block_size"
            )
        self.block_size = block_size
        self.layers = nn.ModuleList(
            [
                self._build_decoder_layer(config, layer_idx, kernels)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.target_layer_ids = dflash_config.get(
            "target_layer_ids",
            build_target_layer_ids(config.num_target_layers, config.num_hidden_layers),
        )
        self.target_hidden_size = int(
            dflash_config.get("target_hidden_size", config.hidden_size)
        )
        self.norm = kernels.make_rms_norm(
            self.target_hidden_size, config.rms_norm_eps
        )
        self.rotary_emb = Qwen3RotaryEmbedding(
            _rope_config(config, self.attention_mode)
        )
        self._flare = dflash_config.get("fusion_mode") == "flare"
        if self._flare and isinstance(self.target_layer_ids[0], list):
            raise ValueError(
                "fusion_mode='flare' is not compatible with per-layer "
                "target_layer_ids; use a flat list."
            )
        self._heterogeneous_kv = bool(dflash_config.get("heterogeneous_kv", False))
        if (
            self._flare
            and self.target_hidden_size != config.hidden_size
            and not self._heterogeneous_kv
        ):
            raise ValueError(
                "flare with target_hidden_size != hidden_size requires "
                "dflash_config.heterogeneous_kv=true"
            )
        self._per_layer = (
            not self._flare
            and bool(self.target_layer_ids)
            and isinstance(self.target_layer_ids, list)
            and isinstance(self.target_layer_ids[0], list)
        )

        if self._flare:
            self.num_target_layers = len(self.target_layer_ids)
            self.layer_fusion_weights = nn.Parameter(
                torch.empty(
                    config.num_hidden_layers,
                    self.num_target_layers,
                )
            )
            self._init_fusion_weights()
            self.hidden_norm = kernels.make_rms_norm(
                self.target_hidden_size, config.rms_norm_eps
            )
            self.fc = None
            self.fcs = None
            self.hidden_norms = None
        elif self._per_layer:
            if len(self.target_layer_ids) != config.num_hidden_layers:
                raise ValueError(
                    "per-layer target_layer_ids must have one sub-list per "
                    "draft layer"
                )
            unique_ids = sorted(
                {lid for sublist in self.target_layer_ids for lid in sublist}
            )
            id_to_pos = {lid: i for i, lid in enumerate(unique_ids)}
            H = self.target_hidden_size
            self.fcs = nn.ModuleList()
            self.hidden_norms = nn.ModuleList()
            self._per_layer_gather = []
            for sublist in self.target_layer_ids:
                sub_unique = sorted(set(sublist))
                if not sub_unique:
                    raise ValueError(
                        "each per-layer target_layer_ids sub-list must be non-empty"
                    )
                self._per_layer_gather.append(
                    [id_to_pos[lid] for lid in sub_unique]
                )
                self.fcs.append(
                    nn.Linear(
                        len(sub_unique) * H,
                        config.hidden_size,
                        bias=False,
                    )
                )
                self.hidden_norms.append(
                    kernels.make_rms_norm(
                        config.hidden_size, config.rms_norm_eps
                    )
                )
            self.fc = None
            self.hidden_norm = None
        else:
            self.fc = nn.Linear(
                len(self.target_layer_ids) * self.target_hidden_size,
                config.hidden_size,
                bias=False,
            )
            self.hidden_norm = kernels.make_rms_norm(
                config.hidden_size, config.rms_norm_eps
            )
            self.fcs = None
            self.hidden_norms = None

        if self.target_hidden_size != config.hidden_size:
            self.input_proj = nn.Linear(
                self.target_hidden_size, config.hidden_size, bias=False
            )
            self.output_proj = nn.Linear(
                config.hidden_size, self.target_hidden_size, bias=False
            )
        else:
            self.input_proj = None
            self.output_proj = None
        self.layer_sliding_windows = tuple(
            get_layer_sliding_window(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        )
        validate_dflash_attention_backend(
            config, getattr(config, "_attn_implementation", None)
        )
        self.mask_token_id = dflash_config.get("mask_token_id", None)
        self.projector_type = dflash_config.get("projector_type", None)
        self.pure_draft_prefix_len = dflash_config.get("pure_draft_prefix_len", 0)
        self.shift_label = dflash_config.get("shift_label", False)
        self._init_draft_head(config, dflash_config)
        self.register_load_state_dict_pre_hook(normalize_draft_head_checkpoint_keys)
        self.post_init()

    def _build_decoder_layer(
        self,
        config: Qwen3Config,
        layer_idx: int,
        kernels: DFlashKernels,
    ) -> nn.Module:
        """Build one backbone layer; architecture variants override this seam."""

        return self.decoder_layer_class(config, layer_idx, kernels)

    def _init_draft_head(self, config, dflash_config: dict) -> None:
        del config, dflash_config

    def _init_fusion_weights(self) -> None:
        initialize_fusion_weights(self.layer_fusion_weights)

    @property
    def capture_layer_ids(self):
        if self._per_layer:
            return sorted(
                {lid for sublist in self.target_layer_ids for lid in sublist}
            )
        return self.target_layer_ids

    def apply_logits_head(
        self,
        base_logits: torch.Tensor,
        *,
        prev_token_ids: Optional[torch.Tensor] = None,
        prev_token_embeddings: Optional[torch.Tensor] = None,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        del prev_token_ids, prev_token_embeddings, hidden_states
        return base_logits

    def apply_markov_logits(
        self,
        base_logits: torch.Tensor,
        *,
        prev_token_ids: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.apply_logits_head(
            base_logits,
            prev_token_ids=prev_token_ids,
            hidden_states=hidden_states,
        )

    def predict_confidence(
        self,
        hidden_states: torch.Tensor,
        *,
        prev_token_ids: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        del hidden_states, prev_token_ids
        return None

    def _get_inference_gru(self):
        """Lazily create an ``nn.GRU`` mirroring the Domino prefix GRU."""
        if not hasattr(self, "_inference_gru") or self._inference_gru is None:
            device = next(self.prefix_gru.parameters()).device
            dtype = next(self.prefix_gru.parameters()).dtype
            self._inference_gru = nn.GRU(
                input_size=self.target_hidden_size,
                hidden_size=self.gru_hidden_dim,
                num_layers=1,
                batch_first=True,
                bias=False,
            ).to(device=device, dtype=dtype)
            self._inference_gru.load_state_dict(
                self.prefix_gru.state_dict(),
                strict=True,
            )
        return self._inference_gru

    def _run_inference_gru(self, input: torch.Tensor, h0: Optional[torch.Tensor] = None):
        """Run the inference GRU, with the existing NPU bf16 workaround."""
        gru = self._get_inference_gru()
        if get_device_type() == "npu" and input.dtype == torch.bfloat16:
            from torch.func import functional_call

            fp16_parameters = {
                name: parameter.to(dtype=torch.float16)
                for name, parameter in gru.named_parameters()
            }
            args = (input.to(dtype=torch.float16),)
            if h0 is not None:
                args = args + (h0.to(dtype=torch.float16),)
            output, h_n = functional_call(
                gru,
                fp16_parameters,
                args,
                strict=True,
            )
            return output.to(dtype=input.dtype), h_n.to(dtype=input.dtype)
        return gru(input, h0)

    def _domino_generate_step(
        self,
        start: int,
        block_size: int,
        target: nn.Module,
        output_ids: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values_target,
        past_key_values_draft,
        target_hidden: torch.Tensor,
        temperature: float,
    ):
        """One Domino decode step matching vLLM's anchor-as-first draft path."""
        text_target = _target_text_model(target)
        target_embed = _target_embed_tokens(target)
        target_lm_head = _target_lm_head(target)
        block_output_ids = output_ids[:, start : start + block_size].clone()
        noise_embedding = target_embed(block_output_ids)
        anchor_positions = torch.full(
            (target_hidden.shape[0], 1),
            target_hidden.shape[1] - 1,
            dtype=torch.long,
            device=target_hidden.device,
        )
        full_hidden = self(
            target_hidden=target_hidden,
            noise_embedding=noise_embedding,
            anchor_positions=anchor_positions,
            position_ids=position_ids[
                :,
                past_key_values_draft.get_seq_length() : start + block_size,
            ],
            past_key_values=past_key_values_draft,
            use_cache=True,
            is_causal=False,
        )
        past_key_values_draft.crop(start)

        k_draft = block_size
        prefix_len = int(getattr(self, "pure_draft_prefix_len", 0))
        base_logits = target_lm_head(full_hidden)

        verify_ids = torch.full(
            (1, k_draft + 1),
            self.mask_token_id,
            dtype=torch.long,
            device=_target_device(target),
        )
        verify_ids[:, 0] = block_output_ids[:, 0]
        if prefix_len > 0:
            verify_ids[:, 1 : 1 + prefix_len] = sample(
                base_logits[:, :prefix_len],
                temperature,
            )
        realized_prefix_ids = verify_ids[:, : 1 + prefix_len]
        realized_prefix_embeds = target_embed(realized_prefix_ids)
        _, gru_hidden = self._run_inference_gru(realized_prefix_embeds)

        for i in range(prefix_len, k_draft):
            z_i = full_hidden[:, i : i + 1]
            s_i = gru_hidden.transpose(0, 1)
            concat = torch.cat([z_i, s_i], dim=-1)
            logits_i = base_logits[:, i : i + 1]
            if self.embed_proj is not None:
                logits_i = logits_i + self.embed_proj(concat)
            if self.hidden_proj is not None:
                logits_i = logits_i + target_lm_head(self.hidden_proj(concat))
            current_token_id = sample(logits_i, temperature)
            verify_ids[:, i + 1 : i + 2] = current_token_id
            if i + 1 < k_draft:
                new_embed = target_embed(current_token_id)
                _, gru_hidden = self._run_inference_gru(new_embed, gru_hidden)

        verify_position_ids = position_ids[:, start : start + k_draft + 1]
        output = text_target(
            verify_ids,
            position_ids=verify_position_ids,
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=True,
        )
        posterior = sample(output.logits, temperature)
        acceptance_length = (
            (verify_ids[:, 1:] == posterior[:, :-1])
            .long()
            .cumprod(dim=1)
            .sum(dim=1)[0]
            .item()
        )
        return output, posterior, acceptance_length, verify_ids, k_draft

    def _sample_draft_tokens(
        self,
        target: nn.Module,
        draft_hidden: torch.Tensor,
        block_output_ids: torch.LongTensor,
    ) -> torch.LongTensor:
        """Sample one speculative block from the draft-model hidden states.

        DFlash predicts the whole suffix in one LM-head call. Draft families
        with an auxiliary logits head can override this boundary without
        duplicating the target-cache and acceptance logic in ``spec_generate``.
        """
        del block_output_ids
        draft_logits = _target_lm_head(target)(
            draft_hidden[:, -self.block_size + 1 :, :]
        )
        return sample(draft_logits)

    def forward(
        self,
        position_ids: torch.LongTensor,
        attention_mask: Optional[object] = None,
        noise_embedding: Optional[torch.Tensor] = None,
        target_hidden: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        hidden_states = noise_embedding
        if self.input_proj is not None:
            hidden_states = self.input_proj(hidden_states)

        if self._flare:
            B, S, _ = target_hidden.shape
            H = self.target_hidden_size
            T = self.num_target_layers
            target_reshaped = target_hidden.view(B, S, T, H)
            flare_weights = torch.softmax(self.layer_fusion_weights, dim=1)
            per_layer_targets = [
                self.hidden_norm(
                    (target_reshaped * flare_weights[i].view(1, 1, -1, 1)).sum(
                        dim=2
                    )
                )
                for i in range(len(self.layers))
            ]
        elif self._per_layer:
            H = self.target_hidden_size
            per_layer_targets = [
                self.hidden_norms[i](
                    self.fcs[i](
                        torch.cat(
                            [
                                target_hidden[
                                    :, :, idx * H : (idx + 1) * H
                                ]
                                for idx in self._per_layer_gather[i]
                            ],
                            dim=-1,
                        )
                    )
                )
                for i in range(len(self.layers))
            ]
        else:
            target_hidden = self.hidden_norm(self.fc(target_hidden))
            per_layer_targets = None

        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer_idx, layer in enumerate(self.layers):
            if isinstance(attention_mask, dict):
                layer_type = self.layer_types[layer_idx]
                layer_attention_mask = attention_mask.get(layer_type)
            elif isinstance(attention_mask, list):
                layer_attention_mask = attention_mask[layer_idx]
            else:
                layer_attention_mask = attention_mask
            if self._flare or self._per_layer:
                layer_target = per_layer_targets[layer_idx]
            else:
                layer_target = target_hidden
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden=layer_target,
                attention_mask=layer_attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        if self.output_proj is not None:
            hidden_states = self.output_proj(hidden_states)
        return self.norm(hidden_states)

    @torch.inference_mode()
    def spec_generate(
        self,
        target: nn.Module,
        input_ids: torch.LongTensor,
        max_new_tokens: int,
        stop_token_ids: list[int],
        temperature: float,
        return_acceptance_stats: bool = False,
    ):
        self.eval()
        text_target = _target_text_model(target)
        target_embed = _target_embed_tokens(target)
        num_input_tokens = input_ids.shape[1]
        max_length = num_input_tokens + max_new_tokens

        block_size = self.block_size
        output_ids = torch.full(
            (1, max_length + block_size),
            self.mask_token_id,
            dtype=torch.long,
            device=_target_device(target),
        )
        position_ids = torch.arange(
            output_ids.shape[1], device=_target_device(target)
        ).unsqueeze(0)

        past_key_values_target = DynamicCache()
        past_key_values_draft = DynamicCache()

        # Prefill stage
        output = text_target(
            input_ids,
            position_ids=position_ids[:, :num_input_tokens],
            past_key_values=past_key_values_target,
            use_cache=True,
            logits_to_keep=1,
            output_hidden_states=True,
        )

        output_ids[:, :num_input_tokens] = input_ids
        output_ids[:, num_input_tokens : num_input_tokens + 1] = sample(
            output.logits, temperature
        )
        target_hidden = extract_context_feature(
            output.hidden_states, self.target_layer_ids
        )

        # Decode stage
        acceptance_lengths = []
        block_complete_flags = []
        per_pos_correct = torch.zeros(block_size, device=_target_device(target))
        per_pos_total = torch.zeros(block_size, device=_target_device(target))
        start = input_ids.shape[1]
        while start < max_length:
            if getattr(self, "projector_type", None) == "domino":
                block_is_complete = is_complete_block(
                    start, block_size + 1, max_length
                )
                output, posterior, acceptance_length, verify_ids, _ = (
                    self._domino_generate_step(
                        start,
                        block_size,
                        text_target,
                        output_ids,
                        position_ids,
                        past_key_values_target,
                        past_key_values_draft,
                        target_hidden,
                        temperature,
                    )
                )
                pos_match = (verify_ids[:, 1:] == posterior[:, :-1]).float()
                per_pos_correct += pos_match[0]
                per_pos_total += 1
                output_ids[:, start : start + acceptance_length + 1] = (
                    verify_ids[:, : acceptance_length + 1]
                )
                if start + acceptance_length + 1 < output_ids.shape[1]:
                    output_ids[:, start + acceptance_length + 1] = posterior[
                        :, acceptance_length
                    ]
                start += acceptance_length + 1
                past_key_values_target.crop(start)
                target_hidden = extract_context_feature(
                    output.hidden_states, self.target_layer_ids
                )[:, : acceptance_length + 1, :]
            else:
                block_is_complete = is_complete_block(
                    start, block_size, max_length
                )
                block_output_ids = output_ids[
                    :, start : start + block_size
                ].clone()
                block_position_ids = position_ids[:, start : start + block_size]
                noise_embedding = target_embed(block_output_ids)
                draft_hidden = self(
                    target_hidden=target_hidden,
                    noise_embedding=noise_embedding,
                    position_ids=position_ids[
                        :,
                        past_key_values_draft.get_seq_length() : start
                        + block_size,
                    ],
                    past_key_values=past_key_values_draft,
                    use_cache=True,
                    is_causal=False,
                )
                past_key_values_draft.crop(start)
                block_output_ids[:, 1:] = self._sample_draft_tokens(
                    text_target,
                    draft_hidden,
                    block_output_ids,
                )

                output = text_target(
                    block_output_ids,
                    position_ids=block_position_ids,
                    past_key_values=past_key_values_target,
                    use_cache=True,
                    output_hidden_states=True,
                )

                posterior = sample(output.logits, temperature)
                acceptance_length = (
                    (block_output_ids[:, 1:] == posterior[:, :-1])
                    .long()
                    .cumprod(dim=1)
                    .sum(dim=1)[0]
                    .item()
                )
                pos_match = (
                    block_output_ids[:, 1:] == posterior[:, :-1]
                ).float()
                per_pos_correct[: block_size - 1] += pos_match[0]
                per_pos_total[: block_size - 1] += 1
                output_ids[
                    :, start : start + acceptance_length + 1
                ] = block_output_ids[:, : acceptance_length + 1]
                if start + acceptance_length + 1 < output_ids.shape[1]:
                    output_ids[:, start + acceptance_length + 1] = posterior[
                        :, acceptance_length
                    ]
                start += acceptance_length + 1
                past_key_values_target.crop(start)
                target_hidden = extract_context_feature(
                    output.hidden_states, self.target_layer_ids
                )[:, : acceptance_length + 1, :]
            acceptance_lengths.append(acceptance_length + 1)
            block_complete_flags.append(block_is_complete)
            if stop_token_ids is not None and any(
                stop_token_id in output_ids[:, num_input_tokens:]
                for stop_token_id in stop_token_ids
            ):
                break
        output_ids = output_ids[:, :max_length]
        output_ids = output_ids[:, output_ids[0] != self.mask_token_id]
        if stop_token_ids is not None:
            stop_token_ids = torch.tensor(stop_token_ids, device=output_ids.device)
            stop_token_indices = torch.isin(
                output_ids[0][num_input_tokens:], stop_token_ids
            ).nonzero(as_tuple=True)[0]
            if stop_token_indices.numel() > 0:
                output_ids = output_ids[
                    :, : num_input_tokens + stop_token_indices[0] + 1
                ]

        if return_acceptance_stats:
            stats = compute_acceptance_stats(
                acceptance_lengths,
                block_complete_flags,
                block_size,
            )
            mask = per_pos_total > 0
            if mask.any():
                stats["per_position_accuracy"] = (
                    (per_pos_correct[mask] / per_pos_total[mask])
                    .cpu()
                    .tolist()
                )
            else:
                stats["per_position_accuracy"] = []
            return output_ids, stats
        return output_ids
