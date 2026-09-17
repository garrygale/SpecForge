"""Module factories used by the DFlash draft backbone."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config, Qwen3MLP, Qwen3RMSNorm

try:  # ACT2FN ships with the Qwen3 modeling module on supported releases
    from transformers.models.qwen3.modeling_qwen3 import ACT2FN
except ImportError:  # pragma: no cover - relocated activations fallback
    from transformers.activations import ACT2FN


@dataclass(frozen=True)
class DFlashKernels:
    """Stable construction boundary between DFlash and kernel providers."""

    make_rms_norm: Callable[[int, float], nn.Module]
    make_mlp: Callable[[Qwen3Config], nn.Module]


# ---------------------------------------------------------------------------
# Folded softmax FFN readout (opt-in through ``dflash_config.ffn_readout``)
# ---------------------------------------------------------------------------
#
# The dense ``down_proj`` (intermediate_size -> hidden_size) becomes
#
#     y[q] = sum_j softmax_j(logits[j, q % K]) * h[j * s + q]      s = M / c
#     out  = W y,          W: hidden_size x s
#
# i.e. the gated hidden is cut into ``c`` equal contiguous chunks of width
# ``s``; every offset ``q`` is a *normalized* (convex) mixture over the ``c``
# chunks at the same offset, and the mixture pattern only depends on ``q % K``,
# so one shared pattern serves all ``s / K`` repetitions of a chunk.  Per token
# the mixture costs ``c * s = M`` macs and the dense readout ``hidden_size * s``
# (instead of ``hidden_size * M``), and the readout holds
# ``hidden_size * s + c * K`` parameters instead of ``hidden_size * M``.

FOLDED_READOUT_KEY = "ffn_readout"
FOLDED_READOUT_MODE = "folded_softmax"
FOLDED_READOUT_MODES = frozenset({"folded", "folded_softmax", "softmax_fold"})
DEFAULT_FOLDED_GRANULARITY = 16


def _divisors(value: int, limit: int = 1024) -> list:
    """Small divisors of ``value``, used only for actionable error messages."""

    return [d for d in range(2, min(value, limit) + 1) if value % d == 0]


def _nearest_divisor(value: int, target: int) -> Optional[int]:
    divisors = _divisors(value)
    if not divisors:
        return None
    return min(divisors, key=lambda d: (abs(d - target), d))


def resolve_folded_readout(config) -> Optional[Dict[str, int]]:
    """Resolve ``dflash_config.ffn_readout`` for one draft config.

    Returns ``{"branches": c, "granularity": K}`` when the folded readout is
    enabled and ``None`` for the plain dense ``down_proj`` (the default).

    ``c`` defaults to the fold ratio ``intermediate_size / hidden_size`` rounded
    to the nearest divisor of ``intermediate_size``: ``c = 3`` reproduces the
    ``3N -> N -> N`` case, while the 35B-A3B draft (9728 / 2560 = 3.8) resolves
    to ``c = 4`` and a folded width of 2432.  ``K`` defaults to
    ``DEFAULT_FOLDED_GRANULARITY`` and must divide ``intermediate_size / c``.
    """

    dflash_config = getattr(config, "dflash_config", None) or {}
    raw = dflash_config.get(FOLDED_READOUT_KEY)
    if raw is None or raw is False:
        return None
    if isinstance(raw, str):
        mode, spec = raw, {}
    elif isinstance(raw, dict):
        spec = dict(raw)
        mode = spec.pop("mode", FOLDED_READOUT_MODE)
        unknown = sorted(set(spec) - {"branches", "granularity"})
        if unknown:
            raise ValueError(
                f"unknown dflash_config.{FOLDED_READOUT_KEY} entries {unknown}; "
                "expected 'mode', 'branches' and 'granularity'"
            )
    else:
        raise ValueError(
            f"dflash_config.{FOLDED_READOUT_KEY} must be a mode string or a "
            f"dict, got {type(raw).__name__}"
        )
    if mode in {"dense", "off", "none"}:
        return None
    if mode not in FOLDED_READOUT_MODES:
        raise ValueError(
            f"unknown dflash_config.{FOLDED_READOUT_KEY} mode {mode!r}; expected "
            f"one of {sorted(FOLDED_READOUT_MODES)} or 'dense'"
        )

    hidden_size = int(getattr(config, "hidden_size", 0) or 0)
    intermediate_size = int(getattr(config, "intermediate_size", 0) or 0)
    if hidden_size <= 0 or intermediate_size <= 0:
        raise ValueError(
            "folded readout needs positive hidden_size and intermediate_size"
        )

    if "branches" in spec:
        branches = int(spec["branches"])
    else:
        ratio = max(2, int(round(intermediate_size / hidden_size)))
        branches = _nearest_divisor(intermediate_size, ratio) or 1
    if branches < 1:
        raise ValueError(f"folded readout branches must be >= 1, got {branches}")
    if intermediate_size % branches:
        raise ValueError(
            f"folded readout branches={branches} must divide "
            f"intermediate_size={intermediate_size}; nearby divisors are "
            f"{_divisors(intermediate_size)[:12]}"
        )

    folded_size = intermediate_size // branches
    granularity = int(spec.get("granularity", DEFAULT_FOLDED_GRANULARITY))
    if granularity < 1 or folded_size % granularity:
        raise ValueError(
            f"folded readout granularity={granularity} must divide the folded "
            f"width intermediate_size / branches = {intermediate_size} / "
            f"{branches} = {folded_size}; nearby divisors are "
            f"{_divisors(folded_size)[:12]}"
        )
    return {"branches": branches, "granularity": granularity}


class FoldedSoftmaxReadout(nn.Module):
    """Normalized folded mixture followed by a dense ``folded -> hidden`` map.

    The mixture itself is weight-free apart from ``branches * granularity``
    logits and stays in the activation dtype — it is a vector product plus a
    sum.  Only ``self.proj`` is a plain ``nn.Linear``, so QAT
    (``replace_linear_with_quantized``) quantizes it exactly like the dense
    ``down_proj`` it replaces while the mixture stays unquantized.

    A baseline checkpoint that still holds a dense
    ``[hidden_size, intermediate_size]`` ``down_proj.weight`` is folded on
    load: the dense matrix is summed over its ``branches`` chunks, which is the
    least-squares projection of that dense readout onto this family for a
    uniform mixture, and the mixture logits stay at their uniform init.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        branches: int = 4,
        granularity: int = DEFAULT_FOLDED_GRANULARITY,
        bias: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        branches = int(branches)
        granularity = int(granularity)
        if branches < 1:
            raise ValueError(f"branches must be >= 1, got {branches}")
        if intermediate_size % branches:
            raise ValueError(
                f"branches={branches} must divide intermediate_size="
                f"{intermediate_size}"
            )
        folded_size = intermediate_size // branches
        if granularity < 1 or folded_size % granularity:
            raise ValueError(
                f"granularity={granularity} must divide the folded width "
                f"intermediate_size / branches = {folded_size}"
            )

        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.branches = branches
        self.granularity = granularity
        self.folded_size = folded_size
        self.repeats = folded_size // granularity

        factory_kwargs = {"device": device, "dtype": dtype}
        self.proj = nn.Linear(folded_size, hidden_size, bias=bias, **factory_kwargs)
        self.fold_logits = nn.Parameter(
            torch.zeros(branches, granularity, **factory_kwargs)
        )

    def reset_parameters(self) -> None:
        # nn.Linear's own init keeps a fresh folded readout comparable to a
        # dense one; the mixture starts uniform (a plain chunk average).
        self.proj.reset_parameters()
        with torch.no_grad():
            self.fold_logits.zero_()

    def fold_weights(self) -> torch.Tensor:
        """Current convex mixture, shape ``[branches, granularity]``.

        The softmax runs in the logits' own dtype, mirroring the flare fusion
        weights (``torch.softmax(self.layer_fusion_weights, dim=1)``).  The
        logits ride the module dtype, so the mixture already matches the
        activations it multiplies — no fp32 round-trip, no cast back.
        """

        return torch.softmax(self.fold_logits, dim=0)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        lead = hidden_states.shape[:-1]
        chunks = hidden_states.reshape(
            *lead, self.branches, self.repeats, self.granularity
        )
        weights = self.fold_weights().reshape(
            *([1] * len(lead)), self.branches, 1, self.granularity
        )
        mixed = (chunks * weights).sum(dim=-3)
        mixed = mixed.reshape(*lead, self.folded_size)
        return self.proj(mixed)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        legacy_key = prefix + "weight"
        proj_key = prefix + "proj.weight"
        logits_key = prefix + "fold_logits"
        legacy_weight = state_dict.get(legacy_key)
        legacy_dense = (
            torch.is_tensor(legacy_weight)
            and legacy_weight.dim() == 2
            and legacy_weight.shape[-1] == self.intermediate_size
        )
        if legacy_dense:
            # Baseline dense down_proj: project it onto the folded family under
            # a uniform mixture (least squares) and keep the logits uniform.
            state_dict.pop(legacy_key, None)
            state_dict[proj_key] = legacy_weight.reshape(
                legacy_weight.shape[0], self.branches, self.folded_size
            ).sum(dim=1)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        if legacy_dense and logits_key in missing_keys:
            # Absent logits on a legacy checkpoint are intentional, not a gap
            # the caller has to fill in.
            missing_keys.remove(logits_key)


class FoldedSoftmaxMLP(nn.Module):
    """``Qwen3MLP``-compatible SwiGLU MLP with a folded softmax readout.

    ``gate_proj``/``up_proj``/``act_fn`` are unchanged, ``down_proj`` is a
    :class:`FoldedSoftmaxReadout`, so a baseline checkpoint warm-starts into
    this module and only the readout is re-parameterized.
    """

    def __init__(
        self,
        config: Qwen3Config,
        *,
        branches: int,
        granularity: int = DEFAULT_FOLDED_GRANULARITY,
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = FoldedSoftmaxReadout(
            self.hidden_size,
            self.intermediate_size,
            branches=branches,
            granularity=granularity,
        )
        self.act_fn = ACT2FN[getattr(config, "hidden_act", "silu")]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Shared gate/up projections (opt-in through ``dflash_config.ffn_sharing``)
# ---------------------------------------------------------------------------
#
# The SwiGLU channel ``j`` becomes
#
#     h_j = act( gate[gate_idx[j]] ) * up[up_idx[j]]      j < intermediate_size
#
# where ``gate`` is the ``G_g``-wide ``gate_proj`` output and ``up`` the
# ``G_u``-wide ``up_proj`` output, so the two projections hold
# ``hidden * (G_g + G_u)`` weights instead of ``2 * hidden * M`` while
# ``down_proj`` keeps its ``hidden * M`` shape.  The index maps pick the
# pairing structure:
#
#     nested:  gate_idx[j] = j // (M / G_g),   up_idx[j] = j // (M / G_u)
#              contiguous block grouping — pure gate sharing (``G_u = M``),
#              pure up sharing (``G_g = M``) and hierarchical mixes.
#     outer:   gate_idx[j] = j % G_g,           up_idx[j] = (j // G_g) % G_u
#              the full ``G_g x G_u`` lattice spread over the ``M`` down
#              channels; each (gate, up) combo repeats ``M / (G_g * G_u)``
#              times with distinct down columns (``G_g * G_u = M`` is the
#              exactly-once lattice).
#
# A baseline dense checkpoint warm-starts by scatter-averaging its ``[M, hidden]``
# gate/up rows per shared group — the uniform-mixture least-squares analogue
# of the folded readout's chunk sum.

SHARING_KEY = "ffn_sharing"
SHARING_PAIRINGS = frozenset({"nested", "outer"})


def _sharing_indices(
    intermediate_size: int, gate_groups: int, up_groups: int, pairing: str
) -> tuple:
    """Per-channel (gate, up) index maps for one pairing structure."""

    channel = torch.arange(intermediate_size)
    if pairing == "nested":
        gate_idx = channel // (intermediate_size // gate_groups)
        up_idx = channel // (intermediate_size // up_groups)
    else:  # outer
        gate_idx = channel % gate_groups
        up_idx = (channel // gate_groups) % up_groups
    return gate_idx, up_idx


def resolve_ffn_sharing(config) -> Optional[Dict[str, object]]:
    """Resolve ``dflash_config.ffn_sharing`` for one draft config.

    Returns ``{"pairing": p, "gate_groups": G_g, "up_groups": G_u}`` when
    sharing is enabled and ``None`` for the plain ``Qwen3MLP`` (the default).
    ``gate_groups``/``up_groups`` default to ``intermediate_size``, i.e. no
    sharing on that side, so ``{"gate_groups": 4864}`` alone is pure gate
    sharing with ``k = 2`` on the 35B-A3B draft (9728 = 512 * 19 has no
    factor 3).  ``pairing='outer'`` needs ``G_g * G_u`` to divide
    ``intermediate_size``: 76 x 128 keeps 9728 exactly, while a 64 x 64
    lattice requires switching ``intermediate_size`` to 4096.
    """

    dflash_config = getattr(config, "dflash_config", None) or {}
    raw = dflash_config.get(SHARING_KEY)
    if raw is None or raw is False:
        return None
    if not isinstance(raw, dict):
        raise ValueError(
            f"dflash_config.{SHARING_KEY} must be a dict with 'pairing', "
            f"'gate_groups' and 'up_groups', got {type(raw).__name__}"
        )
    spec = dict(raw)
    unknown = sorted(set(spec) - {"pairing", "gate_groups", "up_groups"})
    if unknown:
        raise ValueError(
            f"unknown dflash_config.{SHARING_KEY} entries {unknown}; expected "
            "'pairing', 'gate_groups' and 'up_groups'"
        )

    pairing = spec.get("pairing", "nested")
    if pairing not in SHARING_PAIRINGS:
        raise ValueError(
            f"unknown dflash_config.{SHARING_KEY} pairing {pairing!r}; expected "
            f"one of {sorted(SHARING_PAIRINGS)}"
        )

    intermediate_size = int(getattr(config, "intermediate_size", 0) or 0)
    if intermediate_size <= 0:
        raise ValueError("ffn sharing needs a positive intermediate_size")
    gate_groups = int(spec.get("gate_groups", intermediate_size))
    up_groups = int(spec.get("up_groups", intermediate_size))
    for name, groups in (("gate_groups", gate_groups), ("up_groups", up_groups)):
        if groups < 1 or groups > intermediate_size:
            raise ValueError(
                f"dflash_config.{SHARING_KEY}.{name}={groups} must be between 1 "
                f"and intermediate_size={intermediate_size}"
            )

    if pairing == "nested":
        for name, groups in (("gate_groups", gate_groups), ("up_groups", up_groups)):
            if intermediate_size % groups:
                raise ValueError(
                    f"nested sharing {name}={groups} must divide "
                    f"intermediate_size={intermediate_size}; nearby divisors "
                    f"are {_divisors(intermediate_size)[:12]}"
                )
    elif intermediate_size % (gate_groups * up_groups):
        pairs = sorted(
            (
                (d, intermediate_size // d)
                for d in _divisors(intermediate_size)
                if d <= intermediate_size // d
            ),
            key=lambda pair: abs(pair[0] - pair[1]),
        )[:6]
        raise ValueError(
            f"outer sharing needs gate_groups * up_groups to divide "
            f"intermediate_size, got {gate_groups} * {up_groups} = "
            f"{gate_groups * up_groups} with intermediate_size="
            f"{intermediate_size}; either pick a factorization of "
            f"{intermediate_size} (most balanced: "
            f"{', '.join(f'{d}x{q}' for d, q in pairs)}) or change "
            f"intermediate_size to {gate_groups * up_groups}"
        )
    return {"pairing": pairing, "gate_groups": gate_groups, "up_groups": up_groups}


class SharedGLUMLP(nn.Module):
    """SwiGLU MLP whose gate/up projections serve multiple channels.

    ``gate_proj`` outputs ``gate_groups`` channels and ``up_proj`` outputs
    ``up_groups`` channels; intermediate channel ``j`` pairs them through the
    precomputed ``gate_idx``/``up_idx`` maps (see the module note above).
    ``down_proj`` is either a plain ``nn.Linear`` or a
    :class:`FoldedSoftmaxReadout` when ``ffn_readout`` is also enabled.  The
    index maps live as non-persistent buffers: derived from the config,
    invisible to checkpoints and untouched by the QAT/NPU linear walkers.

    A baseline checkpoint holding dense ``[intermediate_size, hidden]``
    ``gate_proj``/``up_proj`` weights warm-starts by scatter-averaging the
    legacy rows per shared group.
    """

    def __init__(
        self,
        config: Qwen3Config,
        *,
        pairing: str = "nested",
        gate_groups: int,
        up_groups: int,
        folded: Optional[Dict[str, int]] = None,
    ) -> None:
        super().__init__()
        gate_groups = int(gate_groups)
        up_groups = int(up_groups)
        if pairing not in SHARING_PAIRINGS:
            raise ValueError(
                f"pairing must be one of {sorted(SHARING_PAIRINGS)}, got "
                f"{pairing!r}"
            )
        intermediate_size = int(config.intermediate_size)
        hidden_size = int(config.hidden_size)
        if not 1 <= gate_groups <= intermediate_size:
            raise ValueError(
                f"gate_groups={gate_groups} must be between 1 and "
                f"intermediate_size={intermediate_size}"
            )
        if not 1 <= up_groups <= intermediate_size:
            raise ValueError(
                f"up_groups={up_groups} must be between 1 and "
                f"intermediate_size={intermediate_size}"
            )
        if pairing == "nested":
            if intermediate_size % gate_groups or intermediate_size % up_groups:
                raise ValueError(
                    f"nested sharing needs gate_groups={gate_groups} and "
                    f"up_groups={up_groups} to divide intermediate_size="
                    f"{intermediate_size}"
                )
        elif intermediate_size % (gate_groups * up_groups):
            raise ValueError(
                f"outer sharing needs gate_groups * up_groups = "
                f"{gate_groups * up_groups} to divide intermediate_size="
                f"{intermediate_size}"
            )

        self.config = config
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.pairing = pairing
        self.gate_groups = gate_groups
        self.up_groups = up_groups

        self.gate_proj = nn.Linear(hidden_size, gate_groups, bias=False)
        self.up_proj = nn.Linear(hidden_size, up_groups, bias=False)
        if folded is None:
            self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        else:
            self.down_proj = FoldedSoftmaxReadout(
                hidden_size, intermediate_size, **folded
            )
        gate_idx, up_idx = _sharing_indices(
            intermediate_size, gate_groups, up_groups, pairing
        )
        self.register_buffer("gate_idx", gate_idx, persistent=False)
        self.register_buffer("up_idx", up_idx, persistent=False)
        self.act_fn = ACT2FN[getattr(config, "hidden_act", "silu")]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        hidden = self.act_fn(gate[..., self.gate_idx]) * up[..., self.up_idx]
        return self.down_proj(hidden)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        for proj_name, groups, idx in (
            ("gate_proj", self.gate_groups, self.gate_idx),
            ("up_proj", self.up_groups, self.up_idx),
        ):
            key = prefix + proj_name + ".weight"
            legacy = state_dict.get(key)
            if not (torch.is_tensor(legacy) and legacy.dim() == 2):
                continue
            if legacy.shape[0] == self.intermediate_size != groups:
                # Baseline dense projection: scatter-average the rows per
                # shared group (in fp32, then back to the checkpoint dtype).
                index = idx.to(legacy.device)
                sums = torch.zeros(
                    groups, legacy.shape[1], dtype=torch.float32,
                    device=legacy.device,
                )
                sums.index_add_(0, index, legacy.float())
                counts = torch.bincount(index, minlength=groups).clamp(min=1)
                state_dict[key] = (sums / counts.unsqueeze(1)).to(legacy.dtype)
            elif legacy.shape[0] != groups:
                error_msgs.append(
                    f"size mismatch for {key}: ffn sharing cannot fold a legacy "
                    f"{tuple(legacy.shape)} projection; expected a dense "
                    f"[{self.intermediate_size}, {self.hidden_size}] checkpoint "
                    f"(train from scratch or re-export the baseline) or a "
                    f"sharing checkpoint with {groups} rows"
                )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


def _make_qwen3_rms_norm(hidden_size: int, eps: float) -> nn.Module:
    return Qwen3RMSNorm(hidden_size, eps=eps)


def _make_qwen3_mlp(config: Qwen3Config) -> nn.Module:
    folded = resolve_folded_readout(config)
    sharing = resolve_ffn_sharing(config)
    if sharing is None:
        if folded is None:
            return Qwen3MLP(config)
        return FoldedSoftmaxMLP(config, **folded)
    return SharedGLUMLP(config, folded=folded, **sharing)


DEFAULT_DFLASH_KERNELS = DFlashKernels(
    make_rms_norm=_make_qwen3_rms_norm,
    make_mlp=_make_qwen3_mlp,
)


def load_liger_dflash_kernels() -> DFlashKernels:
    """Load Liger lazily and adapt its constructors to the DFlash boundary."""

    try:
        from liger_kernel.transformers import LigerRMSNorm, LigerSwiGLUMLP
    except ModuleNotFoundError as exc:
        if exc.name in {"liger_kernel", "liger_kernel.transformers"}:
            raise ImportError(
                "model.use_liger_kernel=true requires the optional "
                "`specforge[liger]` extra. Install it with "
                '`pip install "specforge[liger]"`.'
            ) from exc
        raise

    def make_rms_norm(hidden_size: int, eps: float) -> nn.Module:
        return LigerRMSNorm(hidden_size, eps=eps)

    def make_mlp(config: Qwen3Config) -> nn.Module:
        if (
            resolve_folded_readout(config) is None
            and resolve_ffn_sharing(config) is None
        ):
            return LigerSwiGLUMLP(config)
        # Liger has no fused kernel for the folded readout or the shared
        # projections, so keep the knobs working by falling back to the
        # portable PyTorch MLP.
        return _make_qwen3_mlp(config)

    return DFlashKernels(
        make_rms_norm=make_rms_norm,
        make_mlp=make_mlp,
    )
