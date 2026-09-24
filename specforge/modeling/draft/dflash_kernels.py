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
# Shared gate/up projections (``dflash_config.ffn_sharing``)
# ---------------------------------------------------------------------------
#
# ``mode='lattice'`` (the default): ``gate_proj``/``up_proj`` shrink to
# ``gate_groups``/``up_groups`` unique channels and intermediate channel ``j``
# pairs them through precomputed index maps:
#
#     h_j = act( gate[gate_idx[j]] ) * up[up_idx[j]]
#
#     nested:  gate_idx[j] = j // (M / G_g),   up_idx[j] = j // (M / G_u)
#     outer:   gate_idx[j] = j % G_g,           up_idx[j] = (j // G_g) % G_u
#
# ``down_proj`` stays dense.  TP > 1 is rejected (the gather reads channels
# owned by other ranks).
#
# ``mode='routed_outer'``: ``experts`` independent outer lattices (G_g x G_u
# each), blended by a router before one shared dense ``down_proj``:
#
#     H(x) = sum_e w_e(x) * act(W_g^e x) (W_u^e x)^T        rank <= experts
#     y    = W_down vec(H)
#
# Each expert has its OWN up view — the multi-value-view break the single
# lattice cannot express.  The router rides the fused ``gate_up_proj`` GEMM
# (zero-initialized: training starts at the uniform expert average) and its
# granularity is a config knob: ``expert`` (one softmax scalar per expert) or
# ``gate_slot`` (a per-gate-slot blend across experts — mixing stays on the
# nonlinear side, per the fold-era lesson).  ``experts=1`` is exactly the
# plain outer lattice.  The low-rank structure is TRAINED IN, sidestepping
# the post-hoc CP decomposability that measured absent.
#
# The folded readout (``dflash_config.ffn_readout``) is RETIRED: the static
# convex mixture measured equivalent to shrinking the lattice (fold-2 ==
# half lattice).  Old checkpoints live under the ``outer_ffn_fold`` tag.

FOLDED_READOUT_KEY = "ffn_readout"  # retired; kept only for the guard below

SHARING_KEY = "ffn_sharing"
SHARING_MODES = frozenset({"lattice", "routed_outer"})
SHARING_PAIRINGS = frozenset({"nested", "outer"})
ROUTER_GRANULARITIES = frozenset({"expert", "gate_slot"})


def _divisors(value: int, limit: int = 1024) -> list:
    """Small divisors of ``value``, used only for actionable error messages."""

    return [d for d in range(2, min(value, limit) + 1) if value % d == 0]


def _reject_retired_fold(config) -> None:
    """Refuse retired ``ffn_readout`` entries with an actionable message."""

    raw = (getattr(config, "dflash_config", None) or {}).get(FOLDED_READOUT_KEY)
    if raw is None or raw is False:
        return
    if isinstance(raw, str) and raw in {"dense", "off", "none"}:
        return
    if isinstance(raw, dict) and raw.get("mode") in {"dense", "off", "none"}:
        return
    raise ValueError(
        "dflash_config.ffn_readout is retired: the folded readout measured "
        "equivalent to shrinking the gate lattice (the static mixture is "
        "inert). Drop the key, or serve old checkpoints from the "
        "outer_ffn_fold tag."
    )


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

    Returns ``None`` for the plain ``Qwen3MLP`` (the default).  ``mode``
    defaults to ``"lattice"`` and returns ``{"mode", "pairing", "gate_groups",
    "up_groups"}``; ``"routed_outer"`` returns ``{"mode", "experts",
    "gate_groups", "up_groups", "router"}`` with ``intermediate_size`` equal
    to ``experts * gate_groups * up_groups`` (the virtual channel count).
    """

    dflash_config = getattr(config, "dflash_config", None) or {}
    raw = dflash_config.get(SHARING_KEY)
    if raw is None or raw is False:
        return None
    if not isinstance(raw, dict):
        raise ValueError(
            f"dflash_config.{SHARING_KEY} must be a dict, got "
            f"{type(raw).__name__}"
        )
    spec = dict(raw)
    mode = spec.pop("mode", "lattice")
    if mode not in SHARING_MODES:
        raise ValueError(
            f"unknown dflash_config.{SHARING_KEY} mode {mode!r}; expected "
            f"one of {sorted(SHARING_MODES)}"
        )

    if mode == "routed_outer":
        unknown = sorted(set(spec) - {"experts", "gate_groups", "up_groups", "router"})
        if unknown:
            raise ValueError(
                f"unknown dflash_config.{SHARING_KEY} entries {unknown}; "
                "expected 'experts', 'gate_groups', 'up_groups' and 'router'"
            )
        router = spec.get("router", "expert")
        if router not in ROUTER_GRANULARITIES:
            raise ValueError(
                f"dflash_config.{SHARING_KEY}.router={router!r} must be one "
                f"of {sorted(ROUTER_GRANULARITIES)}"
            )
        experts = int(spec.get("experts", 1))
        gate_groups = int(spec.get("gate_groups", 0) or 0)
        up_groups = int(spec.get("up_groups", 0) or 0)
        if min(experts, gate_groups, up_groups) < 1:
            raise ValueError(
                f"routed_outer needs experts/gate_groups/up_groups >= 1, got "
                f"{experts}/{gate_groups}/{up_groups}"
            )
        if experts == 1 and router == "gate_slot":
            raise ValueError(
                "router='gate_slot' needs experts >= 2 (with one expert the "
                "blend is identically 1)"
            )
        intermediate_size = int(getattr(config, "intermediate_size", 0) or 0)
        if intermediate_size != experts * gate_groups * up_groups:
            raise ValueError(
                f"routed_outer needs intermediate_size == experts * "
                f"gate_groups * up_groups, got {intermediate_size} vs "
                f"{experts} * {gate_groups} * {up_groups}"
            )
        return {
            "mode": "routed_outer",
            "experts": experts,
            "gate_groups": gate_groups,
            "up_groups": up_groups,
            "router": router,
        }

    unknown = sorted(set(spec) - {"pairing", "gate_groups", "up_groups"})
    if unknown:
        raise ValueError(
            f"unknown dflash_config.{SHARING_KEY} entries {unknown}; expected "
            "'mode', 'pairing', 'gate_groups' and 'up_groups'"
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
            f"{intermediate_size}; pick a factorization of "
            f"{intermediate_size} (most balanced: "
            f"{', '.join(f'{d}x{q}' for d, q in pairs)}) or change "
            f"intermediate_size to {gate_groups * up_groups}"
        )
    return {
        "mode": "lattice",
        "pairing": pairing,
        "gate_groups": gate_groups,
        "up_groups": up_groups,
    }


class SharedGLUMLP(nn.Module):
    """SwiGLU MLP whose gate/up projections serve multiple channels.

    ``gate_proj`` outputs ``gate_groups`` channels and ``up_proj`` outputs
    ``up_groups`` channels; intermediate channel ``j`` pairs them through the
    precomputed ``gate_idx``/``up_idx`` maps (see the module note above), and
    ``down_proj`` stays dense.  The index maps live as non-persistent
    buffers: derived from the config, invisible to checkpoints and untouched
    by the QAT/NPU linear walkers.

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
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
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


class RoutedOuterMLP(nn.Module):
    """``experts`` independent outer-product FFNs blended by a router.

    Every expert holds its own ``gate_groups`` gate channels and
    ``up_groups`` up channels, fused expert-major into one ``gate_up_proj``
    linear whose trailing ``router_width`` rows produce the routing logits
    (zero-initialized: training starts at the uniform expert average).  The
    per-token computation is

        H(x) = sum_e w_e(x) * act(W_g^e x) (W_u^e x)^T     rank <= experts
        y    = down_proj(vec(H))

    ``router='expert'`` blends with one softmax scalar per expert;
    ``router='gate_slot'`` gives every gate slot its own per-expert blend
    (the mixture stays on the gate/nonlinear side, per the fold-era lesson).
    ``experts=1`` degenerates to the plain outer lattice with a dense
    ``down_proj``, so the E-sweep carries its own baseline.
    """

    def __init__(
        self,
        config: Qwen3Config,
        *,
        experts: int,
        gate_groups: int,
        up_groups: int,
        router: str = "expert",
    ) -> None:
        super().__init__()
        experts = int(experts)
        gate_groups = int(gate_groups)
        up_groups = int(up_groups)
        if min(experts, gate_groups, up_groups) < 1:
            raise ValueError(
                f"routed_outer needs experts/gate_groups/up_groups >= 1, got "
                f"{experts}/{gate_groups}/{up_groups}"
            )
        if router not in ROUTER_GRANULARITIES:
            raise ValueError(
                f"router must be one of {sorted(ROUTER_GRANULARITIES)}, got "
                f"{router!r}"
            )
        if experts == 1 and router == "gate_slot":
            raise ValueError(
                "router='gate_slot' needs experts >= 2 (with one expert the "
                "blend is identically 1)"
            )
        intermediate_size = int(config.intermediate_size)
        if intermediate_size != experts * gate_groups * up_groups:
            raise ValueError(
                f"routed_outer needs intermediate_size == experts * "
                f"gate_groups * up_groups, got {intermediate_size} vs "
                f"{experts} * {gate_groups} * {up_groups}"
            )

        self.config = config
        self.hidden_size = int(config.hidden_size)
        self.intermediate_size = intermediate_size
        self.experts = experts
        self.gate_groups = gate_groups
        self.up_groups = up_groups
        self.router = router
        self.router_width = experts if router == "expert" else experts * gate_groups

        self.gate_up_proj = nn.Linear(
            self.hidden_size,
            experts * (gate_groups + up_groups) + self.router_width,
            bias=False,
        )
        with torch.no_grad():
            # Zero-init router rows: softmax(0) is the uniform blend.
            self.gate_up_proj.weight[experts * (gate_groups + up_groups):].zero_()
        self.down_proj = nn.Linear(
            gate_groups * up_groups, self.hidden_size, bias=False
        )
        self.act_fn = ACT2FN[getattr(config, "hidden_act", "silu")]

    def combine(self, fused: torch.Tensor) -> torch.Tensor:
        """Post-GEMM computation: slice experts + router, blend outer products.

        Split out so the Ascend fused norm+quant path can reuse the exact
        post-GEMM computation over the fused projection output.
        """

        lead = fused.shape[:-1]
        E, G, U = self.experts, self.gate_groups, self.up_groups
        feats = fused[..., : E * (G + U)].reshape(*lead, E, G + U)
        gate = self.act_fn(feats[..., :G])
        up = feats[..., G:]
        route = fused[..., E * (G + U):]
        if self.router == "expert":
            gate = gate * torch.softmax(route, dim=-1).unsqueeze(-1)
        else:
            delta = torch.softmax(route.reshape(*lead, G, E), dim=-1)
            gate = gate * delta.transpose(-1, -2)
        hidden = (gate.unsqueeze(-1) * up.unsqueeze(-2)).sum(dim=-3)
        return self.down_proj(hidden.reshape(*lead, G * U))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.combine(self.gate_up_proj(x))


def _make_qwen3_rms_norm(hidden_size: int, eps: float) -> nn.Module:
    return Qwen3RMSNorm(hidden_size, eps=eps)


def _make_qwen3_mlp(config: Qwen3Config) -> nn.Module:
    _reject_retired_fold(config)
    sharing = resolve_ffn_sharing(config)
    if sharing is None:
        return Qwen3MLP(config)
    spec = dict(sharing)
    spec.pop("mode")
    if sharing["mode"] == "routed_outer":
        return RoutedOuterMLP(config, **spec)
    return SharedGLUMLP(config, **spec)


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
        if resolve_ffn_sharing(config) is None:
            return LigerSwiGLUMLP(config)
        # Liger has no fused kernel for the shared or routed projections, so
        # keep the knob working by falling back to the portable PyTorch MLP.
        return _make_qwen3_mlp(config)

    return DFlashKernels(
        make_rms_norm=make_rms_norm,
        make_mlp=make_mlp,
    )
