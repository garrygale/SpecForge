"""Virtual Head Attention (arXiv:2610.17959) config resolution and folding.

Training-time reparameterization over the GQA draft attention: physical query
heads (``num_attention_heads // num_key_value_heads`` of them) are expanded
into ``num_key_value_heads * physical`` virtual heads by a per-KV-group
near-identity premix before q_norm/RoPE, and the attention output heads are
blended by a low-rank residual ``I + U V^T`` postmix before o_proj. Both
operators are linear, so a trained checkpoint folds back into a standard GQA
one (premix into q_proj, postmix into o_proj) that serves with zero VHA
awareness.
"""

import re
from typing import Optional

import torch

VHA_KEY = "vha"
VHA_FIELDS = frozenset({"physical_q_heads", "postmix_rank", "premix_alpha"})

PREMIX = "vha_premix"
POSTMIX_U = "vha_postmix_u"
POSTMIX_V = "vha_postmix_v"


def resolve_vha_config(config) -> Optional[dict]:
    """Resolve ``dflash_config.vha``; None when VHA is disabled."""

    method = dict(getattr(config, "dflash_config", None) or {})
    raw = method.get(VHA_KEY)
    if raw is None or raw is False:
        return None
    if not isinstance(raw, dict):
        raise ValueError(
            f"dflash_config.{VHA_KEY} must be a dict, got {type(raw).__name__}"
        )
    spec = dict(raw)
    unknown = sorted(set(spec) - VHA_FIELDS)
    if unknown:
        raise ValueError(
            f"unknown dflash_config.{VHA_KEY} entries {unknown}; expected a "
            f"subset of {sorted(VHA_FIELDS)}"
        )

    mode = str(method.get("attention_mode", "gqa")).lower()
    if mode != "gqa":
        raise ValueError(
            f"dflash_config.{VHA_KEY} requires attention_mode 'gqa', got {mode!r}"
        )

    physical = spec.get("physical_q_heads")
    if not isinstance(physical, int) or isinstance(physical, bool):
        raise ValueError(
            f"dflash_config.vha.physical_q_heads must be an integer, got {physical!r}"
        )
    num_heads = int(config.num_attention_heads)
    num_kv = int(config.num_key_value_heads)
    if physical < 1 or num_heads != physical * num_kv:
        raise ValueError(
            "dflash_config.vha needs num_attention_heads == physical_q_heads * "
            f"num_key_value_heads, got {num_heads} vs {physical} * {num_kv}"
        )

    rank = spec.get("postmix_rank")
    if rank is None:
        rank = num_heads // 4
    if (
        not isinstance(rank, int)
        or isinstance(rank, bool)
        or not 1 <= rank <= num_heads
    ):
        raise ValueError(
            f"dflash_config.vha.postmix_rank must be in [1, {num_heads}], got {rank!r}"
        )

    alpha = spec.get("premix_alpha", 0.1)
    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool) or alpha <= 0:
        raise ValueError(f"dflash_config.vha.premix_alpha must be positive, got {alpha!r}")

    return {
        "physical_q_heads": physical,
        "postmix_rank": rank,
        "premix_alpha": float(alpha),
    }


def init_vha_params(num_heads: int, num_kv: int, head_dim: int, vha: dict) -> dict:
    """Fresh VHA parameters: near-identity premix, zero-init postmix residual."""

    alpha = vha["premix_alpha"]
    premix = torch.eye(head_dim).unsqueeze(0) + torch.randn(
        num_kv, head_dim, head_dim
    ) * (alpha / head_dim**0.5)
    rank = vha["postmix_rank"]
    return {
        PREMIX: premix,
        POSTMIX_U: torch.randn(num_heads, rank) * 0.01,
        POSTMIX_V: torch.zeros(num_heads, rank),
    }


def apply_vha_premix(query: torch.Tensor, premix: torch.Tensor) -> torch.Tensor:
    """Expand physical query heads into group-major virtual heads.

    ``query`` is ``[b, s, physical, head_dim]`` and ``premix`` is
    ``[num_kv, head_dim, head_dim]``; virtual head ``v = group * physical + i``
    so the standard ``v // (num_heads // num_kv)`` GQA mapping applies.

    Formulation is einsum-free on purpose (einsum lowering measured slow on
    the dflare fusion work and again in scripts/bench_vha_ops.py): one dense
    GEMM per KV group on the flat [b*s*p, d] view, concatenated group-major.
    """

    bsz, q_len, physical, head_dim = query.shape
    mixed = torch.cat(
        [
            (query @ pm).reshape(bsz, q_len, physical * head_dim)
            for pm in premix.unbind(0)
        ],
        dim=2,
    )
    return mixed.reshape(bsz, q_len, physical * premix.shape[0], head_dim)


def apply_vha_postmix(attn_output: torch.Tensor, u: torch.Tensor, v: torch.Tensor):
    """Blend attention output heads by ``I + U V^T`` (rank-r residual).

    Fastest einsum-free form from scripts/bench_vha_ops.py: one shared
    ``M = U V^T`` GEMM plus a batched left-multiply over the head axis.
    """

    bsz, q_len, flat = attn_output.shape
    num_heads = u.shape[0]
    head_dim = flat // num_heads
    heads = attn_output.view(bsz, q_len, num_heads, head_dim)
    m = u @ v.t()
    delta = torch.bmm(
        m.t().expand(bsz * q_len, num_heads, num_heads),
        heads.reshape(bsz * q_len, num_heads, head_dim),
    ).view(bsz, q_len, num_heads, head_dim)
    return (heads + delta).reshape(bsz, q_len, flat)


def fold_q_proj(q_weight: torch.Tensor, q_bias, premix: torch.Tensor):
    """Fold premix into q_proj; virtual row block ``group * physical + i``."""

    num_kv, head_dim, head_dim2 = premix.shape
    if head_dim != head_dim2:
        raise ValueError(f"premix must be square per group, got {head_dim}x{head_dim2}")
    physical = q_weight.shape[0] // head_dim
    if q_weight.shape[0] != physical * head_dim:
        raise ValueError(f"q_proj width {q_weight.shape[0]} is not a head multiple")
    w = q_weight.view(physical, head_dim, -1)
    folded = torch.stack(
        [torch.matmul(pm.t().to(w.dtype), w) for pm in premix.unbind(0)]
    ).reshape(num_kv * physical * head_dim, w.shape[-1])
    folded_bias = None
    if q_bias is not None:
        b = q_bias.view(physical, head_dim)
        folded_bias = torch.stack(
            [torch.matmul(pm.t().to(b.dtype), b) for pm in premix.unbind(0)]
        ).reshape(num_kv * physical * head_dim)
    return folded, folded_bias


def fold_o_proj(o_weight: torch.Tensor, u: torch.Tensor, v: torch.Tensor):
    """Fold the postmix residual ``I + U V^T`` into o_proj."""

    hidden, flat = o_weight.shape
    num_heads, rank = u.shape
    if v.shape != u.shape or flat % num_heads:
        raise ValueError(
            f"postmix shape mismatch: u {tuple(u.shape)}, v {tuple(v.shape)}, "
            f"o_proj width {flat}"
        )
    head_dim = flat // num_heads
    m = u.to(o_weight.dtype) @ v.to(o_weight.dtype).t()
    mix = torch.eye(num_heads, dtype=m.dtype, device=m.device) + m
    w3 = o_weight.view(hidden, num_heads, head_dim)
    return torch.matmul(w3.permute(2, 0, 1), mix.t()).permute(1, 2, 0).reshape(
        hidden, flat
    )


def vha_layer_prefixes(state_dict: dict) -> list:
    return sorted(
        {
            match.group(1)
            for key in state_dict
            if (match := re.match(rf"(.*)\.{PREMIX}$", key))
        }
    )


def fold_vha_state_dict(state_dict: dict, layer_prefixes=None) -> dict:
    """Return a state dict with every layer's VHA modules folded away."""

    prefixes = vha_layer_prefixes(state_dict) if layer_prefixes is None else layer_prefixes
    if not prefixes:
        raise ValueError("no vha_premix keys found in the state dict")
    folded = dict(state_dict)
    for prefix in prefixes:
        premix = folded.pop(f"{prefix}.{PREMIX}", None)
        u = folded.pop(f"{prefix}.{POSTMIX_U}", None)
        v = folded.pop(f"{prefix}.{POSTMIX_V}", None)
        if premix is None or u is None or v is None:
            raise ValueError(f"incomplete VHA parameters under {prefix!r}")
        q_key = f"{prefix}.q_proj.weight"
        b_key = f"{prefix}.q_proj.bias"
        q_bias = folded.get(b_key)
        folded[q_key], folded_bias = fold_q_proj(folded[q_key], q_bias, premix)
        if q_bias is not None:
            folded[b_key] = folded_bias
        folded[f"{prefix}.o_proj.weight"] = fold_o_proj(
            folded[f"{prefix}.o_proj.weight"], u, v
        )
    return folded
