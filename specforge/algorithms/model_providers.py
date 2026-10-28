# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Algorithm-owned model and configuration providers.

The public assembler owns topology and lifecycle.  This module owns the facts
that vary by draft method: how to create its config and model, which target
features it captures, and which dataset policies it needs.  All heavy imports
remain inside callables so resolving the algorithm registry stays import-light.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from specforge.config import Config
    from specforge.modeling.draft.dflash_kernels import DFlashKernels


@dataclass
class AlgorithmModelParts:
    """Algorithm-specific pieces returned to the application assembler."""

    model: Any
    target_head: Any = None
    capture_layers: Optional[List[int]] = None


def dflash_needs_input_tools(cfg: Config, draft_model: Any) -> bool:
    method_config = getattr(draft_model.config, "dflash_config", None) or {}
    needs_mask_fallback = (
        cfg.model.mask_token_id is None and method_config.get("mask_token_id") is None
    )
    return cfg.mode == "online" or needs_mask_fallback


def _torch_dtype(cfg: Config):
    import torch

    return getattr(torch, cfg.model.torch_dtype)


def _device():
    from specforge.utils import get_local_device

    return get_local_device()


def _warm_start(
    cfg: Config,
    draft_model: Any,
    draft_config: Any,
    *,
    allow_missing_embedding: bool = False,
) -> None:
    if not cfg.model.draft_checkpoint_path:
        return
    from specforge.training.model_loading import warm_start_draft_model

    warm_start_draft_model(
        draft_model,
        cfg.model.draft_checkpoint_path,
        draft_config=draft_config,
        strategy=cfg.training.strategy,
        allow_missing_embedding=allow_missing_embedding,
        cache_dir=cfg.model.cache_dir,
        trust_remote_code=cfg.model.trust_remote_code,
    )


def _load_vocab_mapping(cfg: Config, draft_model: Any) -> None:
    if cfg.model.vocab_mapping_path:
        draft_model.load_vocab_mapping(cfg.model.vocab_mapping_path)


def _apply_draft_qat(draft_model: Any) -> None:
    """Apply the verifiedBase two-pass QAT replacement before FSDP wrapping."""
    from specforge.layers.wxay import replace_linear_with_quantized

    method_config = getattr(draft_model.config, "dflash_config", None) or {}
    qat_w_bit = method_config.get("qat_w_bit")
    if qat_w_bit is None:
        return
    qat_a_bit = method_config.get("qat_a_bit")
    qat_exclude = list(method_config.get("qat_exclude", []) or [])
    channel_balanced = bool(method_config.get("channel_balanced", False))
    channel_balanced_w4a4_only = bool(
        method_config.get("channel_balanced_w4a4_only", False)
    )
    w4a4_layers = set(method_config.get("qat_w4a4_layers", []) or [])
    stochastic_weight = bool(method_config.get("stochastic_weight", False))

    replace_linear_with_quantized(
        draft_model,
        w_bit=int(qat_w_bit),
        a_bit=int(qat_a_bit) if qat_a_bit is not None else None,
        channel_balanced=channel_balanced and not channel_balanced_w4a4_only,
        exclude_names=qat_exclude + list(w4a4_layers),
        stochastic_weight=stochastic_weight,
    )
    if w4a4_layers:
        replace_linear_with_quantized(
            draft_model,
            w_bit=4,
            a_bit=4,
            channel_balanced=channel_balanced,
            include_only=w4a4_layers,
            stochastic_weight=stochastic_weight,
        )


def build_eagle3_draft(cfg: Config, draft_config: PretrainedConfig):
    from specforge.modeling.auto import AutoDraftModel

    draft_model = AutoDraftModel.from_config(
        draft_config,
        attention_backend=cfg.training.attention_backend,
        torch_dtype=_torch_dtype(cfg),
    )
    _warm_start(
        cfg,
        draft_model,
        draft_config,
        allow_missing_embedding=True,
    )
    _load_vocab_mapping(cfg, draft_model)
    if cfg.model.load_target_embedding:
        draft_model.load_embedding(
            cfg.model.target_model_path,
            embedding_key=cfg.model.embedding_key,
        )
    draft_model.freeze_embedding()
    return draft_model.to(device=_device(), dtype=_torch_dtype(cfg))


def build_peagle_draft(cfg: Config, draft_config: PretrainedConfig):
    from specforge.modeling.draft.peagle import PEagleDraftModel

    draft_model = PEagleDraftModel(
        draft_config,
        norm_before_residual=cfg.training.norm_before_residual,
    )
    if cfg.model.load_target_embedding:
        draft_model.load_embedding(
            cfg.model.target_model_path,
            embedding_key=cfg.model.embedding_key,
        )
    _warm_start(
        cfg,
        draft_model,
        draft_config,
        allow_missing_embedding=True,
    )
    _load_vocab_mapping(cfg, draft_model)
    return draft_model.to(device=_device(), dtype=_torch_dtype(cfg))


def peagle_resume_contract(
    _cfg: Config, draft_model: Any, model: Any
) -> Dict[str, Any]:
    """Persist the resolved P-EAGLE model and objective semantics."""

    return {
        "peagle_num_draft_layers": len(draft_model.layers),
        "peagle_norm_before_residual": bool(draft_model.norm_before_residual),
        "peagle_num_depths": int(model.num_depths),
        "peagle_down_sample_ratio": float(model.down_sample_ratio),
        "peagle_down_sample_ratio_min": float(model.down_sample_ratio_min),
        "peagle_mask_token_id": int(model.mask_token_id),
    }


def _finish_registered_draft(
    cfg: Config,
    draft_config: PretrainedConfig,
    draft_model: Any,
):
    _warm_start(cfg, draft_model, draft_config)
    return draft_model.to(device=_device(), dtype=_torch_dtype(cfg))


def build_registered_draft(cfg: Config, draft_config: PretrainedConfig):
    from specforge.modeling.auto import AutoDraftModel

    draft_config._attn_implementation = cfg.training.attention_backend
    draft_model = AutoDraftModel.from_config(
        draft_config,
        torch_dtype=_torch_dtype(cfg),
    )
    _apply_draft_qat(draft_model)
    return _finish_registered_draft(cfg, draft_config, draft_model)


def build_dflash_draft(
    cfg: Config,
    draft_config: PretrainedConfig,
    kernels: Optional[DFlashKernels],
):
    from specforge.modeling.auto import AutoDraftModel

    draft_config._attn_implementation = cfg.training.attention_backend
    draft_model = AutoDraftModel.from_config(
        draft_config,
        torch_dtype=_torch_dtype(cfg),
        dflash_kernels=kernels,
    )
    _apply_draft_qat(draft_model)
    return _finish_registered_draft(cfg, draft_config, draft_model)


def resolve_eagle_capture_layers(
    cfg: Config, draft_config: Any, target_config: Any
) -> List[int]:
    """Resolve EAGLE capture layers from run override, draft config, or target."""

    layers = cfg.model.aux_hidden_state_layer_ids
    if layers is None:
        eagle_config = (
            draft_config.get("eagle_config", {})
            if isinstance(draft_config, dict)
            else getattr(draft_config, "eagle_config", {})
        ) or {}
        layers = eagle_config.get("eagle_aux_hidden_state_layer_ids")
    if layers is None:
        target_config = getattr(target_config, "text_config", target_config)
        num_layers = int(target_config.num_hidden_layers)
        layers = [1, num_layers // 2 - 1, num_layers - 4]
    layers = list(layers)
    if len(layers) != 3 or any(not isinstance(i, int) or i < 0 for i in layers):
        raise ValueError(
            "resolved EAGLE capture layers must contain exactly three "
            f"non-negative integers, got {layers!r}"
        )
    return layers


def _validate_dflash_block_size(draft_config: Any) -> None:
    if isinstance(draft_config, dict):
        method_config = draft_config.get("dflash_config", {}) or {}
        block_size = draft_config.get("block_size", method_config.get("block_size"))
    else:
        method_config = getattr(draft_config, "dflash_config", {}) or {}
        block_size = getattr(draft_config, "block_size", None)
        if block_size is None:
            block_size = method_config.get("block_size")
    if (
        not isinstance(block_size, int)
        or isinstance(block_size, bool)
        or block_size < 2
    ):
        raise ValueError(
            "DFlash-family draft config must define an integer block_size >= 2"
        )


def resolve_dflash_capture_layers(
    _cfg: Config, draft_config: Any, _target_config: Any
) -> List[int]:
    method_config = (
        draft_config.get("dflash_config", {})
        if isinstance(draft_config, dict)
        else getattr(draft_config, "dflash_config", {})
    ) or {}
    layers = list(method_config.get("target_layer_ids", []))
    if not layers:
        raise ValueError("draft config does not define target capture layer ids")
    return layers


def _resolve_mask_token_id(cfg: Config, draft_model: Any, tokenizer: Any) -> int:
    from specforge.training.model_utils import resolve_mask_token_id

    return resolve_mask_token_id(
        explicit=cfg.model.mask_token_id,
        tokenizer=tokenizer,
        draft_model=draft_model,
        embedding_vocab_size=int(draft_model.config.vocab_size),
    )


def build_eagle3_model(
    cfg: Config,
    draft_model: Any,
    draft_config: Any,
    target_config: Any,
    _tokenizer: Any,
) -> AlgorithmModelParts:
    from specforge.algorithms.eagle3.model import OnlineEagle3Model

    model = OnlineEagle3Model(
        draft_model=draft_model,
        length=cfg.training.ttt_length,
        attention_backend=cfg.training.attention_backend,
        lk_loss_type=cfg.training.lk_loss_type,
        kl_scale=cfg.training.kl_scale,
        kl_decay=cfg.training.kl_decay,
    ).to(device=_device(), dtype=_torch_dtype(cfg))
    needs_target_head = cfg.mode == "offline" or (
        cfg.deployment.mode == "disaggregated" and cfg.training.role == "consumer"
    )
    target_head = None
    if needs_target_head:
        from specforge.modeling.target.target_head import TargetHead

        target_head = TargetHead.from_pretrained(
            cfg.model.target_model_path,
            lm_head_key=cfg.model.lm_head_key,
            cache_dir=cfg.model.cache_dir,
            trust_remote_code=cfg.model.trust_remote_code,
        )
    print_draft_launch_banner(draft_model)
    return AlgorithmModelParts(
        model=model,
        target_head=target_head,
    )


def build_peagle_model(
    cfg: Config,
    draft_model: Any,
    _draft_config: Any,
    _target_config: Any,
    tokenizer: Any,
) -> AlgorithmModelParts:
    from specforge.algorithms.peagle.model import OnlinePEagleModel

    mask_token_id = _resolve_mask_token_id(cfg, draft_model, tokenizer)
    model = OnlinePEagleModel(
        draft_model=draft_model,
        mask_token_id=mask_token_id,
        num_depths=cfg.training.num_depths,
        down_sample_ratio=cfg.training.down_sample_ratio,
        down_sample_ratio_min=cfg.training.down_sample_ratio_min,
    ).to(device=_device(), dtype=_torch_dtype(cfg))
    # Server capture transports the final target hidden state instead of full
    # vocabulary logits.  The consumer resolves that representation through the
    # same frozen target head used by offline EAGLE3; no target model is loaded
    # in the trainer process.
    target_head = None
    if (
        cfg.mode == "online"
        and cfg.deployment.mode == "disaggregated"
        and cfg.training.role == "consumer"
    ):
        from specforge.modeling.target.target_head import TargetHead

        target_head = TargetHead.from_pretrained(
            cfg.model.target_model_path,
            lm_head_key=cfg.model.lm_head_key,
            cache_dir=cfg.model.cache_dir,
            trust_remote_code=cfg.model.trust_remote_code,
        )
    print_draft_launch_banner(draft_model)
    return AlgorithmModelParts(model=model, target_head=target_head)


def describe_draft_mask_choice(training_model: Any, draft_model: Any) -> str:
    """One-line launch banner for the sliding draft-block mask choice.

    Reports the mask the training forward will actually build, the per-layer
    sliding windows, and where the causality decision came from, so a run log
    shows it without opening the draft config.
    """
    from specforge.modeling.draft.dflash import describe_sliding_draft_causal

    config_causal, reason = describe_sliding_draft_causal(
        getattr(draft_model, "config", None)
    )
    causal = bool(getattr(training_model, "sliding_draft_causal", config_causal))
    windows = tuple(getattr(draft_model, "layer_sliding_windows", ()) or ())
    sliding_windows = tuple(window for window in windows if window is not None)
    full_layers = sum(1 for window in windows if window is None)

    if not sliding_windows:
        return (
            "[draft-mask] FULL ATTENTION on every draft layer "
            f"(full_layers={full_layers}); dflash_config.causal has no effect"
        )
    mode = "DRAFT BLOCK CAUSAL" if causal else "DRAFT BLOCK NON-CAUSAL"
    band = "q-(W-1)..q" if causal else "q-(W-1)..q+W"
    return (
        f"[draft-mask] {mode} (band={band}): "
        f"sliding_layers={len(sliding_windows)} "
        f"windows={list(sliding_windows)} full_layers={full_layers}; {reason}"
    )


def describe_draft_features(draft_model: Any) -> str:
    """One-line launch banner for the optional draft features.

    The toggles that change the checkpoint parameterization (shared_emb_readout,
    out_proj_after_norm, vha, ffn_sharing, grouped conv) or the served numerics
    (target_tap_norm, attn_sink, heterogeneous_kv) all ride dflash_config, and
    the serving side builds a different module tree from the same keys — a run
    log should show them without opening the draft config.
    """
    from specforge.modeling.draft.dflash_kernels import resolve_ffn_sharing

    config = getattr(draft_model, "config", None)
    dflash_config = getattr(config, "dflash_config", None) or {}

    features = []
    if dflash_config.get("shared_emb_readout"):
        features.append(
            f"shared_emb_readout(emb_dim={dflash_config.get('emb_dim')})"
        )
    if dflash_config.get("out_proj_after_norm"):
        features.append("out_proj_after_norm")
    vha = dflash_config.get("vha")
    if isinstance(vha, dict) and vha.get("physical_q_heads"):
        features.append(
            "vha(physical_q_heads={p}, postmix_rank={r})".format(
                p=vha.get("physical_q_heads"),
                r=vha.get("postmix_rank"),
            )
        )
    try:
        sharing = resolve_ffn_sharing(config)
    except ValueError:
        # Banner only; the draft build raises the real validation error.
        sharing = {"mode": "<invalid>"}
    if sharing is not None:
        if sharing.get("mode") == "routed_outer":
            features.append(
                "ffn_sharing(routed_outer, experts={e}, router={r})".format(
                    e=sharing.get("experts"),
                    r=sharing.get("router"),
                )
            )
        elif sharing.get("mode") == "<invalid>":
            features.append("ffn_sharing(<invalid>)")
        else:
            features.append("ffn_sharing(lattice)")
    taps = dflash_config.get("conv_kernel_size")
    if taps:
        features.append(
            f"conv(k={taps}, g={dflash_config.get('conv_group_size')})"
        )
    if dflash_config.get("target_tap_norm"):
        features.append("target_tap_norm")
    if int(dflash_config.get("attn_sink", 0) or 0):
        features.append(f"attn_sink={dflash_config.get('attn_sink')}")
    if dflash_config.get("heterogeneous_kv"):
        features.append("heterogeneous_kv")
    layer_types = list(getattr(config, "layer_types", None) or [])
    hca_config = dflash_config.get("hca")
    if isinstance(hca_config, dict) and "heavily_compressed_attention" in layer_types:
        hops = hca_config.get("hops", 128)
        if not isinstance(hops, (list, tuple)):
            hops = [hops]
        features.append(
            "hca(layers={l}, hops={h}, variants={v}, bias={b}, entries={e})".format(
                l=layer_types.count("heavily_compressed_attention"),
                h=list(hops),
                v=hca_config.get("variants", "vanilla"),
                b=hca_config.get("bias_mode", "learnable"),
                e=hca_config.get("max_entries", "auto(2048-W-draft-64)"),
            )
        )

    joined = ", ".join(features) if features else "none (base draft config)"
    return f"[draft-features] {joined}"


def describe_draft_qat(draft_model: Any) -> Optional[str]:
    """One-line launch banner for the draft QAT scheme, None without qat_w_bit."""
    method_config = getattr(getattr(draft_model, "config", None), "dflash_config", None) or {}
    qat_w_bit = method_config.get("qat_w_bit")
    if qat_w_bit is None:
        return None
    w4a4_layers = set(method_config.get("qat_w4a4_layers", []) or [])
    return (
        f"[draft-qat] enabled: w_bit={qat_w_bit}, "
        f"a_bit={method_config.get('qat_a_bit')}, "
        f"exclude={list(method_config.get('qat_exclude', []) or [])}, "
        f"w4a4_layers={sorted(w4a4_layers)}, "
        f"channel_balanced={bool(method_config.get('channel_balanced', False))}"
    )


def print_draft_launch_banner(
    draft_model: Any, training_model: Any = None
) -> None:
    """Print every draft feature line as one rank-0 launch banner block."""
    lines = []
    if getattr(getattr(draft_model, "config", None), "dflash_config", None):
        lines.append(describe_draft_features(draft_model))
    qat = describe_draft_qat(draft_model)
    if qat is not None:
        lines.append(qat)
    if training_model is not None:
        lines.append(describe_draft_mask_choice(training_model, draft_model))
    if lines:
        _print_on_launch_rank0("\n".join(lines))


def _print_on_launch_rank0(message: str) -> None:
    # Feature banners are per-run, not per-rank.
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0:
        print(message, flush=True)


def _build_dflash_family_model(
    cfg: Config,
    draft_model: Any,
    tokenizer: Any,
    model_factory: Callable[[Dict[str, Any]], Any],
) -> AlgorithmModelParts:
    from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead

    _validate_dflash_block_size(draft_model)
    mask_token_id = _resolve_mask_token_id(cfg, draft_model, tokenizer)
    draft_model.mask_token_id = mask_token_id
    method_config = getattr(draft_model.config, "dflash_config", None)
    if method_config is None:
        draft_model.config.dflash_config = {}
        method_config = draft_model.config.dflash_config
    method_config["mask_token_id"] = mask_token_id
    method_config["target_layer_ids"] = list(draft_model.target_layer_ids)

    target_parts = TargetEmbeddingsAndHead.from_pretrained(
        cfg.model.target_model_path,
        embed_key=cfg.model.embedding_key,
        lm_head_key=cfg.model.lm_head_key,
        cache_dir=cfg.model.cache_dir,
        device=_device().type,
        dtype=_torch_dtype(cfg),
        trust_remote_code=cfg.model.trust_remote_code,
    )
    common = {
        "draft_model": draft_model,
        "target_lm_head": target_parts.lm_head,
        "target_embed_tokens": target_parts.embed_tokens,
        "mask_token_id": mask_token_id,
        "block_size": int(draft_model.block_size),
        "attention_backend": cfg.training.attention_backend,
        "num_anchors": cfg.training.num_anchors,
        "loss_decay_gamma": cfg.training.loss_decay_gamma,
        "objective_chunk_blocks": cfg.training.objective_chunk_blocks,
    }
    model = model_factory(common).to(device=_device(), dtype=_torch_dtype(cfg))
    print_draft_launch_banner(draft_model, training_model=model)
    return AlgorithmModelParts(
        model=model,
        capture_layers=list(draft_model.target_layer_ids),
    )


def build_dflash_model(
    cfg: Config,
    draft_model: Any,
    _draft_config: Any,
    _target_config: Any,
    tokenizer: Any,
) -> AlgorithmModelParts:
    from specforge.algorithms.common.dflash_family_model import OnlineDFlashModel

    if cfg.training.mined_loss_variant == "opd":
        raise ValueError(
            "training.mined_loss_variant='opd' is implemented for the domino "
            "strategy only"
        )
    return _build_dflash_family_model(
        cfg,
        draft_model,
        tokenizer,
        lambda common: OnlineDFlashModel(
            **common,
            loss_type=cfg.training.loss_type,
            dpace_alpha=cfg.training.dpace_alpha,
            selector_loss_alpha=cfg.training.dflash2_selector_loss_alpha,
            selector_warmup_ratio=cfg.training.dflash2_selector_warmup_ratio,
            selector_ramp_ratio=cfg.training.dflash2_selector_ramp_ratio,
            selector_stop_gradient=cfg.training.dflash2_selector_stop_gradient,
            lk_loss_type=cfg.training.lk_loss_type,
            kl_scale=cfg.training.kl_scale,
            kl_decay=cfg.training.kl_decay,
            anchor_policy=cfg.training.anchor_policy,
            anchor_mining_chunk_candidates=cfg.training.anchor_mining_chunk_candidates,
        ),
    )


def build_domino_model(
    cfg: Config,
    draft_model: Any,
    _draft_config: Any,
    _target_config: Any,
    tokenizer: Any,
) -> AlgorithmModelParts:
    from specforge.algorithms.common.dflash_family_model import OnlineDominoModel

    return _build_dflash_family_model(
        cfg,
        draft_model,
        tokenizer,
        lambda common: OnlineDominoModel(
            **common,
            shift_label=bool(getattr(draft_model, "shift_label", False)),
            ce_loss_alpha=cfg.training.domino_ce_loss_alpha,
            l1_loss_alpha=cfg.training.domino_l1_loss_alpha,
            base_tv_loss=cfg.training.domino_base_tv_loss,
            final_tv_loss=cfg.training.domino_final_tv_loss,
            anchor_policy=cfg.training.anchor_policy,
            anchor_mining_chunk_candidates=cfg.training.anchor_mining_chunk_candidates,
            anchor_mining_head_chunk_blocks=cfg.training.anchor_mining_head_chunk_blocks,
            mined_loss_variant=cfg.training.mined_loss_variant,
            mined_rejected_decay=cfg.training.mined_rejected_decay,
            mined_rejected_weight=cfg.training.mined_rejected_weight,
            mined_rejected_max_offsets=cfg.training.mined_rejected_max_offsets,
            mined_opd_row_chunk=cfg.training.mined_opd_row_chunk,
        ),
    )


def build_dspark_model(
    cfg: Config,
    draft_model: Any,
    _draft_config: Any,
    _target_config: Any,
    tokenizer: Any,
) -> AlgorithmModelParts:
    from specforge.algorithms.common.dflash_family_model import OnlineDSparkModel

    if cfg.training.anchor_policy == "mined":
        raise ValueError(
            "training.anchor_policy='mined' is not supported for the dspark "
            "strategy yet; its anchor semantics are unverified for chain "
            "mining. Remove the key or switch the strategy."
        )
    return _build_dflash_family_model(
        cfg,
        draft_model,
        tokenizer,
        lambda common: OnlineDSparkModel(
            **common,
            dspark_ce_loss_alpha=cfg.training.dspark_ce_loss_alpha,
            dspark_l1_loss_alpha=cfg.training.dspark_l1_loss_alpha,
            dspark_confidence_head_alpha=(cfg.training.dspark_confidence_head_alpha),
        ),
    )


def eagle3_strategy_kwargs(cfg: Config) -> Dict[str, Any]:
    return {
        "trim_loss_positions": cfg.training.trim_loss_positions,
        "compact_teacher": cfg.training.compact_teacher,
        "compact_teacher_chunk_size": cfg.training.compact_teacher_chunk_size,
    }


def domino_strategy_kwargs(cfg: Config) -> Dict[str, Any]:
    return {
        "lambda_start": cfg.training.lambda_base_start,
        "decay_ratio": cfg.training.lambda_base_decay_ratio,
    }


def dflash_min_loss_tokens(_cfg: Config, draft_config: Any) -> int:
    _validate_dflash_block_size(draft_config)
    return 2


def populate_dflash_generated_config(
    payload: Dict[str, Any], target_config: Any, _cfg: Config
) -> None:
    from specforge.modeling.draft.dflash import build_target_layer_ids

    target_layers = getattr(target_config, "num_hidden_layers", None)
    if not isinstance(target_layers, int) or target_layers < 1:
        raise ValueError(
            "DFlash auto-generation requires target num_hidden_layers, got "
            f"{target_layers!r}"
        )
    payload["num_target_layers"] = target_layers
    payload["block_size"] = 16
    payload["layer_types"] = ["full_attention"] * int(payload["num_hidden_layers"])
    payload["sliding_window"] = None
    payload["use_sliding_window"] = False
    payload["dflash_config"] = {
        "target_layer_ids": build_target_layer_ids(target_layers, 1)
    }


def apply_dflash_overrides(cfg: Config, draft_config: Any) -> None:
    from specforge.modeling.draft.dflash import (
        build_target_layer_ids,
        resolve_dflash_attention_layout,
        validate_dflash_attention_config,
    )

    method_config = dict(getattr(draft_config, "dflash_config", None) or {})
    if cfg.model.draft_block_size is not None:
        method_config["block_size"] = cfg.model.draft_block_size
        draft_config.dflash_config = method_config

    requested_layers = cfg.model.draft_num_hidden_layers
    if requested_layers is not None:
        layer_types = draft_config.layer_types
        if len(layer_types) != requested_layers:
            if len(set(layer_types)) > 1:
                raise ValueError(
                    "model.draft_num_hidden_layers cannot resize a mixed "
                    "DFlash layer_types layout; provide a draft config with "
                    "exactly the requested per-layer layout"
                )
            draft_config.layer_types = [layer_types[0]] * requested_layers

        draft_config.dflash_config = {
            **method_config,
            "target_layer_ids": build_target_layer_ids(
                int(draft_config.num_target_layers),
                requested_layers,
            ),
        }

    resolve_dflash_attention_layout(draft_config)
    validate_dflash_attention_config(draft_config)


__all__ = [
    "AlgorithmModelParts",
    "apply_dflash_overrides",
    "build_dflash_model",
    "build_domino_model",
    "build_dspark_model",
    "build_eagle3_draft",
    "build_eagle3_model",
    "build_peagle_draft",
    "peagle_resume_contract",
    "build_peagle_model",
    "build_registered_draft",
    "dflash_min_loss_tokens",
    "dflash_needs_input_tools",
    "domino_strategy_kwargs",
    "eagle3_strategy_kwargs",
    "populate_dflash_generated_config",
    "resolve_dflash_capture_layers",
    "resolve_eagle_capture_layers",
]
