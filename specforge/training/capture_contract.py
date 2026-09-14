# coding=utf-8
"""One resolved contract shared by server-capture launch and production."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import FrozenSet

from specforge.algorithms.common.providers import ServerCaptureLayout
from specforge.algorithms.registry import AlgorithmRegistration
from specforge.config import Config


@dataclass(frozen=True)
class ServerCaptureContract:
    method: str
    aux_layer_ids: tuple[int, ...]
    target_hidden_size: int
    target_vocab_size: int
    draft_vocab_size: int


@dataclass(frozen=True)
class ResolvedStreamingCapture:
    """The capture request this run actually issues.

    ``required_features`` is the streaming contract's required set plus the
    optional tensors the resolved objective consumes, and ``layout`` drops the
    artifacts that back no needed tensor. A run whose objective never reads an
    optional tensor therefore never asks the capture server for it.
    """

    required_features: FrozenSet[str]
    layout: ServerCaptureLayout


def resolve_streaming_capture(
    cfg: Config,
    *,
    algorithm: AlgorithmRegistration,
) -> ResolvedStreamingCapture:
    """Resolve the effective streaming capture request for one run."""

    modality = cfg.model.input_modality
    streaming = algorithm.providers.server_streaming_for(modality)
    contract = algorithm.spec.feature_contract("streaming", modality)
    needed = frozenset()
    if streaming.resolve_optional_tensors is not None:
        needed = frozenset(streaming.resolve_optional_tensors(cfg))
    undeclared = needed - contract.optional_tensors
    if undeclared:
        raise ValueError(
            f"algorithm {algorithm.name!r} requested undeclared optional "
            f"tensors {sorted(undeclared)} for modality {modality!r}"
        )

    layout = streaming.layout
    last_hidden = layout.last_hidden_feature
    if (
        last_hidden is not None
        and last_hidden in contract.optional_tensors
        and last_hidden not in needed
    ):
        # The artifact backs an optional tensor this run does not consume, so
        # keep it out of the capture request entirely.
        layout = replace(layout, last_hidden_feature=None)

    emitted = {
        *(
            feature
            for feature in (
                layout.aux_feature,
                layout.last_hidden_feature,
                layout.attention_mask_feature,
            )
            if feature is not None
        ),
        *(feature for feature, _payload, _shape in layout.passthrough),
    }
    unbacked = needed - emitted
    if unbacked:
        raise ValueError(
            f"algorithm {algorithm.name!r} needs optional tensors "
            f"{sorted(unbacked)} for modality {modality!r}, but its server "
            f"capture layout only emits {sorted(emitted)}"
        )
    return ResolvedStreamingCapture(
        required_features=frozenset(contract.required_tensors) | needed,
        layout=layout,
    )


def resolve_server_capture_contract(
    cfg: Config,
    *,
    algorithm: AlgorithmRegistration,
) -> ServerCaptureContract:
    """Resolve engine flags and feature dimensions from canonical model config."""
    from specforge.modeling.target.target_utils import (
        load_target_config,
        target_text_config,
        target_vocab_size,
    )
    from specforge.training.model_loading import draft_config_dict

    streaming = algorithm.providers.server_streaming_for(cfg.model.input_modality)

    target_cfg = load_target_config(
        cfg.model.target_model_path,
        cache_dir=cfg.model.cache_dir,
        trust_remote_code=cfg.model.trust_remote_code,
    )
    target_cfg = target_text_config(target_cfg)
    model_provider = algorithm.providers.model
    draft_cfg = draft_config_dict(cfg, provider=model_provider.draft_config)
    layers = model_provider.resolve_capture_layers(cfg, draft_cfg, target_cfg)
    if not layers:
        raise ValueError("draft config does not define target capture layer ids")
    if any(
        isinstance(layer, bool) or not isinstance(layer, int) or layer < 0
        for layer in layers
    ):
        raise ValueError(
            "resolved server capture layer ids must be non-negative integers, "
            f"got {layers!r}"
        )
    if len(set(layers)) != len(layers):
        raise ValueError(
            f"resolved server capture layer ids must be unique, got {layers!r}"
        )

    return ServerCaptureContract(
        method=streaming.capture_method,
        aux_layer_ids=tuple(layers),
        target_hidden_size=int(target_cfg.hidden_size),
        target_vocab_size=target_vocab_size(target_cfg),
        draft_vocab_size=int(
            draft_cfg.get("draft_vocab_size") or draft_cfg["vocab_size"]
        ),
    )


__all__ = [
    "ResolvedStreamingCapture",
    "ServerCaptureContract",
    "resolve_server_capture_contract",
    "resolve_streaming_capture",
]
