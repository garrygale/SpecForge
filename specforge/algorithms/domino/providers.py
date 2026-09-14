"""Built-in Domino registration and executable providers."""

from __future__ import annotations

from functools import partial

from specforge.algorithms.common.defaults import no_missing_checkpoint_keys
from specforge.algorithms.common.hidden_states_data import (
    DOMINO_NORMALIZER_ID,
    build_domino_collator,
    build_domino_offline_normalizer,
    build_domino_offline_reader,
)
from specforge.algorithms.common.providers import (
    AlgorithmProviders,
    DraftConfigProvider,
    ModelProvider,
    OfflineCaptureLayout,
    OfflineDataProvider,
    ServerCaptureLayout,
    ServerStreamingProvider,
    StepProvider,
    make_registration,
)
from specforge.algorithms.contracts import (
    AlgorithmCapabilities,
    AlgorithmSpec,
    DraftRequirement,
    FeatureContract,
    FeatureMode,
    OfflineStorageContract,
)
from specforge.data.loss_mask import has_consecutive_supervised_tokens

ALGORITHM_NAME = "domino"
DRAFT_ARCHITECTURE = "DominoDraftModel"


def build_step(
    wrapped_model,
    *,
    target_head=None,
    lambda_start=1.0,
    decay_ratio=0.5,
):
    del target_head
    from specforge.training.strategies.base import DominoTrainStrategy

    return DominoTrainStrategy(
        wrapped_model,
        lambda_start=lambda_start,
        decay_ratio=decay_ratio,
    )


def step_options(config):
    from specforge.algorithms.model_providers import domino_strategy_kwargs

    return domino_strategy_kwargs(config)


def resolve_optional_tensors(config):
    """Request the captured target distribution only when the objective uses it.

    ``target_last_hidden_states`` is optional for Domino: the CE-only objective
    never reads it, so an online run must not depend on a capture server that
    produces the ``last_hidden`` artifact.
    """

    if config.training.domino_l1_enabled:
        return frozenset({"target_last_hidden_states"})
    return frozenset()


def resume_contract(config, draft_model, training_model):
    """Persist resolved Domino model, sampling, and objective semantics."""
    from specforge.modeling.draft.dflash import (
        resolve_training_sliding_draft_causal,
    )

    sliding_draft_causal = resolve_training_sliding_draft_causal(
        training_model, draft_model
    )

    return {
        "domino_draft_num_hidden_layers": int(draft_model.config.num_hidden_layers),
        "domino_target_layer_ids": tuple(
            int(layer_id) for layer_id in draft_model.target_layer_ids
        ),
        "domino_block_size": int(training_model.block_size),
        "domino_mask_token_id": int(training_model.mask_token_id),
        "domino_attention_backend": str(training_model.attention_backend),
        "domino_num_anchors": int(training_model.num_anchors),
        "domino_loss_decay_gamma": training_model.loss_decay_gamma,
        "domino_shift_label": bool(training_model.shift_label),
        "domino_sliding_draft_causal": bool(sliding_draft_causal),
        "domino_pure_draft_prefix_len": int(draft_model.pure_draft_prefix_len),
        "domino_target_hidden_size": int(draft_model.target_hidden_size),
        "domino_fusion_mode": str(
            getattr(draft_model.config, "dflash_config", {}).get("fusion_mode", "")
        ),
        "domino_heterogeneous_kv": bool(
            getattr(draft_model.config, "dflash_config", {}).get(
                "heterogeneous_kv", False
            )
        ),
        "domino_lambda_base_start": float(config.training.lambda_base_start),
        "domino_lambda_base_decay_ratio": float(
            config.training.lambda_base_decay_ratio
        ),
        "domino_ce_loss_alpha": float(training_model.domino_ce_loss_alpha),
        "domino_l1_loss_alpha": float(training_model.domino_l1_loss_alpha),
        "domino_base_tv_loss": bool(training_model.domino_base_tv_loss),
        "domino_final_tv_loss": bool(training_model.domino_final_tv_loss),
    }


def build_draft(config, draft_config):
    from specforge.algorithms.model_providers import build_registered_draft

    return build_registered_draft(config, draft_config)


def build_training_model(config, draft_model, draft_config, target_config, tokenizer):
    from specforge.algorithms.model_providers import build_domino_model

    return build_domino_model(
        config,
        draft_model,
        draft_config,
        target_config,
        tokenizer,
    )


def resolve_capture_layers(config, draft_config, target_config):
    from specforge.algorithms.model_providers import resolve_dflash_capture_layers

    return resolve_dflash_capture_layers(config, draft_config, target_config)


def minimum_loss_tokens(config, draft_config):
    from specforge.algorithms.model_providers import dflash_min_loss_tokens

    return dflash_min_loss_tokens(config, draft_config)


def needs_input_tools(config, draft_model):
    from specforge.algorithms.model_providers import dflash_needs_input_tools

    return dflash_needs_input_tools(config, draft_model)


def algorithm_spec() -> AlgorithmSpec:
    ready = {
        "input_ids",
        "loss_mask",
        "hidden_states",
        "target_last_hidden_states",
    }
    return AlgorithmSpec(
        name=ALGORITHM_NAME,
        draft=DraftRequirement(
            compatible_architectures={DRAFT_ARCHITECTURE},
            default_architecture=DRAFT_ARCHITECTURE,
        ),
        feature_contracts=(
            FeatureContract(
                mode=FeatureMode.OFFLINE,
                modality="text",
                required_tensors=ready,
                allowed_target_representations={"hidden_state"},
                default_target_representation="hidden_state",
                storage=OfflineStorageContract(
                    format="specforge_hidden_states_v1",
                    required_tensors=ready,
                    normalizer=DOMINO_NORMALIZER_ID,
                ),
            ),
            FeatureContract(
                mode=FeatureMode.STREAMING,
                modality="text",
                required_tensors=ready - {"target_last_hidden_states"},
                optional_tensors={"target_last_hidden_states"},
                allowed_target_representations={"hidden_state"},
                default_target_representation="hidden_state",
            ),
        ),
        capabilities=AlgorithmCapabilities(
            attention_backends={"eager", "sdpa", "flex_attention"},
        ),
    )


def algorithm_providers() -> AlgorithmProviders:
    collator = build_domino_collator
    return AlgorithmProviders(
        algorithm_name=ALGORITHM_NAME,
        step=StepProvider(
            build=build_step,
            options=step_options,
            resume_contract=resume_contract,
            allowed_missing_checkpoint_keys=no_missing_checkpoint_keys,
            uses_external_target_head=False,
        ),
        model=ModelProvider(
            draft_config=DraftConfigProvider(
                architecture=DRAFT_ARCHITECTURE,
                expected_auto_map_model="domino.DominoDraftModel",
            ),
            build_draft=build_draft,
            build_training_model=build_training_model,
            resolve_capture_layers=resolve_capture_layers,
            minimum_loss_tokens=minimum_loss_tokens,
            needs_input_tools=needs_input_tools,
            default_dataloader_num_workers=8,
            loss_mask_filter=has_consecutive_supervised_tokens,
        ),
        offline=(
            OfflineDataProvider(
                modality="text",
                normalizer_id=DOMINO_NORMALIZER_ID,
                capture_layout=OfflineCaptureLayout(
                    capture_method="dflash",
                    aux_feature="hidden_states",
                    last_hidden_feature="target_last_hidden_states",
                    passthrough=(
                        ("input_ids", "input_ids"),
                        ("loss_mask", "loss_mask"),
                    ),
                ),
                build_reader=partial(build_domino_offline_reader, ALGORITHM_NAME),
                build_normalizer=build_domino_offline_normalizer,
                build_collator=collator,
            ),
        ),
        server_streaming=(
            ServerStreamingProvider(
                modality="text",
                capture_method="dflash",
                target_representation="hidden_state",
                layout=ServerCaptureLayout(
                    aux_feature="hidden_states",
                    last_hidden_feature="target_last_hidden_states",
                    passthrough=(
                        ("input_ids", "input_ids", ()),
                        ("loss_mask", "loss_mask", ()),
                    ),
                ),
                build_collator=collator,
                resolve_optional_tensors=resolve_optional_tensors,
            ),
        ),
    )


def create_registration():
    return make_registration(algorithm_spec(), algorithm_providers())


__all__ = [
    "algorithm_providers",
    "algorithm_spec",
    "create_registration",
    "resolve_optional_tensors",
]
