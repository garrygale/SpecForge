"""Speculative decoding against a hybrid linear-attention target.

Qwen3.5 / Qwen3.5-MoE interleave linear-attention layers with full-attention
layers. Their cache has to be built from the model config and rolled back by
token count, so this covers the regression where ``spec_generate`` handed the
target a bare ``DynamicCache()`` and every problem failed with
``has_previous_state can only be called on LinearAttention layers``.

The check is numerical rather than structural: at temperature 0 the committed
sequence of a correct speculative loop is exactly the target's greedy
continuation, so it is compared against one.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

import torch
from transformers import AutoModelForCausalLM

from specforge.modeling.auto import AutoDraftModel, AutoDraftModelConfig
from specforge.modeling.draft import dflash as dflash_module
from inference.check_acceptance import (
    _install_output_localizers,
    _verify_tp_outputs,
)

try:
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
        Qwen3_5MoeTextConfig,
    )
except Exception:  # transformers without the Qwen3.5-MoE architecture
    Qwen3_5MoeTextConfig = None

_HIDDEN_SIZE = 32
_VOCAB_SIZE = 64
_TARGET_LAYERS = [1, 2]
_BLOCK_SIZE = 4
_EOS_TOKEN_ID = 1
_MAX_NEW_TOKENS = 8


def _hybrid_target_config():
    return Qwen3_5MoeTextConfig(
        vocab_size=_VOCAB_SIZE,
        hidden_size=_HIDDEN_SIZE,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=64,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        layer_types=[
            "linear_attention",
            "full_attention",
            "linear_attention",
            "full_attention",
        ],
    )


def _draft_config(
    architecture: str, projector: str | None, sliding_window: bool = False
) -> dict:
    dflash_config = {
        "mask_token_id": _VOCAB_SIZE - 1,
        "target_layer_ids": list(_TARGET_LAYERS),
    }
    if projector is not None:
        dflash_config.update(
            {
                "projector_type": projector,
                "fusion_mode": "flare",
                "heterogeneous_kv": True,
                "pure_draft_prefix_len": 1,
                "emb_dim": 8,
                "gru_hidden_dim": 16,
                "shift_label": True,
                "use_hidden_proj": False,
                "hidden_proj_dim": 16,
            }
        )
    layer_type = "sliding_attention" if sliding_window else "full_attention"
    return {
        "architectures": [architecture],
        "model_type": "qwen3",
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "num_hidden_layers": 1,
        "layer_types": [layer_type],
        "num_target_layers": 4,
        "target_hidden_size": len(_TARGET_LAYERS) * _HIDDEN_SIZE,
        "block_size": _BLOCK_SIZE,
        "vocab_size": _VOCAB_SIZE,
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000,
        "sliding_window": [4] if sliding_window else None,
        "use_sliding_window": sliding_window,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "tie_word_embeddings": False,
        "use_cache": True,
        "dflash_config": dflash_config,
    }


def _build_draft(architecture: str, projector: str | None, sliding_window: bool = False):
    directory = tempfile.mkdtemp()
    path = os.path.join(directory, "tiny_draft.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_draft_config(architecture, projector, sliding_window), handle)
    config = AutoDraftModelConfig.from_file(path)
    draft = AutoDraftModel.from_config(config, torch_dtype=torch.float32)
    return draft.eval()


def _greedy_reference(target, input_ids: torch.Tensor, max_new_tokens: int) -> list[int]:
    """Target-only greedy continuation, the acceptance loop's reference."""
    tokens = input_ids
    generated: list[int] = []
    for _ in range(max_new_tokens):
        with torch.inference_mode():
            output = target(tokens, use_cache=False, logits_to_keep=1)
        next_token = int(output.logits[:, -1, :].argmax(dim=-1).item())
        generated.append(next_token)
        tokens = torch.cat(
            [tokens, torch.tensor([[next_token]], dtype=torch.long)], dim=1
        )
        if next_token == _EOS_TOKEN_ID:
            break
    return generated


@unittest.skipIf(
    Qwen3_5MoeTextConfig is None, "transformers has no Qwen3.5-MoE config"
)
class HybridTargetSpecGenerateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.target = AutoModelForCausalLM.from_config(
            _hybrid_target_config(), dtype=torch.float32
        ).eval()
        cls.input_ids = torch.tensor([[2, 3, 4, 5]])

    def _run(self, architecture: str, projector: str | None, sliding_window=False):
        draft = _build_draft(architecture, projector, sliding_window)
        generated, stats = draft.spec_generate(
            target=self.target,
            input_ids=self.input_ids,
            max_new_tokens=_MAX_NEW_TOKENS,
            stop_token_ids=[_EOS_TOKEN_ID],
            temperature=0.0,
            return_acceptance_stats=True,
        )
        self.assertIsNotNone(stats["mean_acceptance_length"])
        self.assertGreater(stats["num_complete_blocks"], 0)
        committed = generated[0, self.input_ids.shape[1] :].tolist()
        self.assertEqual(
            committed,
            _greedy_reference(self.target, self.input_ids, _MAX_NEW_TOKENS),
        )

    def test_domino_draft_matches_target_greedy_on_hybrid_target(self):
        self._run("DominoDraftModel", "domino")

    def test_dflash_draft_matches_target_greedy_on_hybrid_target(self):
        self._run("DFlashDraftModel", None)

    def test_sliding_window_draft_rolls_back_on_long_generation(self):
        self._run("DominoDraftModel", "domino", sliding_window=True)

    def test_target_cache_is_hybrid_aware(self):
        cache = dflash_module.build_decoding_cache(self.target)
        layer_names = [type(layer).__name__ for layer in cache.layers]
        self.assertTrue(
            any(name.startswith("LinearAttention") for name in layer_names),
            layer_names,
        )
        self.assertIn("DynamicLayer", layer_names)
        self.assertTrue(
            all(getattr(layer, "record_past", True) for layer in cache.layers)
        )

    def test_tensor_parallel_output_probe_accepts_the_hybrid_target(self):
        # The probe runs at startup for --tp > 1 and rejects an ungathered
        # vocabulary, so it has to work on a real hybrid target.
        hooked = _install_output_localizers(self.target)
        self.assertIn("target", hooked)
        _verify_tp_outputs(self.target)


if __name__ == "__main__":
    unittest.main()
