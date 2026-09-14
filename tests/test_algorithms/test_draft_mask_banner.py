"""Launch banner for the sliding draft-block mask choice."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from specforge.algorithms.model_providers import (
    _build_dflash_family_model,
    describe_draft_mask_choice,
)


def _draft(dflash_config, windows, **overrides):
    return SimpleNamespace(
        config=SimpleNamespace(dflash_config=dict(dflash_config)),
        layer_sliding_windows=tuple(windows),
        **overrides,
    )


class DraftMaskBannerTest(unittest.TestCase):
    def test_domino_default_reports_non_causal_band(self):
        line = describe_draft_mask_choice(
            SimpleNamespace(sliding_draft_causal=False),
            _draft({"projector_type": "domino"}, (3072, 512, None)),
        )
        self.assertIn("[draft-mask] DRAFT BLOCK NON-CAUSAL", line)
        self.assertIn("band=q-(W-1)..q+W", line)
        self.assertIn("sliding_layers=2", line)
        self.assertIn("windows=[3072, 512]", line)
        self.assertIn("full_layers=1", line)
        self.assertIn("Domino default false", line)

    def test_explicit_causal_override_is_reported(self):
        line = describe_draft_mask_choice(
            SimpleNamespace(sliding_draft_causal=True),
            _draft({"projector_type": "domino", "causal": True}, (1024,)),
        )
        self.assertIn("[draft-mask] DRAFT BLOCK CAUSAL", line)
        self.assertIn("band=q-(W-1)..q", line)
        self.assertIn("windows=[1024]", line)
        self.assertIn("dflash_config.causal=True", line)

    def test_dflash_default_reports_causal_band(self):
        line = describe_draft_mask_choice(
            SimpleNamespace(sliding_draft_causal=True),
            _draft({"projector_type": "dflash"}, (2048, 2048)),
        )
        self.assertIn("[draft-mask] DRAFT BLOCK CAUSAL", line)
        self.assertIn("DFlash/DSpark default true", line)

    def test_full_attention_draft_reports_no_window(self):
        line = describe_draft_mask_choice(
            SimpleNamespace(sliding_draft_causal=False),
            _draft({"projector_type": "domino"}, (None, None)),
        )
        self.assertIn("FULL ATTENTION on every draft layer", line)
        self.assertIn("full_layers=2", line)
        self.assertIn("dflash_config.causal has no effect", line)

    def test_training_model_resolution_wins_over_config(self):
        # The banner must show the mask the forward actually builds.
        line = describe_draft_mask_choice(
            SimpleNamespace(sliding_draft_causal=True),
            _draft({"projector_type": "domino"}, (512,)),
        )
        self.assertIn("[draft-mask] DRAFT BLOCK CAUSAL", line)
        self.assertIn("Domino default false", line)


class _Wrapper(nn.Module):
    def __init__(self, **_kwargs):
        super().__init__()
        self.sliding_draft_causal = False


class DraftMaskLaunchPrintTest(unittest.TestCase):
    def test_model_builder_prints_the_banner(self):
        draft = _draft(
            {"projector_type": "domino"},
            (3072, 2048, 512, 512, 1024, 1024, 3072),
            block_size=16,
            target_layer_ids=[1, 7],
            mask_token_id=31,
        )
        config = SimpleNamespace(
            model=SimpleNamespace(
                target_model_path="unused",
                embedding_key=None,
                lm_head_key=None,
                cache_dir=None,
                trust_remote_code=False,
            ),
            training=SimpleNamespace(
                attention_backend="sdpa",
                num_anchors=4,
                loss_decay_gamma=None,
                objective_chunk_blocks=0,
            ),
        )
        target_parts = SimpleNamespace(
            lm_head=nn.Identity(),
            embed_tokens=nn.Embedding(8, 8),
        )
        with (
            mock.patch(
                "specforge.modeling.target.target_utils."
                "TargetEmbeddingsAndHead.from_pretrained",
                return_value=target_parts,
            ),
            mock.patch(
                "specforge.algorithms.model_providers._validate_dflash_block_size"
            ),
            mock.patch(
                "specforge.algorithms.model_providers._resolve_mask_token_id",
                return_value=31,
            ),
            mock.patch(
                "specforge.algorithms.model_providers._device",
                return_value=torch.device("cpu"),
            ),
            mock.patch(
                "specforge.algorithms.model_providers._torch_dtype",
                return_value=torch.float32,
            ),
            mock.patch("builtins.print") as printed,
        ):
            parts = _build_dflash_family_model(
                config,
                draft,
                tokenizer=None,
                model_factory=lambda common: _Wrapper(**common),
            )

        printed.assert_called_once()
        banner = printed.call_args.args[0]
        self.assertIn("[draft-mask] DRAFT BLOCK NON-CAUSAL", banner)
        self.assertIn("[3072, 2048, 512, 512, 1024, 1024, 3072]", banner)
        self.assertEqual(parts.capture_layers, [1, 7])


if __name__ == "__main__":
    unittest.main()
