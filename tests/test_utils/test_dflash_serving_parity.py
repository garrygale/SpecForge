"""Training masks must reproduce the served draft KV visibility.

vLLM / vLLM-Ascend run sliding draft layers with a per-layer window. For a
non-causal draft (Domino, DFlash2) that window is a symmetric band
``[q-(W-1), q+W]`` — Ascend FIA ``sparse_mode=4`` with
``pre_tokens=next_tokens=W``; CUDA symmetrizes ``(W-1, 0)`` the same way — so
draft positions may read the other elements of their own block. For a causal
draft (DFlash, DSpark) the served band is ``[q-(W-1), q]``.

These tests pin the training mask to those two contracts and check the
config-level resolution that selects between them.
"""

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn
from transformers import Qwen3Config

from specforge.algorithms.common.dflash_family_model import (
    FLEX_ATTENTION_AVAILABLE,
    OnlineDFlashModel,
    create_dflash_block_mask,
    create_dflash_sdpa_mask,
)
from specforge.modeling.draft.dflash import (
    describe_sliding_draft_causal,
    resolve_sliding_draft_causal,
)


def _served_band_mask(
    *,
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    S: int,
    block_size: int,
    window: int,
    causal: bool,
) -> torch.Tensor:
    """Element-level reference of the served per-layer draft window.

    Block element ``k`` of the query's own block sits at absolute position
    ``anchor + k`` (training) / ``L + 1 + k`` (serving), so with
    ``q = anchor + j`` the served visibility is

    * context: ``kv < anchor`` and ``kv >= q - (W-1)``
    * block:   ``q - (W-1) <= kv_position <= (q if causal else q + W)``
    """
    B, N = anchor_positions.shape
    Q_LEN = N * block_size
    KV_LEN = S + N * block_size
    mask = torch.zeros(B, 1, Q_LEN, KV_LEN, dtype=torch.bool)
    for b in range(B):
        for q_idx in range(Q_LEN):
            q_block = q_idx // block_size
            q_offset = q_idx % block_size
            anchor = int(anchor_positions[b, q_block])
            q = anchor + q_offset
            if not bool(block_keep_mask[b, q_block]):
                continue
            lower = q - (window - 1)
            upper = q if causal else q + window
            for kv_idx in range(KV_LEN):
                if kv_idx < S:
                    visible = kv_idx < anchor and kv_idx >= lower
                else:
                    kv_block = (kv_idx - S) // block_size
                    if kv_block != q_block:
                        continue
                    kv_pos = anchor + (kv_idx - S) % block_size
                    visible = lower <= kv_pos <= upper
                mask[b, 0, q_idx, kv_idx] = visible
    return mask


class TestSlidingDraftServingParity(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _mask(self, mask_builder, *, window, block_size, anchor, S=64, causal=False):
        anchors = torch.tensor([[anchor]]).to(self.device)
        keep = torch.tensor([[True]]).to(self.device)
        return mask_builder(
            anchor_positions=anchors,
            block_keep_mask=keep,
            S=S,
            block_size=block_size,
            device=self.device,
            sliding_window=window,
            sliding_draft_causal=causal,
        ), anchors, keep

    def test_default_sliding_mask_matches_served_non_causal_band(self):
        for window in (1, 2, 4, 8, 16, 32, 512):
            for block_size in (4, 16):
                for anchor in (0, 6, 12):
                    with self.subTest(
                        window=window, block_size=block_size, anchor=anchor
                    ):
                        mask, anchors, keep = self._mask(
                            create_dflash_sdpa_mask,
                            window=window,
                            block_size=block_size,
                            anchor=anchor,
                        )
                        expected = _served_band_mask(
                            anchor_positions=anchors.cpu(),
                            block_keep_mask=keep.cpu(),
                            S=64,
                            block_size=block_size,
                            window=window,
                            causal=False,
                        )
                        self.assertTrue(
                            torch.equal(mask.cpu(), expected),
                            "training mask diverged from the served band",
                        )

    def test_legacy_flag_matches_served_causal_band(self):
        for window in (1, 4, 16, 512):
            with self.subTest(window=window):
                mask, anchors, keep = self._mask(
                    create_dflash_sdpa_mask,
                    window=window,
                    block_size=8,
                    anchor=16,
                    causal=True,
                )
                expected = _served_band_mask(
                    anchor_positions=anchors.cpu(),
                    block_keep_mask=keep.cpu(),
                    S=64,
                    block_size=8,
                    window=window,
                    causal=True,
                )
                self.assertTrue(torch.equal(mask.cpu(), expected))

    def test_default_and_causal_differ_for_small_windows(self):
        banded, _, _ = self._mask(
            create_dflash_sdpa_mask, window=4, block_size=8, anchor=16
        )
        causal, _, _ = self._mask(
            create_dflash_sdpa_mask, window=4, block_size=8, anchor=16, causal=True
        )
        self.assertFalse(torch.equal(banded.cpu(), causal.cpu()))
        # Both modes share the sliding lower bound, so the band is a strict
        # superset: it adds exactly the query's own future block elements.
        self.assertTrue(bool((banded & ~causal).any()))
        self.assertFalse(bool((causal & ~banded).any()))
        self.assertTrue(torch.equal(banded[..., :64].cpu(), causal[..., :64].cpu()))

    def test_default_and_causal_differ_only_in_future_block_elements(self):
        for window in (8, 16, 64):
            with self.subTest(window=window):
                banded, _, _ = self._mask(
                    create_dflash_sdpa_mask, window=window, block_size=8, anchor=16
                )
                causal, _, _ = self._mask(
                    create_dflash_sdpa_mask,
                    window=window,
                    block_size=8,
                    anchor=16,
                    causal=True,
                )
                # With W >= block_size the legacy causal mask keeps the whole
                # block prefix and the band adds the forward half.
                self.assertFalse(torch.equal(banded.cpu(), causal.cpu()))
                self.assertFalse(bool((causal & ~banded).any()))
                difference = (banded & ~causal)[0, 0]
                # Every added slot is a block slot of the query's own block.
                self.assertGreater(int(difference.sum()), 0)
                self.assertFalse(bool(difference[:, :64].any()))

    def test_full_attention_layers_ignore_the_flag(self):
        for causal in (False, True):
            with self.subTest(causal=causal):
                anchors = torch.tensor([[32]]).to(self.device)
                keep = torch.tensor([[True]]).to(self.device)
                mask = create_dflash_sdpa_mask(
                    anchor_positions=anchors,
                    block_keep_mask=keep,
                    S=64,
                    block_size=8,
                    device=self.device,
                    sliding_draft_causal=causal,
                )
                # Context before the anchor plus the whole own block; target
                # taps at or after the anchor stay closed.
                self.assertTrue(bool(mask[0, 0, :, :32].all()))
                self.assertFalse(bool(mask[0, 0, :, 32:64].any()))
                self.assertTrue(bool(mask[0, 0, :, 64:].all()))

    @unittest.skipUnless(FLEX_ATTENTION_AVAILABLE, "flex attention unavailable")
    def test_flex_builder_matches_served_band(self):
        for window in (2, 8, 64):
            with self.subTest(window=window):
                dense, anchors, keep = self._mask(
                    create_dflash_sdpa_mask,
                    window=window,
                    block_size=8,
                    anchor=12,
                )
                flex = create_dflash_block_mask(
                    anchor_positions=anchors,
                    block_keep_mask=keep,
                    S=64,
                    block_size=8,
                    device=self.device,
                    sliding_window=window,
                    sliding_draft_causal=False,
                )
                for q in range(8):
                    for kv in range(64 + 8):
                        self.assertEqual(
                            bool(dense[0, 0, q, kv]),
                            bool(
                                flex.mask_mod(
                                    torch.tensor(0),
                                    torch.tensor(0),
                                    torch.tensor(q),
                                    torch.tensor(kv),
                                )
                            ),
                        )


class TestSlidingDraftCausalResolution(unittest.TestCase):
    def _config(self, *, dflash_config=None):
        config = SimpleNamespace(
            num_hidden_layers=2,
            layer_types=["sliding_attention", "sliding_attention"],
            sliding_window=1024,
        )
        if dflash_config is not None:
            config.dflash_config = dflash_config
        return config

    def test_serving_causal_key_is_honoured_for_every_family(self):
        # The service's only knob is dflash_config.causal; training mirrors it,
        # so an explicit value wins for Domino and DFlash/DSpark alike.
        for projector in ("domino", "dflash", "dspark", None):
            for causal in (True, False):
                with self.subTest(projector=projector, causal=causal):
                    method_config = {"causal": causal}
                    if projector is not None:
                        method_config["projector_type"] = projector
                    self.assertIs(
                        resolve_sliding_draft_causal(
                            self._config(dflash_config=method_config)
                        ),
                        causal,
                    )

    def test_family_defaults(self):
        # Domino is served with a non-causal band by default -> bidirectional.
        self.assertFalse(
            resolve_sliding_draft_causal(
                self._config(dflash_config={"projector_type": "domino"})
            )
        )
        # DFlash/DSpark keep sliding layers causal.
        self.assertTrue(resolve_sliding_draft_causal(self._config(dflash_config={})))
        self.assertTrue(
            resolve_sliding_draft_causal(
                self._config(dflash_config={"projector_type": "dflash"})
            )
        )

    def test_describe_reports_the_source_of_the_decision(self):
        domino = self._config(dflash_config={"projector_type": "domino"})
        self.assertEqual(
            describe_sliding_draft_causal(domino),
            (False, "dflash_config.causal not set -> Domino default false"),
        )
        dflash = self._config(dflash_config={"projector_type": "dflash"})
        self.assertEqual(
            describe_sliding_draft_causal(dflash),
            (True, "dflash_config.causal not set -> DFlash/DSpark default true"),
        )
        explicit = self._config(dflash_config={"causal": True})
        self.assertEqual(
            describe_sliding_draft_causal(explicit),
            (True, "dflash_config.causal=True"),
        )

    def test_checked_in_domino_configs_train_the_served_band(self):
        configs = (
            "qwen3-8b-domino-dflare-verifiedBase.json",
            "qwen3.6-35b-a3b-domino-dflare-verifiedBase.json",
            "qwen3.8-27b-domino-dflare-verifiedBase.json",
        )
        root = Path(__file__).resolve().parents[2] / "configs"
        for name in configs:
            with self.subTest(config=name):
                config = Qwen3Config.from_json_file(str(root / name))
                method_config = dict(config.dflash_config or {})
                # No training-only key is needed: the service reads
                # dflash_config.causal (absent -> False = non-causal band) and
                # the resolver mirrors exactly that.
                self.assertNotIn("sliding_draft_causal", method_config)
                self.assertFalse(bool(method_config.get("causal", False)))
                self.assertFalse(resolve_sliding_draft_causal(config))


class _RecordingDraft(nn.Module):
    """Minimal draft stub that records the mask builder kwargs."""

    def __init__(self, *, layer_sliding_windows, sliding_draft_causal):
        super().__init__()
        self.config = SimpleNamespace(num_hidden_layers=len(layer_sliding_windows))
        self.layer_sliding_windows = tuple(layer_sliding_windows)
        self.sliding_window = None
        self.sliding_draft_causal = sliding_draft_causal
        self.attention_mask = None

    def forward(self, *, noise_embedding, attention_mask, **_):
        self.attention_mask = attention_mask
        return noise_embedding


class TestWrapperPlumbsSlidingDraftCausal(unittest.TestCase):
    def _run(self, *, sliding_draft_causal):
        draft = _RecordingDraft(
            layer_sliding_windows=(8, None),
            sliding_draft_causal=sliding_draft_causal,
        )
        wrapper = OnlineDFlashModel(
            draft_model=draft,
            target_lm_head=nn.Identity(),
            target_embed_tokens=nn.Embedding(32, 8),
            mask_token_id=31,
            block_size=4,
            attention_backend="sdpa",
            num_anchors=1,
        )
        anchors = torch.tensor([[12]])
        keep = torch.tensor([[True]])
        with (
            mock.patch.object(
                wrapper,
                "_sample_anchor_positions",
                return_value=(anchors, keep),
            ),
            mock.patch.object(
                wrapper, "_create_noise_embed", return_value=torch.randn(1, 4, 8)
            ),
            mock.patch(
                "specforge.algorithms.common.dflash_family_model."
                "create_dflash_sdpa_mask",
                side_effect=(torch.tensor([0]), torch.tensor([1])),
            ) as builder,
        ):
            wrapper._forward_draft_blocks(
                input_ids=torch.ones(1, 16, dtype=torch.long),
                hidden_states=torch.randn(1, 16, 8),
                loss_mask=torch.ones(1, 16),
            )
        return wrapper, builder

    def test_default_is_non_causal_for_every_sliding_layer(self):
        wrapper, builder = self._run(sliding_draft_causal=False)
        self.assertFalse(wrapper.sliding_draft_causal)
        self.assertEqual(builder.call_count, 2)
        # First call builds the full-attention mask, second one the sliding
        # mask for the layer with window 8.
        self.assertNotIn("sliding_window", builder.call_args_list[0].kwargs)
        sliding_kwargs = builder.call_args_list[1].kwargs
        self.assertEqual(sliding_kwargs["sliding_window"], 8)
        self.assertFalse(sliding_kwargs["sliding_draft_causal"])

    def test_legacy_flag_is_forwarded(self):
        wrapper, builder = self._run(sliding_draft_causal=True)
        self.assertTrue(wrapper.sliding_draft_causal)
        self.assertTrue(builder.call_args_list[1].kwargs["sliding_draft_causal"])


if __name__ == "__main__":
    unittest.main()
