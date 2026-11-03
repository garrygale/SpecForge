# coding=utf-8
"""Regressions for the GRU warm-up window (correction_warmup_len)."""

import unittest
from types import MethodType

import torch
from torch import nn

from specforge.modeling.draft.dflash import DFlashDraftModel
from specforge.modeling.draft.domino import DominoDraftModel


def _bare_domino(
    *,
    hidden_size: int = 4,
    gru_hidden_size: int = 3,
    embedding_size: int = 2,
    vocab_size: int = 7,
    block_size: int = 4,
) -> DominoDraftModel:
    model = DominoDraftModel.__new__(DominoDraftModel)
    nn.Module.__init__(model)
    model.block_size = block_size
    model.shift_label = False
    model.pure_draft_prefix_len = 0
    model.prefix_gru = nn.GRU(
        input_size=hidden_size,
        hidden_size=gru_hidden_size,
        num_layers=1,
        batch_first=True,
        bias=False,
    )
    model.embed_proj = nn.Sequential(
        nn.Linear(hidden_size + gru_hidden_size, embedding_size, bias=False),
        nn.SiLU(),
        nn.Linear(embedding_size, vocab_size, bias=False),
    )
    return model


class _WarmupStub:
    correction_warmup_len = 3
    _warmup_seed_ids = DFlashDraftModel._warmup_seed_ids


class TestWarmupSeedWindow(unittest.TestCase):
    def test_window_is_the_full_history_left_of_the_anchor(self):
        stub = _WarmupStub()
        output_ids = torch.tensor([[10, 11, 12, 13, 20, 21, 22, 23, 24, 25]])

        # Round 1 (anchor = first response token): the warm-up is the prompt
        # tail, matching the training gather and serving's admission seed.
        self.assertEqual(
            stub._warmup_seed_ids(output_ids, 4).tolist(),
            [[11, 12, 13]],
        )
        # The window crosses the prompt boundary freely.
        self.assertEqual(
            stub._warmup_seed_ids(output_ids, 6).tolist(),
            [[13, 20, 21]],
        )
        self.assertEqual(
            stub._warmup_seed_ids(output_ids, 10).tolist(),
            [[23, 24, 25]],
        )
        # Near the sequence start the first token pads, like the gather's
        # clamp at index 0.
        self.assertEqual(
            stub._warmup_seed_ids(output_ids, 2).tolist(),
            [[10, 10, 11]],
        )
        self.assertEqual(
            stub._warmup_seed_ids(output_ids, 0).tolist(),
            [[10, 10, 10]],
        )


class TestCorrectionLogitsWindow(unittest.TestCase):
    def _reference_states(self, draft, prev_embs, warmup_embs):
        bsz, n_blocks, bs = prev_embs.shape[:3]
        warmup_len = 0 if warmup_embs is None else warmup_embs.shape[2]
        if draft.shift_label:
            gru_inputs = prev_embs
            if warmup_len:
                gru_inputs = torch.cat([warmup_embs, prev_embs], dim=2)
            total = warmup_len + bs
            tail_from = warmup_len + draft.suffix_start
        else:
            gru_inputs = prev_embs[:, :, : bs - 1]
            if warmup_len:
                gru_inputs = torch.cat(
                    [warmup_embs, prev_embs[:, :, : bs - 1]], dim=2
                )
            total = warmup_len + bs - 1
            tail_from = warmup_len + draft.suffix_start - 1
        out = draft.prefix_gru(
            gru_inputs.reshape(bsz * n_blocks, total, -1)
        )[0]
        return out.reshape(bsz, n_blocks, total, -1)[:, :, tail_from:]

    def test_states_match_reference_gru_for_both_label_conventions(self):
        hidden_size, gru_hidden_size, vocab_size, block_size = 4, 3, 7, 4
        for shift_label in (False, True):
            for prefix_len in (0, 1):
                with self.subTest(shift_label=shift_label, prefix_len=prefix_len):
                    torch.manual_seed(7)
                    draft = _bare_domino(
                        hidden_size=hidden_size,
                        gru_hidden_size=gru_hidden_size,
                        vocab_size=vocab_size,
                        block_size=block_size,
                    )
                    draft.shift_label = shift_label
                    draft.pure_draft_prefix_len = prefix_len
                    # Identity head exposes [z, s] directly.
                    draft.embed_proj = nn.Identity()

                    prev_embs = torch.randn(2, 3, block_size, hidden_size)
                    warmup_embs = torch.randn(2, 3, 5, hidden_size)
                    hidden_states = torch.randn(
                        2, 3, block_size, hidden_size
                    )

                    for warmup in (None, warmup_embs):
                        out = draft.compute_correction_logits(
                            prev_token_embeddings=prev_embs,
                            warmup_embeddings=warmup,
                            hidden_states=hidden_states,
                        )
                        z_rows = hidden_states[:, :, draft.suffix_start :, :]
                        self.assertTrue(
                            torch.equal(out[..., :hidden_size], z_rows)
                        )
                        states = self._reference_states(
                            draft, prev_embs, warmup
                        )
                        self.assertTrue(
                            torch.allclose(
                                out[..., hidden_size:], states, atol=1e-6
                            )
                        )

    def test_no_warmup_matches_legacy_formula(self):
        torch.manual_seed(23)
        for shift_label in (False, True):
            with self.subTest(shift_label=shift_label):
                draft = _bare_domino()
                draft.shift_label = shift_label
                prev_embs = torch.randn(2, 3, 4, 4)
                hidden_states = torch.randn(2, 3, 4, 4)
                logits = draft.compute_correction_logits(
                    prev_token_embeddings=prev_embs,
                    hidden_states=hidden_states,
                )
                if shift_label:
                    seq = prev_embs
                    tail = draft.suffix_start
                else:
                    seq = prev_embs[:, :, :3]
                    tail = draft.suffix_start - 1
                gru_out = draft.prefix_gru(seq.reshape(6, seq.shape[2], 4))[
                    0
                ].reshape(2, 3, seq.shape[2], -1)
                legacy = draft.embed_proj(
                    torch.cat(
                        [
                            hidden_states[:, :, draft.suffix_start :, :],
                            gru_out[:, :, tail:, :],
                        ],
                        dim=-1,
                    )
                )
                self.assertTrue(torch.equal(logits, legacy))

    def test_warmup_changes_correction_logits(self):
        torch.manual_seed(29)
        draft = _bare_domino()
        prev_embs = torch.randn(1, 2, 4, 4)
        warmup_embs = torch.randn(1, 2, 3, 4)
        hidden_states = torch.randn(1, 2, 4, 4)
        plain = draft.compute_correction_logits(
            prev_token_embeddings=prev_embs,
            hidden_states=hidden_states,
        )
        warmed = draft.compute_correction_logits(
            prev_token_embeddings=prev_embs,
            warmup_embeddings=warmup_embs,
            hidden_states=hidden_states,
        )
        self.assertFalse(torch.allclose(plain, warmed))


class TestTrainingForward(unittest.TestCase):
    def _fixed_blocks(self, output_hidden):
        def _forward(
            self,
            input_ids,
            hidden_states,
            loss_mask,
            max_valid_anchors=None,
        ):
            del (
                self,
                hidden_states,
                loss_mask,
                max_valid_anchors,
            )
            return (
                torch.zeros(1, 1, dtype=torch.long, device=input_ids.device),
                torch.ones(1, 1, dtype=torch.bool, device=input_ids.device),
                output_hidden.to(input_ids.device),
            )

        return _forward

    def test_forward_gathers_clamped_warmup_window(self):
        from specforge.algorithms.common.dflash_family_model import (
            OnlineDominoModel,
        )

        torch.manual_seed(31)
        draft = _bare_domino()
        draft.shift_label = True
        draft.correction_warmup_len = 2
        embed = nn.Embedding(7, 4)
        model = OnlineDominoModel(
            draft_model=draft,
            target_lm_head=nn.Linear(4, 7, bias=False),
            target_embed_tokens=embed,
            mask_token_id=0,
            block_size=4,
            attention_backend="sdpa",
            num_anchors=1,
            shift_label=True,
        )
        captured = {}
        original = draft.compute_correction_logits

        def spy(**kwargs):
            captured.update(kwargs)
            return original(**kwargs)

        draft.compute_correction_logits = spy
        model._forward_draft_blocks = MethodType(
            self._fixed_blocks(torch.randn(1, 4, 4)), model
        )
        loss, _, _ = model(
            input_ids=torch.tensor([[5, 6, 4, 3, 2]]),
            hidden_states=torch.zeros(1, 4, 4),
            loss_mask=torch.ones(1, 5),
            lambda_base=0.0,
        )
        self.assertFalse(torch.isnan(loss))
        # anchor 0 with warmup 2 -> [x0, x0]; prev -> the block window.
        self.assertTrue(
            torch.equal(captured["warmup_embeddings"], embed(torch.tensor([[[5, 5]]])))
        )
        self.assertTrue(
            torch.equal(
                captured["prev_token_embeddings"], embed(torch.tensor([[[5, 6, 4, 3]]]))
            )
        )

    def test_warmup_gradients_reach_gru_and_projection(self):
        from specforge.algorithms.common.dflash_family_model import (
            OnlineDominoModel,
        )

        torch.manual_seed(37)
        draft = _bare_domino()
        draft.shift_label = True
        draft.correction_warmup_len = 2
        model = OnlineDominoModel(
            draft_model=draft,
            target_lm_head=nn.Linear(4, 7, bias=False),
            target_embed_tokens=nn.Embedding(7, 4),
            mask_token_id=0,
            block_size=4,
            attention_backend="sdpa",
            num_anchors=1,
            shift_label=True,
        )
        model._forward_draft_blocks = MethodType(
            self._fixed_blocks(torch.randn(1, 4, 4)), model
        )
        loss, _, _ = model(
            input_ids=torch.tensor([[1, 2, 3, 4]]),
            hidden_states=torch.zeros(1, 4, 4),
            loss_mask=torch.ones(1, 4),
            lambda_base=0.0,
        )
        loss.backward()
        for module in (draft.prefix_gru, draft.embed_proj):
            grads = [p.grad for p in module.parameters()]
            self.assertTrue(all(g is not None for g in grads))
            self.assertGreater(sum(g.abs().sum().item() for g in grads), 0.0)

    def test_opd_variant_with_warmup_fails_closed(self):
        from specforge.algorithms.common.dflash_family_model import (
            OnlineDominoModel,
        )

        draft = _bare_domino()
        draft.correction_warmup_len = 1
        model = OnlineDominoModel(
            draft_model=draft,
            target_lm_head=nn.Linear(4, 7, bias=False),
            target_embed_tokens=nn.Embedding(7, 4),
            mask_token_id=0,
            block_size=4,
            attention_backend="sdpa",
            num_anchors=1,
            shift_label=False,
            mined_loss_variant="opd",
            anchor_policy="mined",
        )

        def mining_blocks(self, input_ids, hidden_states, loss_mask, max_valid_anchors=None):
            del hidden_states, loss_mask, max_valid_anchors
            self._mined_segments = {
                "segment_lens": torch.zeros(1, 1, dtype=torch.long),
                "proposal_ids": torch.zeros(1, 1, 4, dtype=torch.long),
            }
            return (
                torch.zeros(1, 1, dtype=torch.long, device=input_ids.device),
                torch.ones(1, 1, dtype=torch.bool, device=input_ids.device),
                torch.randn(1, 4, 4, device=input_ids.device),
            )

        model._forward_draft_blocks = MethodType(mining_blocks, model)
        with self.assertRaises(ValueError, msg="correction_warmup_len"):
            model(
                input_ids=torch.tensor([[1, 2, 3, 4]]),
                hidden_states=torch.zeros(1, 4, 4),
                loss_mask=torch.ones(1, 4),
                lambda_base=0.0,
            )

    def test_block_local_sample_fails_closed_with_warmup(self):
        draft = _bare_domino()
        draft.correction_warmup_len = 1
        with self.assertRaises(ValueError, msg="warm-up"):
            draft._sample_draft_tokens(None, None, None)


if __name__ == "__main__":
    unittest.main()
