# coding=utf-8
"""Focused CPU regressions for Domino's auxiliary logits head."""

import copy
import unittest
from types import MethodType
from unittest.mock import patch

import torch
from torch import nn

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


class _TargetBody(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)


class _TinyTarget(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.model = _TargetBody(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)


class TestDominoDraftModel(unittest.TestCase):
    def test_training_loss_updates_gru_and_projection(self):
        from specforge.algorithms.common.dflash_family_model import OnlineDominoModel

        torch.manual_seed(11)
        hidden_size, vocab_size, block_size = 4, 7, 4
        draft = _bare_domino(
            hidden_size=hidden_size,
            gru_hidden_size=3,
            embedding_size=2,
            vocab_size=vocab_size,
            block_size=block_size,
        )
        model = OnlineDominoModel(
            draft_model=draft,
            target_lm_head=nn.Linear(hidden_size, vocab_size, bias=False),
            target_embed_tokens=nn.Embedding(vocab_size, hidden_size),
            mask_token_id=0,
            block_size=block_size,
            attention_backend="sdpa",
            num_anchors=1,
            shift_label=False,
        )
        fixed_hidden = torch.randn(1, block_size, hidden_size)

        def fixed_draft_blocks(
            self,
            input_ids,
            hidden_states,
            loss_mask,
            max_valid_anchors=None,
        ):
            del hidden_states, loss_mask, max_valid_anchors
            return (
                torch.zeros(1, 1, dtype=torch.long, device=input_ids.device),
                torch.ones(1, 1, dtype=torch.bool, device=input_ids.device),
                fixed_hidden.to(input_ids.device),
            )

        model._forward_draft_blocks = MethodType(fixed_draft_blocks, model)
        loss, _accuracy, metrics = model(
            input_ids=torch.tensor([[1, 2, 3, 4]]),
            hidden_states=torch.zeros(1, block_size, hidden_size),
            loss_mask=torch.ones(1, block_size),
            lambda_base=0.0,
        )
        self.assertIsInstance(metrics["lambda_base"], float)
        loss.backward()

        for module in (draft.prefix_gru, draft.embed_proj):
            gradients = [parameter.grad for parameter in module.parameters()]
            self.assertTrue(all(gradient is not None for gradient in gradients))
            self.assertGreater(
                sum(gradient.abs().sum().item() for gradient in gradients),
                0.0,
            )

    def test_chunked_objective_matches_full_loss_metrics_and_gradients(self):
        from specforge.algorithms.common.dflash_family_model import OnlineDominoModel

        hidden_size, vocab_size, block_size = 4, 7, 4
        for shift_label in (False, True):
            with self.subTest(shift_label=shift_label):
                torch.manual_seed(19)
                draft = _bare_domino(
                    hidden_size=hidden_size,
                    gru_hidden_size=3,
                    embedding_size=2,
                    vocab_size=vocab_size,
                    block_size=block_size,
                )
                draft.shift_label = shift_label
                target_head = nn.Linear(hidden_size, vocab_size, bias=False)
                target_embedding = nn.Embedding(vocab_size, hidden_size)
                target_head.requires_grad_(False)
                target_embedding.requires_grad_(False)
                full = OnlineDominoModel(
                    draft_model=draft,
                    target_lm_head=target_head,
                    target_embed_tokens=target_embedding,
                    mask_token_id=0,
                    block_size=block_size,
                    attention_backend="sdpa",
                    num_anchors=2,
                    objective_chunk_blocks=0,
                    shift_label=shift_label,
                )
                chunked = copy.deepcopy(full)
                chunked.objective_chunk_blocks = 1

                anchors = torch.tensor([[0, 3]])
                keep_mask = torch.ones(1, 2, dtype=torch.bool)
                full_hidden = torch.randn(
                    1,
                    2 * block_size,
                    hidden_size,
                    requires_grad=True,
                )
                chunked_hidden = full_hidden.detach().clone().requires_grad_()

                def fixed_blocks(output_hidden):
                    def _forward(
                        self,
                        input_ids,
                        hidden_states,
                        loss_mask,
                        max_valid_anchors=None,
                    ):
                        del (
                            self,
                            input_ids,
                            hidden_states,
                            loss_mask,
                            max_valid_anchors,
                        )
                        return anchors, keep_mask, output_hidden

                    return _forward

                full._forward_draft_blocks = MethodType(
                    fixed_blocks(full_hidden),
                    full,
                )
                chunked._forward_draft_blocks = MethodType(
                    fixed_blocks(chunked_hidden),
                    chunked,
                )
                input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 0, 1]])
                loss_mask = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0]])
                hidden_states = torch.zeros(1, input_ids.shape[1], hidden_size)

                full_loss, full_accuracy, full_metrics = full(
                    input_ids=input_ids,
                    hidden_states=hidden_states,
                    loss_mask=loss_mask,
                    lambda_base=0.35,
                )
                chunked_loss, chunked_accuracy, chunked_metrics = chunked(
                    input_ids=input_ids,
                    hidden_states=hidden_states,
                    loss_mask=loss_mask,
                    lambda_base=0.35,
                )

                torch.testing.assert_close(
                    chunked_loss,
                    full_loss,
                    rtol=1e-6,
                    atol=1e-7,
                )
                torch.testing.assert_close(chunked_accuracy, full_accuracy)
                self.assertEqual(chunked_metrics.keys(), full_metrics.keys())
                for name in full_metrics:
                    torch.testing.assert_close(
                        chunked_metrics[name],
                        full_metrics[name],
                        rtol=1e-6,
                        atol=1e-7,
                    )

                full_loss.backward()
                chunked_loss.backward()
                torch.testing.assert_close(
                    chunked_hidden.grad,
                    full_hidden.grad,
                    rtol=1e-6,
                    atol=1e-7,
                )
                full_parameters = dict(full.draft_model.named_parameters())
                chunked_parameters = dict(chunked.draft_model.named_parameters())
                self.assertEqual(full_parameters.keys(), chunked_parameters.keys())
                for name, parameter in full_parameters.items():
                    chunked_parameter = chunked_parameters[name]
                    self.assertEqual(
                        chunked_parameter.grad is None,
                        parameter.grad is None,
                        name,
                    )
                    if parameter.grad is not None:
                        torch.testing.assert_close(
                            chunked_parameter.grad,
                            parameter.grad,
                            rtol=1e-6,
                            atol=1e-7,
                        )

    def _fixed_blocks_model(
        self,
        *,
        anchors,
        keep_mask,
        output_hidden,
        block_size,
        hidden_size,
        vocab_size,
        shift_label=False,
        draft=None,
        head=None,
        embedding=None,
        **model_kwargs,
    ):
        """Build a Domino model whose draft blocks are pinned by the caller."""

        from specforge.algorithms.common.dflash_family_model import OnlineDominoModel

        draft = (
            copy.deepcopy(draft)
            if draft is not None
            else _bare_domino(
                hidden_size=hidden_size,
                gru_hidden_size=3,
                embedding_size=2,
                vocab_size=vocab_size,
                block_size=block_size,
            )
        )
        draft.shift_label = shift_label
        head = (
            head
            if head is not None
            else nn.Linear(hidden_size, vocab_size, bias=False)
        )
        embedding = (
            embedding
            if embedding is not None
            else nn.Embedding(vocab_size, hidden_size)
        )
        head.requires_grad_(False)
        embedding.requires_grad_(False)
        model = OnlineDominoModel(
            draft_model=draft,
            target_lm_head=head,
            target_embed_tokens=embedding,
            mask_token_id=0,
            block_size=block_size,
            attention_backend="sdpa",
            num_anchors=anchors.shape[1],
            shift_label=shift_label,
            **model_kwargs,
        )

        def fixed_draft_blocks(
            self,
            input_ids,
            hidden_states,
            loss_mask,
            max_valid_anchors=None,
        ):
            del hidden_states, loss_mask, max_valid_anchors
            return (
                anchors.to(input_ids.device),
                keep_mask.to(input_ids.device),
                output_hidden,
            )

        model._forward_draft_blocks = MethodType(fixed_draft_blocks, model)
        return model

    def test_l1_tv_requires_target_hidden_states(self):
        hidden_size, vocab_size, block_size = 4, 7, 4
        model = self._fixed_blocks_model(
            anchors=torch.zeros(1, 1, dtype=torch.long),
            keep_mask=torch.ones(1, 1, dtype=torch.bool),
            output_hidden=torch.randn(1, block_size, hidden_size),
            block_size=block_size,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            l1_loss_alpha=0.9,
            final_tv_loss=True,
        )
        with self.assertRaisesRegex(ValueError, "target_last_hidden_states"):
            model(
                input_ids=torch.tensor([[1, 2, 3, 4]]),
                hidden_states=torch.zeros(1, block_size, hidden_size),
                loss_mask=torch.ones(1, block_size),
                lambda_base=0.0,
            )

    def test_strategy_trains_without_teacher_state_when_l1_is_off(self):
        """Regression: a CE-only run must train from three-feature captures."""

        from specforge.runtime.contracts import TrainBatch
        from specforge.training.strategies.base import DominoTrainStrategy

        hidden_size, vocab_size, block_size = 4, 7, 4
        model = self._fixed_blocks_model(
            anchors=torch.tensor([[0, 3]]),
            keep_mask=torch.ones(1, 2, dtype=torch.bool),
            output_hidden=torch.randn(1, 2 * block_size, hidden_size),
            block_size=block_size,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            l1_loss_alpha=0.0,
        )
        batch = TrainBatch(
            sample_ids=["sample-0"],
            strategy="domino",
            tensors={
                "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 0, 1]]),
                "hidden_states": torch.zeros(1, 2 * block_size, hidden_size),
                "loss_mask": torch.ones(1, 2 * block_size),
            },
        )
        output = DominoTrainStrategy(model).forward_loss(batch, None)

        self.assertTrue(torch.isfinite(output.loss).all())
        self.assertEqual(float(output.metrics["final_l1_loss"]), 0.0)
        self.assertEqual(float(output.metrics["base_l1_loss"]), 0.0)

    def test_strategy_surfaces_missing_teacher_state_when_l1_is_on(self):
        from specforge.runtime.contracts import TrainBatch
        from specforge.training.strategies.base import DominoTrainStrategy

        hidden_size, vocab_size, block_size = 4, 7, 4
        model = self._fixed_blocks_model(
            anchors=torch.tensor([[0, 3]]),
            keep_mask=torch.ones(1, 2, dtype=torch.bool),
            output_hidden=torch.randn(1, 2 * block_size, hidden_size),
            block_size=block_size,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            l1_loss_alpha=0.9,
        )
        batch = TrainBatch(
            sample_ids=["sample-0"],
            strategy="domino",
            tensors={
                "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 0, 1]]),
                "hidden_states": torch.zeros(1, 2 * block_size, hidden_size),
                "loss_mask": torch.ones(1, 2 * block_size),
            },
        )

        with self.assertRaisesRegex(ValueError, "target_last_hidden_states"):
            DominoTrainStrategy(model).forward_loss(batch, None)

    def test_l1_tv_matches_reference_teacher_distribution(self):
        """L1 is the true distributional TV against the captured target state."""

        torch.manual_seed(5)
        hidden_size, vocab_size, block_size = 4, 7, 4
        anchors = torch.tensor([[0, 3]])
        keep_mask = torch.ones(1, 2, dtype=torch.bool)
        draft_hidden = torch.randn(1, 2 * block_size, hidden_size)
        target_hidden = torch.randn(1, 8, hidden_size)
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 0, 1]])
        loss_mask = torch.ones(1, 8)
        head = nn.Linear(hidden_size, vocab_size, bias=False)
        model = self._fixed_blocks_model(
            anchors=anchors,
            keep_mask=keep_mask,
            output_hidden=draft_hidden,
            block_size=block_size,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            head=head,
            ce_loss_alpha=1.0,
            l1_loss_alpha=0.5,
            base_tv_loss=False,
            final_tv_loss=True,
        )
        # A zero correction projection makes the corrected logits equal the
        # base logits, so the expected draft distribution is closed form.
        with torch.no_grad():
            model.draft_model.embed_proj[2].weight.zero_()

        loss, _accuracy, metrics = model(
            input_ids=input_ids,
            hidden_states=torch.zeros(1, 2 * block_size, hidden_size),
            loss_mask=loss_mask,
            target_last_hidden_states=target_hidden,
            lambda_base=0.0,
        )

        label_offsets = torch.arange(0, block_size).view(1, 1, -1)
        label_indices = anchors.unsqueeze(-1) + label_offsets
        safe_indices = label_indices.clamp(max=input_ids.shape[1] - 1)
        gathered = target_hidden[
            0,
            (safe_indices - 1).clamp(min=0).reshape(-1),
            :,
        ].reshape(1, anchors.shape[1], block_size, hidden_size)
        weights = keep_mask.unsqueeze(-1).expand(-1, -1, block_size).float()
        weights = weights * (label_indices < input_ids.shape[1]).float()
        weights = weights * (label_offsets > 0).float()
        weights = weights * torch.gather(
            loss_mask.unsqueeze(1).expand(-1, anchors.shape[1], -1),
            2,
            safe_indices,
        )
        with torch.no_grad():
            draft_logits = head(
                draft_hidden.reshape(1, -1, hidden_size)
            ).reshape(1, anchors.shape[1], block_size, -1)
            teacher_logits = head(
                gathered.reshape(1, -1, hidden_size)
            ).reshape(1, anchors.shape[1], block_size, -1)
            l1_per_token = (
                torch.softmax(draft_logits.float(), dim=-1)
                - torch.softmax(teacher_logits.float(), dim=-1)
            ).abs().sum(dim=-1)
        denominator = weights.sum() + 1e-6
        expected_l1 = (l1_per_token * weights).sum() / denominator

        self.assertAlmostEqual(
            float(metrics["final_l1_loss"]),
            float(expected_l1),
            places=6,
        )
        # CE-only on the base path: lambda_base=0 keeps the corrected objective.
        self.assertAlmostEqual(
            float(loss.detach()),
            float(metrics["final_loss"]) + 0.5 * float(expected_l1),
            places=6,
        )

    def test_l1_tv_toggles_select_base_and_corrected_paths(self):
        torch.manual_seed(6)
        hidden_size, vocab_size, block_size = 4, 7, 4
        anchors = torch.tensor([[0, 3]])
        keep_mask = torch.ones(1, 2, dtype=torch.bool)
        draft_hidden = torch.randn(1, 2 * block_size, hidden_size)
        target_hidden = torch.randn(1, 8, hidden_size)
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 0, 1]])
        loss_mask = torch.ones(1, 8)
        head = nn.Linear(hidden_size, vocab_size, bias=False)
        embedding = nn.Embedding(vocab_size, hidden_size)
        prototype = _bare_domino(
            hidden_size=hidden_size,
            gru_hidden_size=3,
            embedding_size=2,
            vocab_size=vocab_size,
            block_size=block_size,
        )

        def run(**overrides):
            model = self._fixed_blocks_model(
                anchors=anchors,
                keep_mask=keep_mask,
                output_hidden=draft_hidden,
                block_size=block_size,
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                draft=prototype,
                head=head,
                embedding=embedding,
                ce_loss_alpha=1.0,
                l1_loss_alpha=0.25,
                **overrides,
            )
            return model(
                input_ids=input_ids,
                hidden_states=torch.zeros(1, 2 * block_size, hidden_size),
                loss_mask=loss_mask,
                target_last_hidden_states=target_hidden,
                lambda_base=0.4,
            )

        ce_loss, _acc, ce_only = run(base_tv_loss=False, final_tv_loss=False)
        base_loss_value, _acc, base_only = run(
            base_tv_loss=True, final_tv_loss=False
        )
        final_loss_value, _acc, final_only = run(
            base_tv_loss=False, final_tv_loss=True
        )
        both_loss, _acc, both = run(base_tv_loss=True, final_tv_loss=True)

        # A disabled path reports no L1 telemetry; an enabled one reports it.
        self.assertEqual(float(ce_only["base_l1_loss"]), 0.0)
        self.assertEqual(float(ce_only["final_l1_loss"]), 0.0)
        self.assertGreater(float(base_only["base_l1_loss"]), 0.0)
        self.assertEqual(float(base_only["final_l1_loss"]), 0.0)
        self.assertEqual(float(final_only["base_l1_loss"]), 0.0)
        self.assertGreater(float(final_only["final_l1_loss"]), 0.0)
        self.assertGreater(float(both["base_l1_loss"]), 0.0)
        self.assertGreater(float(both["final_l1_loss"]), 0.0)

        lambda_base, ce_alpha, l1_alpha = 0.4, 1.0, 0.25
        final_scale = 1.0 - lambda_base
        base_scale = lambda_base

        def mixed(metrics, path):
            return ce_alpha * float(metrics[f"{path}_loss"])

        ce_only_base = base_scale * mixed(ce_only, "base")
        ce_only_final = final_scale * mixed(ce_only, "final")
        self.assertAlmostEqual(
            float(ce_loss.detach()),
            ce_only_final + ce_only_base,
            places=6,
        )
        final_only_corrected = mixed(final_only, "final") + l1_alpha * float(
            final_only["final_l1_loss"]
        )
        self.assertAlmostEqual(
            float(final_loss_value.detach()),
            final_scale * final_only_corrected
            + base_scale * mixed(final_only, "base"),
            places=6,
        )
        base_only_base = mixed(base_only, "base") + l1_alpha * float(
            base_only["base_l1_loss"]
        )
        self.assertAlmostEqual(
            float(base_loss_value.detach()),
            final_scale * mixed(base_only, "final")
            + base_scale * base_only_base,
            places=6,
        )
        expected_both = final_scale * (
            mixed(both, "final") + l1_alpha * float(both["final_l1_loss"])
        ) + base_scale * (
            mixed(both, "base") + l1_alpha * float(both["base_l1_loss"])
        )
        self.assertAlmostEqual(float(both_loss.detach()), expected_both, places=6)

    def test_chunked_l1_tv_matches_full_objective_and_gradients(self):
        from specforge.algorithms.common.dflash_family_model import OnlineDominoModel

        torch.manual_seed(23)
        hidden_size, vocab_size, block_size = 4, 7, 4
        anchors = torch.tensor([[0, 3]])
        keep_mask = torch.ones(1, 2, dtype=torch.bool)
        target_hidden = torch.randn(1, 8, hidden_size)
        head = nn.Linear(hidden_size, vocab_size, bias=False)
        embedding = nn.Embedding(vocab_size, hidden_size)
        prototype = _bare_domino(
            hidden_size=hidden_size,
            gru_hidden_size=3,
            embedding_size=2,
            vocab_size=vocab_size,
            block_size=block_size,
        )

        def build(chunk_blocks, output_hidden):
            return self._fixed_blocks_model(
                anchors=anchors,
                keep_mask=keep_mask,
                output_hidden=output_hidden,
                block_size=block_size,
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                draft=prototype,
                head=head,
                embedding=embedding,
                objective_chunk_blocks=chunk_blocks,
                ce_loss_alpha=1.0,
                l1_loss_alpha=0.75,
                base_tv_loss=True,
                final_tv_loss=True,
            )

        full_hidden = torch.randn(1, 2 * block_size, hidden_size, requires_grad=True)
        chunked_hidden = full_hidden.detach().clone().requires_grad_()
        full = build(0, full_hidden)
        chunked = build(1, chunked_hidden)

        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 0, 1]])
        loss_mask = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0]])
        hidden_states = torch.zeros(1, input_ids.shape[1], hidden_size)

        full_loss, full_accuracy, full_metrics = full(
            input_ids=input_ids,
            hidden_states=hidden_states,
            loss_mask=loss_mask,
            target_last_hidden_states=target_hidden,
            lambda_base=0.35,
        )
        chunked_loss, chunked_accuracy, chunked_metrics = chunked(
            input_ids=input_ids,
            hidden_states=hidden_states,
            loss_mask=loss_mask,
            target_last_hidden_states=target_hidden,
            lambda_base=0.35,
        )
        torch.testing.assert_close(chunked_loss, full_loss, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(chunked_accuracy, full_accuracy)
        self.assertEqual(chunked_metrics.keys(), full_metrics.keys())
        for name in full_metrics:
            torch.testing.assert_close(
                chunked_metrics[name],
                full_metrics[name],
                rtol=1e-6,
                atol=1e-7,
            )
        self.assertGreater(float(full_metrics["final_l1_loss"]), 0.0)
        self.assertGreater(float(full_metrics["base_l1_loss"]), 0.0)

        full_loss.backward()
        chunked_loss.backward()
        torch.testing.assert_close(
            chunked_hidden.grad,
            full_hidden.grad,
            rtol=1e-6,
            atol=1e-7,
        )
        full_parameters = dict(full.draft_model.named_parameters())
        chunked_parameters = dict(chunked.draft_model.named_parameters())
        for name, parameter in full_parameters.items():
            chunked_parameter = chunked_parameters[name]
            self.assertEqual(
                chunked_parameter.grad is None,
                parameter.grad is None,
                name,
            )
            if parameter.grad is not None:
                torch.testing.assert_close(
                    chunked_parameter.grad,
                    parameter.grad,
                    rtol=1e-6,
                    atol=1e-7,
                )

    @unittest.skipUnless(torch.cuda.is_available(), "Triton loss requires CUDA")
    def test_fused_loss_matches_standard_model_path(self):
        from specforge.algorithms.common.dflash_family_model import OnlineDominoModel

        torch.manual_seed(19)
        hidden_size, vocab_size, block_size = 4, 7, 4
        draft = _bare_domino(
            hidden_size=hidden_size,
            gru_hidden_size=3,
            embedding_size=2,
            vocab_size=vocab_size,
            block_size=block_size,
        )
        model = OnlineDominoModel(
            draft_model=draft,
            target_lm_head=nn.Linear(hidden_size, vocab_size, bias=False),
            target_embed_tokens=nn.Embedding(vocab_size, hidden_size),
            mask_token_id=0,
            block_size=block_size,
            attention_backend="sdpa",
            num_anchors=2,
            objective_chunk_blocks=1,
            shift_label=False,
        ).cuda()
        fixed_hidden = torch.randn(
            1, 2 * block_size, hidden_size, device="cuda", requires_grad=True
        )

        def fixed_draft_blocks(
            self,
            input_ids,
            hidden_states,
            loss_mask,
            max_valid_anchors=None,
        ):
            del hidden_states, loss_mask, max_valid_anchors
            return (
                torch.tensor([[0, 4]], device=input_ids.device),
                torch.ones(1, 2, dtype=torch.bool, device=input_ids.device),
                fixed_hidden,
            )

        model._forward_draft_blocks = MethodType(fixed_draft_blocks, model)
        inputs = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 0, 1]], device="cuda"),
            "hidden_states": torch.zeros(1, 2 * block_size, hidden_size, device="cuda"),
            "loss_mask": torch.ones(1, 2 * block_size, device="cuda"),
            "lambda_base": 0.25,
        }
        model._use_fused_domino_ce = False
        standard = model(**inputs)
        standard[0].backward()
        standard_hidden_grad = fixed_hidden.grad.clone()
        standard_parameter_grads = {
            name: None if parameter.grad is None else parameter.grad.clone()
            for name, parameter in model.named_parameters()
        }
        fixed_hidden.grad = None
        model.zero_grad(set_to_none=True)

        model._use_fused_domino_ce = True
        fused = model(**inputs)
        fused[0].backward()

        torch.testing.assert_close(fused[0], standard[0])
        torch.testing.assert_close(fused[1], standard[1])
        for key in standard[2]:
            torch.testing.assert_close(fused[2][key], standard[2][key])

        torch.testing.assert_close(
            fixed_hidden.grad, standard_hidden_grad, rtol=1e-4, atol=1e-4
        )
        for name, parameter in model.named_parameters():
            standard_grad = standard_parameter_grads[name]
            self.assertEqual(parameter.grad is None, standard_grad is None, name)
            if standard_grad is not None:
                torch.testing.assert_close(
                    parameter.grad, standard_grad, rtol=1e-4, atol=1e-4
                )

    def test_npu_bf16_gru_gradients_reach_registered_weights(self):
        torch.manual_seed(7)
        model = _bare_domino().to(dtype=torch.bfloat16)
        inputs = torch.randn(2, 5, 4, dtype=torch.bfloat16)

        with patch(
            "specforge.modeling.draft.dflash.get_device_type", return_value="npu"
        ):
            output = model._run_gru(inputs)
            output.float().square().mean().backward()

        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertFalse(hasattr(model, "_gru_fp16"))
        for parameter in model.prefix_gru.parameters():
            self.assertEqual(parameter.dtype, torch.bfloat16)
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(parameter.grad.abs().sum().item(), 0.0)

    def test_generation_is_sensitive_to_gru_and_projection_weights(self):
        hidden_size, vocab_size, block_size = 2, 5, 4
        model = _bare_domino(
            hidden_size=hidden_size,
            gru_hidden_size=1,
            embedding_size=1,
            vocab_size=vocab_size,
            block_size=block_size,
        )
        target = _TinyTarget(vocab_size, hidden_size)

        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            for parameter in target.parameters():
                parameter.zero_()
            # The anchor and the correction-selected token feed a positive GRU
            # candidate. The projection reads only that GRU state and boosts id 3.
            target.model.embed_tokens.weight[1, 0] = 1.0
            target.model.embed_tokens.weight[3, 0] = 1.0
            model.prefix_gru.weight_ih_l0[2, 0] = 2.0
            model.embed_proj[0].weight[0, hidden_size] = 4.0
            model.embed_proj[2].weight[3, 0] = 4.0

        draft_hidden = torch.zeros(1, block_size, hidden_size)
        block_ids = torch.tensor([[1, 4, 4, 4]])
        corrected = model._sample_draft_tokens(target, draft_hidden, block_ids)
        torch.testing.assert_close(corrected, torch.tensor([[3, 3, 3]]))

        gru_weight = model.prefix_gru.weight_ih_l0.detach().clone()
        with torch.no_grad():
            model.prefix_gru.weight_ih_l0.zero_()
        no_gru_correction = model._sample_draft_tokens(target, draft_hidden, block_ids)
        torch.testing.assert_close(no_gru_correction, torch.tensor([[0, 0, 0]]))

        with torch.no_grad():
            model.prefix_gru.weight_ih_l0.copy_(gru_weight)
            model.embed_proj[2].weight.zero_()
        no_projection = model._sample_draft_tokens(target, draft_hidden, block_ids)
        torch.testing.assert_close(no_projection, torch.tensor([[0, 0, 0]]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
