# coding=utf-8
"""Domino step schedule and offline launcher.

Domino's immutable algorithm registration injects its concrete step provider.
Its loss blends a base loss with a step-decayed weight, so
DominoTrainStrategy reads the StepContext threaded through forward_loss. These
tests prove the lambda schedule logic on CPU and that Domino trains end to end
from precomputed features on GPU.
"""

import os
import tempfile
import unittest

import torch

from specforge.algorithms.builtin import builtin_algorithm_registry

CUDA = torch.cuda.is_available()
ALGORITHM = builtin_algorithm_registry().resolve("domino")


class TestDominoLambdaSchedule(unittest.TestCase):
    """CPU: the StepContext-driven lambda_base schedule (no model needed)."""

    def test_lambda_base_decays_over_total_steps(self):
        from specforge.training.strategies.base import DominoTrainStrategy, StepContext

        # _lambda_base reads only ctx + lambda_start/decay_ratio, not the model.
        s = DominoTrainStrategy(None, lambda_start=1.0, decay_ratio=0.5)
        self.assertAlmostEqual(
            s._lambda_base(StepContext(global_step=0, total_steps=10)), 1.0
        )
        # decay_steps = total_steps*decay_ratio = 5; step 5+ -> 0
        self.assertEqual(
            s._lambda_base(StepContext(global_step=5, total_steps=10)), 0.0
        )
        self.assertEqual(
            s._lambda_base(StepContext(global_step=10, total_steps=10)), 0.0
        )
        # no schedule info -> pure final loss
        self.assertEqual(s._lambda_base(None), 0.0)
        self.assertEqual(
            s._lambda_base(StepContext(global_step=3, total_steps=None)), 0.0
        )

    def test_lambda_base_cosine_shape(self):
        from specforge.training.strategies.base import (
            DominoTrainStrategy,
            StepContext,
            lambda_base_schedule,
        )

        # 0.5*(1+cos(pi*p)) over the same decay window (decay_steps=5): same
        # endpoints, base loss held above linear in the first half and below
        # in the second, equal area underneath.
        total, ratio = 10, 0.5
        s = DominoTrainStrategy(
            None, lambda_start=1.0, decay_ratio=ratio, decay_type="cosine"
        )
        self.assertAlmostEqual(
            s._lambda_base(StepContext(global_step=0, total_steps=total)), 1.0
        )
        # step 1 (p=0.2) and step 2 (p=0.4): cosine above linear
        for step in (1, 2):
            self.assertGreater(
                s._lambda_base(StepContext(global_step=step, total_steps=total)),
                lambda_base_schedule(step, total, 1.0, ratio),
            )
        # symmetric second half: cosine below linear
        for step in (3, 4):
            self.assertLess(
                s._lambda_base(StepContext(global_step=step, total_steps=total)),
                lambda_base_schedule(step, total, 1.0, ratio),
            )
        self.assertEqual(
            s._lambda_base(StepContext(global_step=5, total_steps=total)), 0.0
        )
        # dose-matched: equal area under both shapes over the window
        linear_dose = sum(
            lambda_base_schedule(g, total, 1.0, ratio) for g in range(total + 1)
        )
        cosine_dose = sum(
            lambda_base_schedule(g, total, 1.0, ratio, "cosine")
            for g in range(total + 1)
        )
        self.assertAlmostEqual(linear_dose, cosine_dose, places=1)

    def test_lambda_base_schedule_rejects_unknown_decay_type(self):
        from specforge.training.strategies.base import lambda_base_schedule

        with self.assertRaisesRegex(ValueError, "decay_type"):
            lambda_base_schedule(0, 10, 1.0, 0.5, "sqrt")

    def test_lambda_base_default_matches_legacy_linear_values(self):
        from specforge.training.strategies.base import (
            DominoTrainStrategy,
            StepContext,
        )

        default = DominoTrainStrategy(None, lambda_start=1.0, decay_ratio=0.5)
        for step in (0, 1, 2, 3, 5, 9):
            ctx = StepContext(global_step=step, total_steps=10)
            self.assertEqual(
                default._lambda_base(ctx),
                max(0.0, 1.0 - step / 5.0) if step < 5 else 0.0,
            )

    def test_strategy_preserves_every_model_diagnostic(self):
        from specforge.runtime.contracts import TrainBatch
        from specforge.training.strategies.base import DominoTrainStrategy, StepContext

        class DiagnosticModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.0))
                self.model_metrics = {
                    "final_loss": torch.tensor(1.25),
                    "base_loss": torch.tensor(2.5),
                    "base_accuracy": torch.tensor(0.4),
                    "accept_len": torch.tensor(3.0),
                    "base_accept_len": torch.tensor(2.0),
                    "accuracy_denom": torch.tensor(8.0),
                    "lambda_base": torch.tensor(0.5),
                }

            def forward(self, **kwargs):
                self.last_kwargs = kwargs
                return self.weight.square(), torch.tensor(0.75), self.model_metrics

        model = DiagnosticModel()
        strategy = DominoTrainStrategy(model, lambda_start=1.0, decay_ratio=0.5)
        batch = TrainBatch(
            sample_ids=["sample-0"],
            strategy="domino",
            tensors={
                "input_ids": torch.ones(1, 2, dtype=torch.long),
                "hidden_states": torch.ones(1, 2, 2),
                "loss_mask": torch.ones(1, 2),
                "target_last_hidden_states": torch.ones(1, 2, 2),
            },
        )

        output = strategy.forward_loss(
            batch,
            StepContext(global_step=1, total_steps=4),
        )

        self.assertEqual(
            set(output.metrics),
            {*model.model_metrics, "accuracy"},
        )
        for name, value in model.model_metrics.items():
            self.assertIs(output.metrics[name], value)
        self.assertEqual(float(output.metrics["accuracy"]), 0.75)
        self.assertEqual(model.last_kwargs["max_valid_anchors"], 1)
        torch.testing.assert_close(
            model.last_kwargs["target_last_hidden_states"],
            batch.tensors["target_last_hidden_states"],
        )

    def test_global_loss_denominator_toggle_controls_loss_terms(self):
        from specforge.runtime.contracts import TrainBatch
        from specforge.training.strategies.base import DominoTrainStrategy, StepContext

        class DiagnosticModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.0))
                self.model_metrics = {
                    "loss_denominator": torch.tensor(6.0),
                }

            def forward(self, **kwargs):
                return self.weight.square(), torch.tensor(0.5), self.model_metrics

        batch = TrainBatch(
            sample_ids=["sample-0"],
            strategy="domino",
            tensors={
                "input_ids": torch.ones(1, 2, dtype=torch.long),
                "hidden_states": torch.ones(1, 2, 2),
                "loss_mask": torch.ones(1, 2),
            },
        )
        model = DiagnosticModel()

        off = DominoTrainStrategy(model)
        self.assertIsNone(
            off.forward_loss(batch, StepContext(global_step=1, total_steps=4)).loss_terms
        )

        on = DominoTrainStrategy(model, global_loss_denominator=True)
        output = on.forward_loss(
            batch,
            StepContext(global_step=1, total_steps=4),
        )
        numerator, denominator = output.loss_terms
        torch.testing.assert_close(denominator, torch.tensor(6.0))
        torch.testing.assert_close(numerator, model.weight.square() * 6.0)
        # the metric stays visible as telemetry alongside the loss_terms pair
        self.assertIn("loss_denominator", output.metrics)

        class DenominatorFreeModel(DiagnosticModel):
            def forward(self, **kwargs):
                loss, accuracy, _ = DiagnosticModel.forward(self, **kwargs)
                return loss, accuracy, {}

        with self.assertRaisesRegex(ValueError, "loss_denominator"):
            DominoTrainStrategy(
                DenominatorFreeModel(), global_loss_denominator=True
            ).forward_loss(batch, StepContext(global_step=1, total_steps=4))


class _MockFSDPBackend:
    """FSDP stand-in: the boundary backward leaves every rank holding the
    across-rank mean of the accumulated gradients (what FSDP's predivide/
    postdivide factors and DDP's all-reduce both reduce to). The other rank's
    pre-reduce gradient and denominator enter as constants, so one rank runs
    the real controller while the pair is emulated deterministically."""

    name = "mock-fsdp"

    def __init__(self, model, world_size, remote_grad):
        self.model = model
        self.world_size = world_size
        self.remote_grad = remote_grad
        self.scales = []

    def prepare_model(self, model):
        return model

    def backward(self, loss, *, is_boundary=True):
        loss.backward()
        if is_boundary:
            for p, remote in zip(self.model.parameters(), self.remote_grad):
                p.grad.copy_((p.grad + remote) / self.world_size)

    def scale_gradients(self, factor):
        self.scales.append(float(factor))
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.mul_(factor)

    def step(self):
        return torch.tensor(1.0)

    def state_dict(self):
        return {"model": {}, "optimizer": None, "rng": {}}

    def load_state_dict(self, state):
        pass


class TestDominoGlobalLossDenominatorRankAveraging(unittest.TestCase):
    """End-to-end (strategy -> controller -> mock FSDP) token-mean check.

    Rank 1 supervises 100 tokens, rank 2 only 20; the two ranks chase
    different targets so their local means disagree. With the toggle on the
    final gradient must equal autograd of the single global objective
    (den1*loss1 + den2*loss2)/(den1+den2); with it off it stays the rank-mean
    (dloss1 + dloss2)/2, which the imbalance biases.
    """

    DEN1, DEN2 = 100.0, 20.0
    TARGET1, TARGET2 = 2.0, 4.0

    class _RankModel(torch.nn.Module):
        def __init__(self, target, den):
            super().__init__()
            self.w = torch.nn.Parameter(torch.tensor(1.0))
            self.target = float(target)
            self.den = float(den)

        def forward(self, **kwargs):
            loss = 0.5 * (self.w - self.target).square()
            return loss, torch.tensor(0.5), {
                "loss_denominator": torch.tensor(self.den)
            }

    def _run_rank(self, *, toggle):
        from unittest import mock

        from specforge.runtime.contracts import TrainBatch
        from specforge.training.controller import TrainerCore
        from specforge.training.strategies.base import (
            DominoTrainStrategy,
            StepContext,
        )

        # the remote rank backwarded its own contribution before the boundary
        remote = torch.nn.Parameter(torch.tensor(1.0))
        remote_loss = 0.5 * (remote - self.TARGET2).square()
        remote_loss.backward(
            torch.tensor(self.DEN2 if toggle else 1.0)
        )

        model = self._RankModel(self.TARGET1, self.DEN1)
        strategy = DominoTrainStrategy(model, global_loss_denominator=toggle)
        backend = _MockFSDPBackend(
            strategy.trainable_module(), 2, (remote.grad.detach(),)
        )
        core = TrainerCore(strategy, backend)

        batch = TrainBatch(
            sample_ids=["rank-1"],
            strategy="domino",
            tensors={
                "input_ids": torch.ones(1, 2, dtype=torch.long),
                "hidden_states": torch.ones(1, 2, 2),
                "loss_mask": torch.ones(1, 2),
            },
        )

        def add_remote_denominator(tensor, *args, **kwargs):
            tensor.add_(self.DEN2)

        with (
            mock.patch("torch.distributed.is_available", return_value=True),
            mock.patch("torch.distributed.is_initialized", return_value=True),
            mock.patch("torch.distributed.get_world_size", return_value=2),
            mock.patch(
                "torch.distributed.all_reduce",
                side_effect=add_remote_denominator,
            ),
        ):
            core.train_step(batch, StepContext(global_step=1, total_steps=4))
        return model.w.grad.detach(), backend.scales

    def test_toggle_reproduces_the_global_token_mean(self):
        reference = torch.nn.Parameter(torch.tensor(1.0))
        global_mean = (
            0.5 * (reference - self.TARGET1).square() * self.DEN1
            + 0.5 * (reference - self.TARGET2).square() * self.DEN2
        ) / (self.DEN1 + self.DEN2)
        global_mean.backward()

        grad, scales = self._run_rank(toggle=True)

        torch.testing.assert_close(grad, reference.grad)
        # world_size * accumulation_steps / global_denominator
        torch.testing.assert_close(scales, [2.0 / (self.DEN1 + self.DEN2)])

    def test_default_stays_the_biased_rank_mean(self):
        grad, scales = self._run_rank(toggle=False)

        # (dloss1 + dloss2)/2 with dloss = w - target: (-1 - 3)/2
        torch.testing.assert_close(grad, torch.tensor(-2.0))
        self.assertEqual(scales, [])


@unittest.skipUnless(CUDA, "Domino offline launcher path requires CUDA")
class TestDominoOfflineLaunch(unittest.TestCase):
    def test_domino_trains_from_precomputed_dflash_features(self):
        from tests.test_runtime import _fixtures as fx

        fx.build_single_rank_distributed(port="29571")

        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        from specforge.launch import build_offline_runtime
        from specforge.optimizer import BF16Optimizer

        hidden, sequence_length = 64, 32
        workdir = tempfile.mkdtemp(prefix="domino_offline_")
        feature_dir = fx.write_offline_files_dflash(
            os.path.join(workdir, "features"),
            n=4,
            seq=sequence_length,
            hidden=hidden,
            target_hidden=hidden,
        )
        model, width, _target_dir, _layers = fx.build_domino(
            workdir,
            hidden=hidden,
            block_size=4,
            num_anchors=8,
            attention_backend="sdpa",
            # End-to-end coverage for the DSpark-style L1/TV distillation on
            # both the corrected and the base logits.
            ce_loss_alpha=1.0,
            l1_loss_alpha=0.5,
            base_tv_loss=True,
            final_tv_loss=True,
        )
        self.assertEqual(width, hidden)

        def optimizer_factory(module):
            return BF16Optimizer(
                module,
                lr=1e-3,
                max_grad_norm=0.5,
                warmup_ratio=0.0,
                total_steps=2,
            )

        trainer = build_offline_runtime(
            algorithm=ALGORITHM,
            hidden_states_path=feature_dir,
            draft_model=model,
            target_head=None,
            optimizer_factory=optimizer_factory,
            run_id="domino-offline",
            output_dir=os.path.join(workdir, "out"),
            max_len=sequence_length,
            batch_size=1,
            num_epochs=1,
            max_steps=2,
            total_steps=2,
            strategy_kwargs={"lambda_start": 1.0, "decay_ratio": 0.5},
        )

        module = trainer.core.strategy.trainable_module()
        self.assertIsInstance(module, FSDP)
        self.assertEqual(trainer.fit(), 2)
        self.assertTrue(all(torch.isfinite(p).all() for p in module.parameters()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
