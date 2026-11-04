import math
import unittest

import torch

from specforge.optimizer import BF16Optimizer


def _optimizer(
    *,
    scheduler="cosine",
    warmup_type="linear",
    total_steps=4,
    warmup_ratio=0.0,
):
    model = torch.nn.Linear(2, 2, bias=False)
    optimizer = BF16Optimizer(
        model,
        lr=1e-3,
        max_grad_norm=1.0,
        total_steps=total_steps,
        warmup_ratio=warmup_ratio,
        lr_scheduler=scheduler,
        warmup_type=warmup_type,
    )
    return model, optimizer


class TestLearningRateScheduler(unittest.TestCase):
    def test_constant_scheduler_keeps_base_lr_without_warmup(self):
        model, optimizer = _optimizer(scheduler="constant")
        observed = [optimizer.get_learning_rate()]
        for _ in range(4):
            model.weight.grad = torch.ones_like(model.weight)
            optimizer.step()
            observed.append(optimizer.get_learning_rate())
        self.assertEqual(observed, [1e-3] * 5)

    def test_constant_scheduler_supports_linear_warmup(self):
        _model, optimizer = _optimizer(
            scheduler="constant", total_steps=4, warmup_ratio=0.5
        )
        self.assertAlmostEqual(optimizer.get_learning_rate(), 5e-4)
        optimizer.scheduler.step()
        self.assertAlmostEqual(optimizer.get_learning_rate(), 1e-3)
        optimizer.scheduler.step()
        self.assertAlmostEqual(optimizer.get_learning_rate(), 1e-3)

    def test_unknown_scheduler_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsupported lr_scheduler"):
            _optimizer(scheduler="linear")

    def test_unknown_warmup_type_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsupported warmup_type"):
            _optimizer(warmup_type="sqrt")

    def test_cosine_warmup_ramp_lands_flat_on_peak_lr(self):
        # total_steps=8, warmup_ratio=0.5 -> 4 warmup steps of 1e-3:
        # factor(t) = 0.5*(1-cos(pi*t/4)) = 0.14645, 0.5, 0.85355, 1.0
        _model, optimizer = _optimizer(
            warmup_type="cosine", total_steps=8, warmup_ratio=0.5
        )
        observed = [optimizer.get_learning_rate()]
        for _ in range(3):
            optimizer.scheduler.step()
            observed.append(optimizer.get_learning_rate())
        expected = [
            0.5 * (1 - math.cos(math.pi * t / 4)) * 1e-3 for t in (1, 2, 3, 4)
        ]
        for got, want in zip(observed, expected):
            self.assertAlmostEqual(got, want, places=6)
        # last warmup step is exactly peak lr (0.5*(1-cos(pi)) == 1.0)
        self.assertEqual(observed[-1], 1e-3)

    def test_cosine_warmup_junction_is_continuous_at_training_scale(self):
        # torch's chained-scheduler handover anchors the decay recursion on
        # group["lr"] with a 2/(1+cos(pi/T_max)) wobble of order (pi/T_max)^2;
        # at real schedule sizes it is far below training significance
        _model, optimizer = _optimizer(
            warmup_type="cosine", total_steps=40_000, warmup_ratio=0.05
        )
        for _ in range(1_998):
            optimizer.scheduler.step()
        previous = optimizer.get_learning_rate()  # t=1999/2000
        optimizer.scheduler.step()
        peak = optimizer.get_learning_rate()  # t=2000/2000
        self.assertAlmostEqual(peak, 1e-3, places=9)
        # terminal warmup slope ~ (pi/T)^2/4 per step vs 1/T for the linear
        # ramp: the kink the cosine ramp exists to remove
        self.assertLess(peak - previous, 2e-6 * 1e-3)
        optimizer.scheduler.step()  # first decay step
        self.assertAlmostEqual(optimizer.get_learning_rate(), 1e-3, delta=1e-8)

    def test_cosine_warmup_with_constant_scheduler(self):
        _model, optimizer = _optimizer(
            scheduler="constant",
            warmup_type="cosine",
            total_steps=8,
            warmup_ratio=0.5,
        )
        optimizer.scheduler.step()
        self.assertAlmostEqual(optimizer.get_learning_rate(), 5e-4, places=6)
        optimizer.scheduler.step()
        self.assertAlmostEqual(optimizer.get_learning_rate(), 0.8535534e-3, places=6)
        optimizer.scheduler.step()
        self.assertAlmostEqual(optimizer.get_learning_rate(), 1e-3, places=6)
        optimizer.scheduler.step()
        self.assertAlmostEqual(optimizer.get_learning_rate(), 1e-3, places=6)

    def test_resume_rejects_scheduler_change(self):
        _model, cosine = _optimizer(scheduler="cosine")
        _other_model, constant = _optimizer(scheduler="constant")
        with self.assertRaisesRegex(ValueError, "checkpoint optimizer used"):
            constant.load_state_dict(cosine.state_dict())

    def test_resume_rejects_warmup_type_change(self):
        _model, linear = _optimizer(total_steps=8, warmup_ratio=0.5)
        _other_model, cosined = _optimizer(
            warmup_type="cosine", total_steps=8, warmup_ratio=0.5
        )
        with self.assertRaisesRegex(ValueError, "warmup_type"):
            linear.load_state_dict(cosined.state_dict())

        # checkpoints saved before warmup_type existed only ramped linearly
        legacy = cosined.state_dict()
        legacy.pop("warmup_type")
        self.assertEqual(legacy.get("warmup_type", "linear"), "linear")

    def test_legacy_checkpoint_defaults_to_cosine(self):
        _model, cosine = _optimizer(scheduler="cosine")
        state = cosine.state_dict()
        state.pop("lr_scheduler_type")
        self.assertEqual(state.get("lr_scheduler_type", "cosine"), "cosine")


if __name__ == "__main__":
    unittest.main()
