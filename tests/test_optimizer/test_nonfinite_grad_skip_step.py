import unittest

import torch

from specforge.optimizer import BF16Optimizer


def _make_optimizer(seed=0, **kwargs):
    torch.manual_seed(seed)
    model = torch.nn.Linear(8, 8, bias=False)
    return model, BF16Optimizer(model, lr=1e-3, max_grad_norm=0.5, **kwargs)


class TestNonFiniteGradSkipStep(unittest.TestCase):
    def _assert_skipped(self, model, optimizer, norm, bad_value):
        self.assertTrue(
            torch.isnan(norm).all() if bad_value != bad_value else torch.isinf(norm).all()
        )
        for master, reference in zip(optimizer.fp32_params, optimizer._masters_before):
            torch.testing.assert_close(master, reference)
            self.assertIsNone(master.grad)
        torch.testing.assert_close(model.weight, optimizer._weight_before)
        self.assertIsNone(model.weight.grad)
        self.assertEqual(optimizer.get_learning_rate(), optimizer._lr_before)

    def _run_skip_case(self, bad_value):
        model, optimizer = _make_optimizer()
        optimizer._masters_before = [mp.detach().clone() for mp in optimizer.fp32_params]
        optimizer._weight_before = model.weight.detach().clone()
        optimizer._lr_before = optimizer.get_learning_rate()

        model.weight.grad = torch.full_like(model.weight, bad_value)
        norm = optimizer.step()
        self._assert_skipped(model, optimizer, norm, bad_value)

        # the next clean step must work normally, i.e. the poison was cleared
        model.weight.grad = torch.full_like(model.weight, 0.1)
        norm = optimizer.step()
        self.assertTrue(torch.isfinite(norm).all())

    def test_inf_grad_skips_step(self):
        self._run_skip_case(float("inf"))

    def test_nan_grad_skips_step(self):
        self._run_skip_case(float("nan"))

    def test_finite_grad_still_steps(self):
        model, optimizer = _make_optimizer()
        weight_before = model.weight.detach().clone()

        model.weight.grad = torch.full_like(model.weight, 0.1)
        norm = optimizer.step()

        self.assertTrue(torch.isfinite(norm).all())
        self.assertFalse(torch.equal(model.weight, weight_before))

    def test_clip_grad_norm_leaves_nonfinite_master_grads_alone(self):
        model, optimizer = _make_optimizer()
        for master in optimizer.fp32_params:
            master.grad = torch.full_like(master, float("inf"))

        norm = optimizer._clip_grad_norm()

        self.assertTrue(torch.isinf(norm).all())
        for master in optimizer.fp32_params:
            # 0 * inf would have turned these into NaN
            self.assertTrue(torch.isinf(master.grad).all())


if __name__ == "__main__":
    unittest.main()
