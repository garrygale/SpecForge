import math
import os
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

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


def _skip_worker(rank, world_size, init_file, results, enabled, poison_ranks):
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        model, optimizer = _make_optimizer()
        optimizer.configure_grad_norm_reduction(enabled=enabled)
        weight_before = model.weight.detach().clone()

        grad = torch.full_like(model.weight, 0.1)
        if rank in poison_ranks:
            grad[0, 0] = float("inf")
        model.weight.grad = grad

        norm = optimizer.step()
        results[f"norm-{rank}"] = norm.item()
        results[f"unchanged-{rank}"] = bool(torch.equal(model.weight, weight_before))
        results[f"grad-cleared-{rank}"] = model.weight.grad is None
        results[f"adam-empty-{rank}"] = len(optimizer.optimizer.state) == 0

        model.weight.grad = torch.full_like(model.weight, 0.2)
        results[f"clean-{rank}"] = optimizer.step().item()
        results[f"final-{rank}"] = model.weight.detach().clone()
    finally:
        dist.destroy_process_group()


class TestNonFiniteGradSkipDistributed(unittest.TestCase):
    def _spawn(self, enabled, poison_ranks):
        world_size = 2
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = os.path.join(tmpdir, "init")
            manager = mp.Manager()
            results = manager.dict()
            mp.spawn(
                _skip_worker,
                args=(world_size, init_file, results, enabled, poison_ranks),
                nprocs=world_size,
                join=True,
            )
        return results

    def test_sharded_poison_on_one_rank_skips_everywhere(self):
        # SHARD_GRAD_OP/FULL_SHARD config: the norm is all-reduced, so a poison
        # grad on one rank's shard must flip every rank to the same decision.
        results = self._spawn(enabled=True, poison_ranks={0})
        for rank in range(2):
            self.assertTrue(math.isinf(results[f"norm-{rank}"]))
            self.assertTrue(results[f"unchanged-{rank}"])
            self.assertTrue(results[f"grad-cleared-{rank}"])
            self.assertTrue(results[f"adam-empty-{rank}"])
        self.assertEqual(results["clean-0"], results["clean-1"])
        torch.testing.assert_close(results["final-0"], results["final-1"])

    def test_replicated_poison_on_all_ranks_skips_everywhere(self):
        # NO_SHARD runs under DDP with reduction disabled: grads are identical
        # on every rank, so an inf grad is inf everywhere and the local-norm
        # decision still agrees.
        results = self._spawn(enabled=False, poison_ranks={0, 1})
        for rank in range(2):
            self.assertTrue(math.isinf(results[f"norm-{rank}"]))
            self.assertTrue(results[f"unchanged-{rank}"])
            self.assertTrue(results[f"adam-empty-{rank}"])
        torch.testing.assert_close(results["final-0"], results["final-1"])


if __name__ == "__main__":
    unittest.main()
