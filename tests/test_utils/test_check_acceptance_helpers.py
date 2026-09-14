"""Local-data tests for acceptance evaluator helpers."""

from __future__ import annotations

import argparse
import dataclasses
import os
import types
import unittest
from unittest import mock

from inference import check_acceptance
from inference.check_acceptance import (
    _assert_tp_sharded,
    _log_path,
    _localize_outputs,
    _parallel_sizes,
    _split_world,
    aggregate_stats,
    load_humaneval,
    load_math500,
    load_mbpp,
)

_INFERENCE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "inference",
)

_STATS = {
    "per_problem": [{"task_id": "HumanEval/0", "mean_acceptance_length": 3.0}],
    "overall_simple": 3.0,
    "overall_weighted": 3.0,
    "valid_count": 1,
    "total_problems": 4,
    "num_evaluated": 2,
    "log_file": "unused.jsonl",
    "per_position_accuracy": None,
}


class _FakeReplicate:
    def __repr__(self):  # pragma: no cover - debug helper
        return "Replicate()"


class _FakeDTensor:
    """Stand-in for ``torch.distributed.tensor.DTensor``."""

    def __init__(self, local, placements):
        self._local = local
        self.placements = placements
        self.gathered = False

    def to_local(self):
        return self._local

    def full_tensor(self):
        self.gathered = True
        return self._local


@dataclasses.dataclass
class _FakeOutput:
    logits: object = None
    hidden_states: object = None


class AcceptanceEvaluatorHelpersTest(unittest.TestCase):
    def test_humaneval_loader(self):
        problems = load_humaneval(
            os.path.join(_INFERENCE_DIR, "human-eval-v2-20210705.jsonl")
        )
        self.assertGreater(len(problems), 100)
        self.assertIn("task_id", problems[0])

    def test_math500_loader(self):
        problems = load_math500(
            os.path.join(_INFERENCE_DIR, "math500-test.jsonl")
        )
        self.assertGreater(len(problems), 100)

    def test_mbpp_loader(self):
        problems = load_mbpp(
            os.path.join(_INFERENCE_DIR, "sanitized-mbpp.json")
        )
        self.assertGreater(len(problems), 50)
        self.assertIn("test_list", problems[0])

    def test_aggregate_excludes_invalid(self):
        rows = [
            {"mean_acceptance_length": 4.0, "num_complete_blocks": 2},
            {"mean_acceptance_length": None, "num_complete_blocks": 0},
        ]
        simple, weighted = aggregate_stats(rows)
        self.assertEqual(simple, 4.0)
        self.assertEqual(weighted, 4.0)


class AcceptanceParallelismTest(unittest.TestCase):
    def test_split_world_keeps_tp_ranks_contiguous(self):
        self.assertEqual(_split_world(2, 2, 0), (0, 0))
        self.assertEqual(_split_world(2, 2, 1), (0, 1))
        self.assertEqual(_split_world(2, 2, 2), (1, 0))
        self.assertEqual(_split_world(2, 2, 3), (1, 1))
        self.assertEqual(_split_world(4, 1, 3), (3, 0))

    def test_split_world_rejects_bad_sizes_and_ranks(self):
        with self.assertRaises(ValueError):
            _split_world(0, 1, 0)
        with self.assertRaises(ValueError):
            _split_world(2, 2, 4)

    def test_log_path_suffixes_only_distributed_runs(self):
        self.assertTrue(
            _log_path("260910000000", 1, 0).endswith(
                "260910000000_acceptance_lengths.jsonl"
            )
        )
        self.assertTrue(
            _log_path("260910000000", 4, 2).endswith(
                "260910000000_acceptance_lengths_dp2.jsonl"
            )
        )

    def test_parallel_sizes_default_and_deprecated_alias(self):
        self.assertEqual(
            _parallel_sizes(argparse.Namespace(dp=None, tp=None, num_npus=None)),
            (1, 1),
        )
        self.assertEqual(
            _parallel_sizes(argparse.Namespace(dp=None, tp=2, num_npus=4)),
            (4, 2),
        )
        with self.assertRaises(SystemExit):
            _parallel_sizes(argparse.Namespace(dp=2, tp=1, num_npus=4))
        with self.assertRaises(SystemExit):
            _parallel_sizes(argparse.Namespace(dp=2, tp=0, num_npus=None))


class AcceptanceLauncherTest(unittest.TestCase):
    def test_main_spawns_one_worker_per_npu(self):
        processes = []

        def _fake_popen(command, env=None):
            process = mock.Mock()
            process.wait.return_value = 0
            processes.append((command, env))
            return process

        with (
            mock.patch.object(check_acceptance.subprocess, "Popen", _fake_popen),
            mock.patch.object(check_acceptance, "_inside_job", return_value=False),
        ):
            code = check_acceptance.main(
                [
                    "--draft-path",
                    "draft",
                    "--target-path",
                    "target",
                    "--dp",
                    "2",
                    "--tp",
                    "1",
                ]
            )

        self.assertEqual(code, 0)
        self.assertEqual([env["RANK"] for _, env in processes], ["0", "1"])
        self.assertEqual([env["LOCAL_RANK"] for _, env in processes], ["0", "1"])
        self.assertEqual({env["WORLD_SIZE"] for _, env in processes}, {"2"})
        for rank, (command, _) in enumerate(processes):
            self.assertIn("--dp", command)
            self.assertEqual(command[command.index("--dp") + 1], "2")
            self.assertEqual(command[command.index("--tp") + 1], "1")
            self.assertEqual(command[command.index("--npu-id") + 1], str(rank))

    def test_deprecated_num_npus_still_launches_workers(self):
        processes = []

        def _fake_popen(command, env=None):
            process = mock.Mock()
            process.wait.return_value = 0
            processes.append(command)
            return process

        with (
            mock.patch.object(check_acceptance.subprocess, "Popen", _fake_popen),
            mock.patch.object(check_acceptance, "_inside_job", return_value=False),
        ):
            code = check_acceptance.main(
                [
                    "--draft-path",
                    "draft",
                    "--target-path",
                    "target",
                    "--num-npus",
                    "3",
                ]
            )

        self.assertEqual(code, 0)
        self.assertEqual(len(processes), 3)
        for command in processes:
            self.assertEqual(command[command.index("--dp") + 1], "3")
            self.assertEqual(command[command.index("--tp") + 1], "1")

    def test_worker_ranks_dispatch_dp_and_tp(self):
        with (
            mock.patch.dict(
                os.environ, {"RANK": "3", "WORLD_SIZE": "4"}, clear=False
            ),
            mock.patch.object(
                check_acceptance.dist, "is_initialized", return_value=False
            ),
            mock.patch.object(
                check_acceptance, "run_acceptance_check", return_value=_STATS
            ) as run_check,
            mock.patch.object(check_acceptance, "_init_tp_process_group"),
        ):
            code = check_acceptance.main(
                [
                    "--draft-path",
                    "draft",
                    "--target-path",
                    "target",
                    "--dp",
                    "2",
                    "--tp",
                    "2",
                    "--npu-id",
                    "3",
                ]
            )

        self.assertEqual(code, 0)
        kwargs = run_check.call_args.kwargs
        self.assertEqual(kwargs["dp_size"], 2)
        self.assertEqual(kwargs["tp_size"], 2)
        self.assertEqual(kwargs["dp_rank"], 1)
        self.assertEqual(kwargs["tp_rank"], 1)
        self.assertEqual(kwargs["npu_id"], 3)

    def test_worker_errors_when_env_world_size_disagrees(self):
        with (
            mock.patch.dict(
                os.environ, {"RANK": "0", "WORLD_SIZE": "8"}, clear=False
            ),
            mock.patch.object(
                check_acceptance.dist, "is_initialized", return_value=False
            ),
        ):
            with self.assertRaises(SystemExit):
                check_acceptance.main(
                    [
                        "--draft-path",
                        "draft",
                        "--target-path",
                        "target",
                        "--dp",
                        "2",
                        "--tp",
                        "1",
                        "--npu-id",
                        "0",
                    ]
                )


class AcceptanceTensorParallelOutputTest(unittest.TestCase):
    def setUp(self):
        self._dtensor_patch = mock.patch.object(
            check_acceptance,
            "_dtensor_types",
            return_value=(_FakeDTensor, _FakeReplicate),
        )
        self._dtensor_patch.start()
        self.addCleanup(self._dtensor_patch.stop)
        self._warnings_patch = mock.patch.object(
            check_acceptance, "_gathered_placement_warnings", set()
        )
        self._warnings_patch.start()
        self.addCleanup(self._warnings_patch.stop)

    def test_replicated_dtensors_are_unwrapped_without_collectives(self):
        dtensor = _FakeDTensor([1, 2, 3], [_FakeReplicate()])
        self.assertEqual(_localize_outputs(dtensor), [1, 2, 3])
        self.assertFalse(dtensor.gathered)

    def test_sharded_dtensors_are_gathered_with_a_warning(self):
        dtensor = _FakeDTensor([1, 2, 3], ["Shard(dim=0)"])
        with mock.patch("builtins.print") as fake_print:
            self.assertEqual(_localize_outputs(dtensor), [1, 2, 3])
        self.assertTrue(dtensor.gathered)
        self.assertTrue(fake_print.called)

    def test_model_outputs_and_sequences_are_localized_in_place(self):
        output = _FakeOutput(
            logits=_FakeDTensor("logits", [_FakeReplicate()]),
            hidden_states=(
                _FakeDTensor("h0", [_FakeReplicate()]),
                None,
            ),
        )
        localized = _localize_outputs(output)
        self.assertIs(localized, output)
        self.assertEqual(output.logits, "logits")
        self.assertEqual(output.hidden_states, ("h0", None))

    def test_plain_values_pass_through(self):
        self.assertEqual(_localize_outputs([1, "two", None]), [1, "two", None])
        self.assertIsNone(_localize_outputs(None))


class AcceptanceTensorParallelGuardTest(unittest.TestCase):
    def setUp(self):
        self._dtensor_patch = mock.patch.object(
            check_acceptance,
            "_dtensor_types",
            return_value=(_FakeDTensor, _FakeReplicate),
        )
        self._dtensor_patch.start()
        self.addCleanup(self._dtensor_patch.stop)

    @staticmethod
    def _model(placements, mesh_size):
        parameter = _FakeDTensor("weight", placements)
        ranks = types.SimpleNamespace(numel=lambda: mesh_size)
        parameter.device_mesh = types.SimpleNamespace(mesh=ranks)
        return types.SimpleNamespace(parameters=lambda: iter([parameter]))

    def test_sharding_over_the_requested_tp_size_passes(self):
        model = self._model(["Shard(dim=0)"], 2)
        _assert_tp_sharded(model, 2)

    def test_sharding_over_the_wrong_size_is_rejected(self):
        model = self._model(["Shard(dim=0)"], 4)
        with self.assertRaises(RuntimeError) as caught:
            _assert_tp_sharded(model, 2)
        self.assertIn("sharded over 4 ranks", str(caught.exception))

    def test_replicated_only_model_warns_instead_of_failing(self):
        model = self._model([_FakeReplicate()], 1)
        with mock.patch("builtins.print") as fake_print:
            _assert_tp_sharded(model, 2)
        self.assertTrue(fake_print.called)


if __name__ == "__main__":
    unittest.main()
