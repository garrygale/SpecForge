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
    _dtensor_sharded,
    _log_path,
    _localize_outputs,
    _parallel_sizes,
    _report_tp_sharding,
    _split_world,
    _tp_loader_attempts,
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
    def _model(
        *,
        placements=(),
        tp_size=None,
        plan=None,
        hooked=False,
        mesh_size=None,
    ):
        parameters = []
        if placements:
            parameter = _FakeDTensor("weight", list(placements))
            ranks = types.SimpleNamespace(numel=lambda: mesh_size or 1)
            parameter.device_mesh = types.SimpleNamespace(mesh=ranks)
            parameters.append(parameter)
        module = types.SimpleNamespace(_is_hooked=True) if hooked else types.SimpleNamespace()
        model = types.SimpleNamespace(parameters=lambda: iter(parameters))
        model.modules = lambda: iter([model, module])
        model.tp_plan = {} if plan is None else plan
        if tp_size is not None:
            model._tp_size = tp_size
        if mesh_size is not None:
            ranks = types.SimpleNamespace(numel=lambda size=mesh_size: size)
            model._device_mesh = types.SimpleNamespace(mesh=ranks)
        return model

    def test_sharded_dtensor_parameters_are_detected(self):
        model = self._model(placements=["Shard(dim=0)"], mesh_size=2)
        self.assertTrue(_dtensor_sharded(model))
        _report_tp_sharding(model, 2)

    def test_replicated_dtensor_parameters_are_not_sharding(self):
        model = self._model(placements=[_FakeReplicate()], mesh_size=1)
        self.assertFalse(_dtensor_sharded(model))

    def test_hooked_modules_report_a_sharded_model(self):
        model = self._model(tp_size=2, plan={"layers.*.q_proj": "colwise"}, hooked=True)
        with mock.patch("builtins.print") as fake_print:
            _report_tp_sharding(model, 2)
        self.assertIn("Target sharded over 2 ranks", str(fake_print.call_args_list[0]))

    def test_sharding_over_the_wrong_group_size_is_rejected(self):
        model = self._model(tp_size=2, hooked=True, mesh_size=4)
        with self.assertRaises(RuntimeError) as caught:
            _report_tp_sharding(model, 2)
        self.assertIn("sharded across 4 ranks", str(caught.exception))

    def test_lm_head_only_plan_is_called_out(self):
        model = self._model(tp_size=2, plan={"lm_head": "colwise_gather_output"}, hooked=True)
        with mock.patch("builtins.print") as fake_print:
            _report_tp_sharding(model, 2)
        printed = " ".join(str(call) for call in fake_print.call_args_list)
        self.assertIn("only covers", printed)

    def test_replicated_only_model_warns_instead_of_failing(self):
        model = self._model()
        with mock.patch("builtins.print") as fake_print:
            _report_tp_sharding(model, 2)
        printed = " ".join(str(call) for call in fake_print.call_args_list)
        self.assertIn("no tensor-parallel sharding", printed)


class AcceptanceTensorParallelLoaderTest(unittest.TestCase):
    def test_loader_attempts_cover_both_transformers_apis(self):
        mesh = object()
        attempts = _tp_loader_attempts(2, mesh)
        self.assertTrue(attempts)
        self.assertEqual(attempts[-1][1]["tp_plan"], "auto")
        for _, kwargs in attempts:
            self.assertIs(kwargs["device_mesh"], mesh)
        if len(attempts) > 1:
            distributed_config = attempts[0][1]["distributed_config"]
            self.assertEqual(distributed_config.tp_size, 2)

    def test_load_falls_back_to_the_older_tp_plan_api(self):
        model = types.SimpleNamespace()
        calls = []

        def _from_pretrained(path, **kwargs):
            calls.append(kwargs)
            if "distributed_config" in kwargs:
                raise TypeError(
                    "Qwen3_5MoeForCausalLM.__init__() got an unexpected keyword "
                    "argument 'distributed_config'"
                )
            return model

        with (
            mock.patch.object(
                check_acceptance, "_tp_device_mesh", return_value="mesh"
            ),
            mock.patch.object(
                check_acceptance.AutoModelForCausalLM,
                "from_pretrained",
                _from_pretrained,
            ),
        ):
            loaded = check_acceptance._load_target_model_tp("target", 1, 2, None)

        self.assertIs(loaded, model)
        self.assertTrue(any("distributed_config" in kwargs for kwargs in calls))
        self.assertEqual(calls[-1]["tp_plan"], "auto")
        self.assertEqual(calls[-1]["device_mesh"], "mesh")

    def test_load_uses_the_new_api_when_accepted(self):
        model = types.SimpleNamespace()
        calls = []

        def _from_pretrained(path, **kwargs):
            calls.append(kwargs)
            return model

        with (
            mock.patch.object(
                check_acceptance, "_tp_device_mesh", return_value="mesh"
            ),
            mock.patch.object(
                check_acceptance.AutoModelForCausalLM,
                "from_pretrained",
                _from_pretrained,
            ),
        ):
            loaded = check_acceptance._load_target_model_tp("target", 1, 2, None)

        self.assertIs(loaded, model)
        self.assertEqual(len(calls), 1)
        self.assertIn("device_mesh", calls[0])


if __name__ == "__main__":
    unittest.main()
