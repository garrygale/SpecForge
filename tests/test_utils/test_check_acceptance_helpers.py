"""Local-data tests for acceptance evaluator helpers."""

from __future__ import annotations

import os
import unittest

from inference.check_acceptance import (
    aggregate_stats,
    load_humaneval,
    load_math500,
    load_mbpp,
)

_INFERENCE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "inference",
)


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


if __name__ == "__main__":
    unittest.main()
