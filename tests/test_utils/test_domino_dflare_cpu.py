"""CPU-side tests for the migrated dflare Domino model path.

These tests intentionally avoid NPU/SGLang imports and use only torch +
transformers, so they run in the basic local environment.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from specforge.modeling.auto import AutoDraftModelConfig
from specforge.modeling.draft.dflash import (
    compute_acceptance_stats,
    is_complete_block,
    resolve_dflash_attention_layout,
    validate_dflash_attention_backend,
)


class DominoDFlareConfigTest(unittest.TestCase):
    def test_verified_base_config_loads(self):
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "configs",
            "qwen3.8-27b-domino-dflare-verifiedBase.json",
        )
        config = AutoDraftModelConfig.from_file(path)
        self.assertEqual(config.hidden_size, 2560)
        self.assertEqual(config.dflash_config["target_hidden_size"], 5120)
        self.assertEqual(
            config.dflash_config["sliding_window"],
            [3072, 2048, 512, 512, 1024, 1024, 3072],
        )
        layout, windows = resolve_dflash_attention_layout(config)
        self.assertEqual(len(layout), 7)
        self.assertEqual(len(windows), 7)
        validate_dflash_attention_backend(config, "sdpa")


class AcceptanceStatsTest(unittest.TestCase):
    def test_complete_block(self):
        self.assertTrue(is_complete_block(0, 4, 4))
        self.assertFalse(is_complete_block(2, 4, 5))

    def test_stats_complete_only(self):
        stats = compute_acceptance_stats([4, 3, 2], [True, True, False], 4)
        self.assertEqual(stats["mean_acceptance_length"], 3.5)
        self.assertEqual(stats["num_complete_blocks"], 2)
        self.assertEqual(stats["num_incomplete_blocks"], 1)


if __name__ == "__main__":
    unittest.main()
