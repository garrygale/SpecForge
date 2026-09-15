# coding=utf-8
"""CPU regressions for the folded softmax FFN readout."""

import json
import unittest
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP

from specforge.modeling.draft.dflash_kernels import (
    DEFAULT_DFLASH_KERNELS,
    FoldedSoftmaxMLP,
    FoldedSoftmaxReadout,
    resolve_folded_readout,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_CONFIG = (
    REPO_ROOT / "configs" / "qwen3.6-35b-a3b-domino-dflare-verifiedBase.json"
)

HIDDEN = 32
INTERMEDIATE = 128
BRANCHES = 4
GRANULARITY = 8


def _config(
    *,
    hidden_size: int = HIDDEN,
    intermediate_size: int = INTERMEDIATE,
    readout=None,
    with_key: bool = True,
):
    config = Qwen3Config(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=hidden_size // 2,
        vocab_size=64,
        hidden_act="silu",
    )
    if with_key:
        config.dflash_config = {"ffn_readout": readout}
    return config


class TestResolveFoldedReadout(unittest.TestCase):
    def test_dense_by_default(self):
        self.assertIsNone(resolve_folded_readout(_config(with_key=False)))
        self.assertIsNone(resolve_folded_readout(_config(readout=None)))
        self.assertIsNone(resolve_folded_readout(_config(readout=False)))
        self.assertIsNone(resolve_folded_readout(_config(readout="dense")))
        self.assertIsNone(
            resolve_folded_readout(_config(readout={"mode": "dense"}))
        )

    def test_mode_string_uses_fold_ratio(self):
        # 128 / 32 == 4 branches, folded width 32, default granularity 16.
        self.assertEqual(
            resolve_folded_readout(_config(readout="folded_softmax")),
            {"branches": 4, "granularity": 16},
        )

    def test_dict_overrides_branches_and_granularity(self):
        self.assertEqual(
            resolve_folded_readout(
                _config(readout={"branches": 8, "granularity": 8})
            ),
            {"branches": 8, "granularity": 8},
        )

    def test_fold_ratio_rounds_to_nearest_divisor(self):
        # The 35B-A3B draft: 9728 / 2560 = 3.8 -> 4 branches, width 2432.
        self.assertEqual(
            resolve_folded_readout(
                _config(
                    hidden_size=2560,
                    intermediate_size=9728,
                    readout="folded_softmax",
                )
            ),
            {"branches": 4, "granularity": 16},
        )
        # A 3N intermediate keeps the 3N -> N -> N shape of the design note.
        self.assertEqual(
            resolve_folded_readout(
                _config(
                    hidden_size=4096,
                    intermediate_size=12288,
                    readout="folded_softmax",
                )
            ),
            {"branches": 3, "granularity": 16},
        )

    def test_rejects_invalid_knobs(self):
        with self.assertRaises(ValueError):
            resolve_folded_readout(_config(readout="not_a_mode"))
        with self.assertRaises(ValueError):
            resolve_folded_readout(_config(readout={"branches": 5}))
        with self.assertRaises(ValueError):
            resolve_folded_readout(
                _config(readout={"branches": 4, "granularity": 48})
            )
        with self.assertRaises(ValueError):
            resolve_folded_readout(_config(readout={"granularity_k": 8}))

    def test_reference_config_resolves(self):
        if not REFERENCE_CONFIG.exists():  # pragma: no cover
            self.skipTest(f"missing {REFERENCE_CONFIG}")
        raw = json.loads(REFERENCE_CONFIG.read_text(encoding="utf-8"))
        config = Qwen3Config(
            hidden_size=raw["hidden_size"],
            intermediate_size=raw["intermediate_size"],
            num_hidden_layers=raw["num_hidden_layers"],
            num_attention_heads=raw["num_attention_heads"],
            num_key_value_heads=raw["num_key_value_heads"],
            head_dim=raw["head_dim"],
            vocab_size=raw["vocab_size"],
            hidden_act=raw["hidden_act"],
            dflash_config={
                **raw["dflash_config"],
                "ffn_readout": {"mode": "folded_softmax"},
            },
        )
        self.assertEqual(
            resolve_folded_readout(config), {"branches": 4, "granularity": 16}
        )
        # Analytic ledger per draft layer: gate/up keep 2NM, the folded path
        # costs N*(M/c) + M + c*K instead of N*M.
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size
        folded_size = intermediate_size // 4
        dense = 3 * hidden_size * intermediate_size
        folded = (
            2 * hidden_size * intermediate_size
            + hidden_size * folded_size
            + intermediate_size
            + 4 * 16
        )
        self.assertGreater(dense / folded, 1.3)
        self.assertLess(dense / folded, 1.4)


class TestFoldedSoftmaxReadout(unittest.TestCase):
    def test_matches_explicit_fold_reference(self):
        torch.manual_seed(0)
        readout = FoldedSoftmaxReadout(
            HIDDEN, INTERMEDIATE, branches=BRANCHES, granularity=GRANULARITY
        )
        with torch.no_grad():
            readout.fold_logits.normal_()
        hidden_states = torch.randn(2, 3, INTERMEDIATE)

        out = readout(hidden_states)

        weights = torch.softmax(readout.fold_logits.float(), dim=0)
        chunks = hidden_states.reshape(2, 3, BRANCHES, HIDDEN)
        mixed = torch.zeros(2, 3, HIDDEN)
        for branch in range(BRANCHES):
            for offset in range(HIDDEN):
                mixed[..., offset] += (
                    weights[branch, offset % GRANULARITY]
                    * chunks[..., branch, offset]
                )
        expected = F.linear(mixed, readout.proj.weight)
        self.assertTrue(torch.allclose(out, expected, atol=1e-6))

    def test_mixture_is_convex(self):
        torch.manual_seed(1)
        readout = FoldedSoftmaxReadout(
            HIDDEN, INTERMEDIATE, branches=BRANCHES, granularity=GRANULARITY
        )
        with torch.no_grad():
            readout.fold_logits.normal_(std=3.0)
        weights = readout.fold_weights()
        self.assertTrue(bool((weights >= 0).all()))
        self.assertTrue(
            torch.allclose(
                weights.sum(dim=0), torch.ones(GRANULARITY), atol=1e-6
            )
        )
        with torch.no_grad():
            readout.fold_logits.zero_()
        self.assertTrue(
            torch.allclose(
                readout.fold_weights(),
                torch.full((BRANCHES, GRANULARITY), 1 / BRANCHES),
                atol=1e-6,
            )
        )

    def test_single_branch_is_a_dense_readout(self):
        torch.manual_seed(2)
        readout = FoldedSoftmaxReadout(16, 48, branches=1, granularity=4)
        hidden_states = torch.randn(5, 48)
        self.assertTrue(
            torch.allclose(
                readout(hidden_states),
                F.linear(hidden_states, readout.proj.weight),
                atol=1e-6,
            )
        )

    def test_factory_returns_folded_mlp(self):
        config = _config(
            readout={"branches": BRANCHES, "granularity": GRANULARITY}
        )
        mlp = DEFAULT_DFLASH_KERNELS.make_mlp(config)
        self.assertIsInstance(mlp, FoldedSoftmaxMLP)
        self.assertEqual(mlp.gate_proj.weight.shape, (INTERMEDIATE, HIDDEN))
        self.assertEqual(mlp.up_proj.weight.shape, (INTERMEDIATE, HIDDEN))
        self.assertIsInstance(mlp.down_proj.proj, nn.Linear)
        self.assertEqual(mlp.down_proj.proj.weight.shape, (HIDDEN, HIDDEN))
        self.assertEqual(
            mlp.down_proj.fold_logits.shape, (BRANCHES, GRANULARITY)
        )
        self.assertIsInstance(
            DEFAULT_DFLASH_KERNELS.make_mlp(_config()), Qwen3MLP
        )

    def test_legacy_dense_down_proj_folds_on_load(self):
        torch.manual_seed(3)
        baseline = Qwen3MLP(_config())
        folded = DEFAULT_DFLASH_KERNELS.make_mlp(
            _config(readout={"branches": BRANCHES, "granularity": GRANULARITY})
        )
        result = folded.load_state_dict(baseline.state_dict(), strict=False)

        expected = baseline.down_proj.weight.reshape(
            HIDDEN, BRANCHES, HIDDEN
        ).sum(dim=1)
        self.assertTrue(
            torch.allclose(folded.down_proj.proj.weight, expected)
        )
        self.assertFalse(bool(torch.any(folded.down_proj.fold_logits)))
        # The absent logits are intentional, not a warm-start gap.
        self.assertEqual(list(result.missing_keys), [])
        self.assertEqual(list(result.unexpected_keys), [])
        self.assertTrue(
            torch.allclose(folded.gate_proj.weight, baseline.gate_proj.weight)
        )

    def test_single_branch_reproduces_the_dense_mlp(self):
        torch.manual_seed(4)
        baseline = Qwen3MLP(_config())
        folded = DEFAULT_DFLASH_KERNELS.make_mlp(
            _config(readout={"branches": 1, "granularity": GRANULARITY})
        )
        folded.load_state_dict(baseline.state_dict(), strict=False)
        inputs = torch.randn(4, HIDDEN)
        self.assertTrue(
            torch.allclose(folded(inputs), baseline(inputs), atol=1e-6)
        )

    def test_parameter_budget_shrinks_the_readout(self):
        config = _config(
            readout={"branches": BRANCHES, "granularity": GRANULARITY}
        )
        mlp = DEFAULT_DFLASH_KERNELS.make_mlp(config)
        total = sum(parameter.numel() for parameter in mlp.parameters())
        expected = (
            2 * HIDDEN * INTERMEDIATE
            + HIDDEN * (INTERMEDIATE // BRANCHES)
            + BRANCHES * GRANULARITY
        )
        self.assertEqual(total, expected)
        self.assertLess(total, 3 * HIDDEN * INTERMEDIATE)

    def test_qat_quantizes_only_the_dense_projection(self):
        from specforge.layers.wxay import (
            QuantizedLinear,
            replace_linear_with_quantized,
        )

        torch.manual_seed(5)
        mlp = DEFAULT_DFLASH_KERNELS.make_mlp(
            _config(readout={"branches": BRANCHES, "granularity": GRANULARITY})
        )
        with torch.no_grad():
            mlp.down_proj.fold_logits.normal_()
        logits_before = mlp.down_proj.fold_logits.detach().clone()
        mixture_before = mlp.down_proj.fold_weights().clone()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            replace_linear_with_quantized(
                mlp, w_bit=4, a_bit=8, exclude_names=["embed_proj"]
            )

        # Every plain linear — including the folded readout's projection — is
        # quantized exactly like the dense down_proj it replaced ...
        self.assertIsInstance(mlp.gate_proj, QuantizedLinear)
        self.assertIsInstance(mlp.up_proj, QuantizedLinear)
        self.assertIsInstance(mlp.down_proj.proj, QuantizedLinear)
        # ... while the softmax mixture stays an exact, unquantized vector op.
        self.assertIsInstance(mlp.down_proj, FoldedSoftmaxReadout)
        self.assertIsInstance(mlp.down_proj.fold_logits, nn.Parameter)
        self.assertTrue(torch.equal(mlp.down_proj.fold_logits, logits_before))
        self.assertTrue(
            torch.allclose(mlp.down_proj.fold_weights(), mixture_before)
        )
        # QAT keeps the parameter names, so export and serving are unaffected.
        self.assertIn("down_proj.proj.weight", mlp.state_dict())
        self.assertIn("down_proj.fold_logits", mlp.state_dict())


if __name__ == "__main__":
    unittest.main()
