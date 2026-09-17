# coding=utf-8
"""CPU regressions for the shared gate/up SwiGLU MLP."""

import unittest
import warnings

import torch
from torch import nn
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP

from specforge.modeling.draft.dflash_kernels import (
    DEFAULT_DFLASH_KERNELS,
    FoldedSoftmaxReadout,
    SharedGLUMLP,
    resolve_ffn_sharing,
)

HIDDEN = 32
INTERMEDIATE = 128

# (pairing, gate_groups, up_groups) against INTERMEDIATE=128.  Covers pure
# gate sharing, pure up sharing, a misaligned hierarchical mix, the exact
# outer lattice and the outer lattice with one repetition.
SHARING_CASES = [
    ("nested", 64, 128),
    ("nested", 128, 64),
    ("nested", 64, 32),
    ("outer", 8, 16),
    ("outer", 8, 8),
]


def _config(
    *,
    hidden_size: int = HIDDEN,
    intermediate_size: int = INTERMEDIATE,
    sharing=None,
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
    if sharing is not None:
        config.dflash_config = {"ffn_sharing": sharing}
    return config


class TestResolveFfnSharing(unittest.TestCase):
    def test_disabled_by_default(self):
        self.assertIsNone(resolve_ffn_sharing(_config()))
        self.assertIsNone(resolve_ffn_sharing(_config(sharing=None)))
        self.assertIsNone(resolve_ffn_sharing(_config(sharing=False)))

    def test_gate_only_defaults_up_to_identity(self):
        self.assertEqual(
            resolve_ffn_sharing(_config(sharing={"gate_groups": 64})),
            {"pairing": "nested", "gate_groups": 64, "up_groups": 128},
        )

    def test_pairing_and_groups_round_trip(self):
        self.assertEqual(
            resolve_ffn_sharing(
                _config(sharing={"pairing": "outer", "gate_groups": 8, "up_groups": 16})
            ),
            {"pairing": "outer", "gate_groups": 8, "up_groups": 16},
        )

    def test_reference_shapes_resolve(self):
        # The 35B-A3B draft: 9728 = 4864 * 2 (pure gate k=2) and the full
        # 76 x 128 = 9728 lattice both keep down_proj untouched.
        big = resolve_ffn_sharing(
            _config(intermediate_size=9728, sharing={"gate_groups": 4864})
        )
        self.assertEqual(
            big, {"pairing": "nested", "gate_groups": 4864, "up_groups": 9728}
        )
        lattice = resolve_ffn_sharing(
            _config(
                intermediate_size=9728,
                sharing={"pairing": "outer", "gate_groups": 76, "up_groups": 128},
            )
        )
        self.assertEqual(
            lattice, {"pairing": "outer", "gate_groups": 76, "up_groups": 128}
        )

    def test_rejects_invalid_knobs(self):
        with self.assertRaises(ValueError):
            resolve_ffn_sharing(_config(sharing="gate"))
        with self.assertRaises(ValueError):
            resolve_ffn_sharing(_config(sharing={"pairing_k": "nested"}))
        with self.assertRaises(ValueError):
            resolve_ffn_sharing(_config(sharing={"pairing": "weave"}))
        with self.assertRaises(ValueError):
            resolve_ffn_sharing(_config(sharing={"gate_groups": 5}))
        with self.assertRaises(ValueError):
            resolve_ffn_sharing(_config(sharing={"gate_groups": 129}))
        with self.assertRaises(ValueError):
            resolve_ffn_sharing(
                _config(intermediate_size=9728, sharing={"gate_groups": 3243})
            )

    def test_outer_divisibility_error_mentions_factorizations(self):
        with self.assertRaises(ValueError) as ctx:
            resolve_ffn_sharing(
                _config(
                    intermediate_size=9728,
                    sharing={"pairing": "outer", "gate_groups": 64, "up_groups": 64},
                )
            )
        message = str(ctx.exception)
        self.assertIn("4096", message)
        self.assertIn("76x128", message)


class TestSharedGLUMLP(unittest.TestCase):
    def _reference(self, mlp: SharedGLUMLP, x: torch.Tensor) -> torch.Tensor:
        gate = mlp.gate_proj(x)
        up = mlp.up_proj(x)
        hidden = torch.stack(
            [
                mlp.act_fn(gate[..., mlp.gate_idx[j]]) * up[..., mlp.up_idx[j]]
                for j in range(mlp.intermediate_size)
            ],
            dim=-1,
        )
        return mlp.down_proj(hidden)

    def test_factory_returns_shared_mlp(self):
        mlp = DEFAULT_DFLASH_KERNELS.make_mlp(
            _config(sharing={"gate_groups": 64})
        )
        self.assertIsInstance(mlp, SharedGLUMLP)
        self.assertEqual(mlp.gate_proj.weight.shape, (64, HIDDEN))
        self.assertEqual(mlp.up_proj.weight.shape, (INTERMEDIATE, HIDDEN))
        self.assertEqual(mlp.down_proj.weight.shape, (HIDDEN, INTERMEDIATE))
        self.assertIsInstance(DEFAULT_DFLASH_KERNELS.make_mlp(_config()), Qwen3MLP)

    def test_matches_reference_indices(self):
        torch.manual_seed(0)
        for pairing, gate_groups, up_groups in SHARING_CASES:
            with self.subTest(
                pairing=pairing, gate_groups=gate_groups, up_groups=up_groups
            ):
                mlp = DEFAULT_DFLASH_KERNELS.make_mlp(
                    _config(
                        sharing={
                            "pairing": pairing,
                            "gate_groups": gate_groups,
                            "up_groups": up_groups,
                        }
                    )
                )
                x = torch.randn(2, 3, HIDDEN)
                self.assertTrue(
                    torch.allclose(mlp(x), self._reference(mlp, x), atol=1e-6)
                )

    def test_outer_indices_form_the_full_lattice(self):
        mlp = SharedGLUMLP(
            _config(),
            pairing="outer",
            gate_groups=8,
            up_groups=16,
        )
        pairs = set(zip(mlp.gate_idx.tolist(), mlp.up_idx.tolist()))
        self.assertEqual(len(pairs), INTERMEDIATE)
        self.assertEqual(pairs, {(r, c) for r in range(8) for c in range(16)})

    def test_identity_sharing_matches_qwen3_mlp_bitwise(self):
        torch.manual_seed(1)
        baseline = Qwen3MLP(_config())
        shared = DEFAULT_DFLASH_KERNELS.make_mlp(
            _config(
                sharing={
                    "gate_groups": INTERMEDIATE,
                    "up_groups": INTERMEDIATE,
                }
            )
        )
        # Names line up exactly (the index maps are non-persistent buffers),
        # so a strict load works and the function is bit-identical.
        shared.load_state_dict(baseline.state_dict())
        x = torch.randn(4, HIDDEN)
        self.assertTrue(torch.equal(shared(x), baseline(x)))

    def test_parameter_budget(self):
        mlp = DEFAULT_DFLASH_KERNELS.make_mlp(
            _config(sharing={"gate_groups": 64, "up_groups": 128})
        )
        total = sum(parameter.numel() for parameter in mlp.parameters())
        self.assertEqual(total, HIDDEN * (64 + 128) + HIDDEN * INTERMEDIATE)
        self.assertLess(total, 3 * HIDDEN * INTERMEDIATE)

    def test_legacy_dense_projections_average_on_load(self):
        torch.manual_seed(2)
        baseline = Qwen3MLP(_config())
        shared = DEFAULT_DFLASH_KERNELS.make_mlp(
            _config(sharing={"gate_groups": 64})
        )
        result = shared.load_state_dict(baseline.state_dict(), strict=False)

        expected_gate = baseline.gate_proj.weight.reshape(64, 2, HIDDEN).mean(dim=1)
        self.assertTrue(
            torch.allclose(shared.gate_proj.weight, expected_gate, atol=1e-6)
        )
        # The untouched sides and the down projection copy verbatim.
        self.assertTrue(torch.equal(shared.up_proj.weight, baseline.up_proj.weight))
        self.assertTrue(
            torch.equal(shared.down_proj.weight, baseline.down_proj.weight)
        )
        self.assertEqual(list(result.missing_keys), [])
        self.assertEqual(list(result.unexpected_keys), [])

    def test_legacy_dense_outer_averages_strided_groups(self):
        torch.manual_seed(3)
        baseline = Qwen3MLP(_config())
        shared = DEFAULT_DFLASH_KERNELS.make_mlp(
            _config(sharing={"pairing": "outer", "gate_groups": 8, "up_groups": 16})
        )
        result = shared.load_state_dict(baseline.state_dict(), strict=False)

        # Gate group r collects the strided rows r, r + 8, r + 16, ...
        self.assertTrue(
            torch.allclose(
                shared.gate_proj.weight,
                baseline.gate_proj.weight.reshape(16, 8, HIDDEN).mean(dim=0),
                atol=1e-6,
            )
        )
        # Up group c collects rows j with (j // 8) % 16 == c.
        up_expected = torch.stack(
            [
                baseline.up_proj.weight[(torch.arange(INTERMEDIATE) // 8) % 16 == c]
                .float()
                .mean(dim=0)
                for c in range(16)
            ]
        )
        self.assertTrue(
            torch.allclose(shared.up_proj.weight, up_expected, atol=1e-6)
        )
        self.assertEqual(list(result.missing_keys), [])
        self.assertEqual(list(result.unexpected_keys), [])

    def test_legacy_intermediate_mismatch_reports_actionable_error(self):
        shared = DEFAULT_DFLASH_KERNELS.make_mlp(
            _config(
                intermediate_size=64,
                sharing={"pairing": "outer", "gate_groups": 8, "up_groups": 8},
            )
        )
        baseline = Qwen3MLP(_config(intermediate_size=128))
        with self.assertRaisesRegex(RuntimeError, "train from scratch"):
            shared.load_state_dict(baseline.state_dict(), strict=False)

    def test_composes_with_folded_readout(self):
        config = _config()
        config.dflash_config = {
            "ffn_readout": {"mode": "folded_softmax", "branches": 4, "granularity": 8},
            "ffn_sharing": {"gate_groups": 64},
        }
        mlp = DEFAULT_DFLASH_KERNELS.make_mlp(config)
        self.assertIsInstance(mlp, SharedGLUMLP)
        self.assertIsInstance(mlp.down_proj, FoldedSoftmaxReadout)
        self.assertEqual(mlp.down_proj.proj.weight.shape, (HIDDEN, INTERMEDIATE // 4))

        # Both legacy conversions compose: gate rows average, dense down folds.
        torch.manual_seed(4)
        baseline = Qwen3MLP(_config())
        result = mlp.load_state_dict(baseline.state_dict(), strict=False)
        self.assertEqual(list(result.missing_keys), [])
        self.assertEqual(list(result.unexpected_keys), [])
        expected_down = baseline.down_proj.weight.reshape(
            HIDDEN, 4, INTERMEDIATE // 4
        ).sum(dim=1)
        self.assertTrue(
            torch.allclose(mlp.down_proj.proj.weight, expected_down)
        )
        expected_gate = baseline.gate_proj.weight.reshape(64, 2, HIDDEN).mean(dim=1)
        self.assertTrue(
            torch.allclose(mlp.gate_proj.weight, expected_gate, atol=1e-6)
        )

    def test_forward_rides_the_module_dtype(self):
        mlp = DEFAULT_DFLASH_KERNELS.make_mlp(
            _config(sharing={"pairing": "outer", "gate_groups": 8, "up_groups": 16})
        ).to(torch.bfloat16)
        x = torch.randn(4, HIDDEN, dtype=torch.bfloat16)
        self.assertEqual(mlp(x).dtype, torch.bfloat16)

    def test_qat_quantizes_projections_and_leaves_indices_alone(self):
        from specforge.layers.wxay import (
            QuantizedLinear,
            replace_linear_with_quantized,
        )

        torch.manual_seed(5)
        mlp = DEFAULT_DFLASH_KERNELS.make_mlp(_config(sharing={"gate_groups": 64}))
        gate_idx_before = mlp.gate_idx.clone()
        up_idx_before = mlp.up_idx.clone()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            replace_linear_with_quantized(mlp, w_bit=4, a_bit=8)

        # Every plain linear is quantized exactly like the dense MLP it
        # replaced, while the shared index maps stay untouched.
        self.assertIsInstance(mlp.gate_proj, QuantizedLinear)
        self.assertIsInstance(mlp.up_proj, QuantizedLinear)
        self.assertIsInstance(mlp.down_proj, QuantizedLinear)
        self.assertTrue(torch.equal(mlp.gate_idx, gate_idx_before))
        self.assertTrue(torch.equal(mlp.up_idx, up_idx_before))
        # QAT keeps the parameter names, so export and serving are unaffected.
        self.assertIn("gate_proj.weight", mlp.state_dict())
        self.assertNotIn("gate_idx", mlp.state_dict())
        self.assertNotIn("up_idx", mlp.state_dict())


if __name__ == "__main__":
    unittest.main()
