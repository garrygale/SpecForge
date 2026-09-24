# coding=utf-8
"""CPU regressions for the shared gate/up SwiGLU MLP and its routed variant."""

import unittest
import warnings

import torch
from torch import nn
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP

from specforge.modeling.draft.dflash_kernels import (
    DEFAULT_DFLASH_KERNELS,
    RoutedOuterMLP,
    SharedGLUMLP,
    resolve_ffn_sharing,
)

HIDDEN = 32
INTERMEDIATE = 128

# (pairing, gate_groups, up_groups) against INTERMEDIATE=128 for the single
# lattice: pure gate sharing, pure up sharing, a misaligned hierarchical mix,
# the exactly-once outer lattice and the outer lattice with one repetition.
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
    readout=None,
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
    dflash_config = {}
    if sharing is not None:
        dflash_config["ffn_sharing"] = sharing
    if readout is not None:
        dflash_config["ffn_readout"] = readout
    if dflash_config:
        config.dflash_config = dflash_config
    return config


class TestResolveFfnSharing(unittest.TestCase):
    def test_disabled_by_default(self):
        self.assertIsNone(resolve_ffn_sharing(_config()))
        self.assertIsNone(resolve_ffn_sharing(_config(sharing=None)))
        self.assertIsNone(resolve_ffn_sharing(_config(sharing=False)))

    def test_lattice_defaults(self):
        self.assertEqual(
            resolve_ffn_sharing(_config(sharing={"gate_groups": 64})),
            {
                "mode": "lattice",
                "pairing": "nested",
                "gate_groups": 64,
                "up_groups": 128,
            },
        )

    def test_routed_outer_resolves(self):
        self.assertEqual(
            resolve_ffn_sharing(
                _config(
                    intermediate_size=48,
                    sharing={
                        "mode": "routed_outer",
                        "experts": 4,
                        "gate_groups": 3,
                        "up_groups": 4,
                    },
                )
            ),
            {
                "mode": "routed_outer",
                "experts": 4,
                "gate_groups": 3,
                "up_groups": 4,
                "router": "expert",
            },
        )
        self.assertEqual(
            resolve_ffn_sharing(
                _config(
                    intermediate_size=24,
                    sharing={
                        "mode": "routed_outer",
                        "experts": 2,
                        "gate_groups": 3,
                        "up_groups": 4,
                        "router": "gate_slot",
                    },
                )
            )["router"],
            "gate_slot",
        )

    def test_routed_outer_rejections(self):
        with self.assertRaises(ValueError):  # wrong intermediate_size
            resolve_ffn_sharing(
                _config(
                    intermediate_size=47,
                    sharing={"mode": "routed_outer", "experts": 4, "gate_groups": 3, "up_groups": 4},
                )
            )
        with self.assertRaises(ValueError):  # gate_slot with one expert
            resolve_ffn_sharing(
                _config(
                    intermediate_size=12,
                    sharing={"mode": "routed_outer", "experts": 1, "gate_groups": 3, "up_groups": 4, "router": "gate_slot"},
                )
            )
        with self.assertRaises(ValueError):  # unknown router
            resolve_ffn_sharing(
                _config(
                    intermediate_size=48,
                    sharing={"mode": "routed_outer", "experts": 4, "gate_groups": 3, "up_groups": 4, "router": "token"},
                )
            )
        with self.assertRaises(ValueError):  # unknown mode
            resolve_ffn_sharing(
                _config(intermediate_size=48, sharing={"mode": "weave"})
            )

    def test_lattice_rejections(self):
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

    def test_retired_fold_guard(self):
        with self.assertRaises(ValueError):
            DEFAULT_DFLASH_KERNELS.make_mlp(
                _config(sharing={"gate_groups": 64}, readout={"mode": "folded_softmax"})
            )
        # dense/off spellings stay accepted no-ops.
        self.assertIsInstance(
            DEFAULT_DFLASH_KERNELS.make_mlp(
                _config(sharing={"gate_groups": 64}, readout="dense")
            ),
            SharedGLUMLP,
        )


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
        mlp = DEFAULT_DFLASH_KERNELS.make_mlp(_config(sharing={"gate_groups": 64}))
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

    def test_identity_sharing_matches_qwen3_mlp_bitwise(self):
        torch.manual_seed(1)
        baseline = Qwen3MLP(_config())
        shared = DEFAULT_DFLASH_KERNELS.make_mlp(
            _config(sharing={"gate_groups": INTERMEDIATE, "up_groups": INTERMEDIATE})
        )
        shared.load_state_dict(baseline.state_dict())
        x = torch.randn(4, HIDDEN)
        self.assertTrue(torch.equal(shared(x), baseline(x)))

    def test_legacy_dense_projections_average_on_load(self):
        torch.manual_seed(2)
        baseline = Qwen3MLP(_config())
        shared = DEFAULT_DFLASH_KERNELS.make_mlp(_config(sharing={"gate_groups": 64}))
        result = shared.load_state_dict(baseline.state_dict(), strict=False)
        expected_gate = baseline.gate_proj.weight.reshape(64, 2, HIDDEN).mean(dim=1)
        self.assertTrue(
            torch.allclose(shared.gate_proj.weight, expected_gate, atol=1e-6)
        )
        self.assertTrue(torch.equal(shared.up_proj.weight, baseline.up_proj.weight))
        self.assertEqual(list(result.missing_keys), [])
        self.assertEqual(list(result.unexpected_keys), [])

    def test_qat_quantizes_projections(self):
        from specforge.layers.wxay import (
            QuantizedLinear,
            replace_linear_with_quantized,
        )

        torch.manual_seed(5)
        mlp = DEFAULT_DFLASH_KERNELS.make_mlp(_config(sharing={"gate_groups": 64}))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            replace_linear_with_quantized(mlp, w_bit=4, a_bit=8)
        self.assertIsInstance(mlp.gate_proj, QuantizedLinear)
        self.assertIsInstance(mlp.up_proj, QuantizedLinear)
        self.assertIsInstance(mlp.down_proj, QuantizedLinear)


class TestRoutedOuterMLP(unittest.TestCase):
    def _make(self, experts, gate_groups, up_groups, router="expert"):
        return DEFAULT_DFLASH_KERNELS.make_mlp(
            _config(
                intermediate_size=experts * gate_groups * up_groups,
                sharing={
                    "mode": "routed_outer",
                    "experts": experts,
                    "gate_groups": gate_groups,
                    "up_groups": up_groups,
                    "router": router,
                },
            )
        )

    def _reference(self, mlp, x):
        fused = mlp.gate_up_proj(x)
        E, G, U = mlp.experts, mlp.gate_groups, mlp.up_groups
        lead = x.shape[:-1]
        feats = fused[..., : E * (G + U)].reshape(*lead, E, G + U)
        route = fused[..., E * (G + U):]
        ref = torch.zeros(*lead, G * U)
        if mlp.router == "expert":
            alpha = torch.softmax(route, dim=-1)
            for e in range(E):
                a = mlp.act_fn(feats[..., e, :G]) * alpha[..., e : e + 1]
                u = feats[..., e, G:]
                ref = ref + (a.unsqueeze(-1) * u.unsqueeze(-2)).reshape(*lead, G * U)
        else:
            delta = torch.softmax(route.reshape(*lead, G, E), dim=-1)
            for e in range(E):
                a = mlp.act_fn(feats[..., e, :G]) * delta[..., :, e]
                u = feats[..., e, G:]
                ref = ref + (a.unsqueeze(-1) * u.unsqueeze(-2)).reshape(*lead, G * U)
        return mlp.down_proj(ref)

    def test_factory_and_shapes(self):
        mlp = self._make(4, 3, 4)
        self.assertIsInstance(mlp, RoutedOuterMLP)
        self.assertEqual(
            mlp.gate_up_proj.weight.shape, (4 * (3 + 4) + 4, HIDDEN)
        )
        self.assertEqual(mlp.down_proj.weight.shape, (HIDDEN, 3 * 4))

    def test_matches_reference_both_routers(self):
        torch.manual_seed(3)
        for experts, gate_groups, up_groups, router in (
            (4, 3, 4, "expert"),
            (2, 3, 4, "gate_slot"),
        ):
            with self.subTest(experts=experts, router=router):
                mlp = self._make(experts, gate_groups, up_groups, router)
                with torch.no_grad():
                    mlp.gate_up_proj.weight[experts * (gate_groups + up_groups):].normal_()
                x = torch.randn(2, 5, HIDDEN)
                self.assertTrue(
                    torch.allclose(mlp(x), self._reference(mlp, x), atol=1e-6)
                )

    def test_router_zero_init_starts_uniform(self):
        torch.manual_seed(4)
        mlp = self._make(4, 3, 4)
        self.assertFalse(
            bool(torch.any(mlp.gate_up_proj.weight[4 * 7:].abs() > 0))
        )
        # With zero routing logits the blend is the uniform expert average.
        x = torch.randn(5, HIDDEN)
        fused = mlp.gate_up_proj(x)
        feats = fused[..., :-4].reshape(5, 4, 7)
        gate = mlp.act_fn(feats[..., :3]) / 4
        up = feats[..., 3:]
        expected = mlp.down_proj(
            (gate.unsqueeze(-1) * up.unsqueeze(-2))
            .sum(dim=-3)
            .reshape(5, 12)
        )
        self.assertTrue(torch.allclose(mlp(x), expected, atol=1e-6))

    def test_single_expert_is_the_plain_outer_lattice(self):
        torch.manual_seed(5)
        routed = self._make(1, 3, 4)
        plain = DEFAULT_DFLASH_KERNELS.make_mlp(
            _config(
                intermediate_size=12,
                sharing={"pairing": "outer", "gate_groups": 3, "up_groups": 4},
            )
        )
        with torch.no_grad():
            # routed gate_up rows: [gate(3) | up(4) | router(1)]
            routed.gate_up_proj.weight[:3] = plain.gate_proj.weight
            routed.gate_up_proj.weight[3:7] = plain.up_proj.weight
            # The lattice enumerates channels as j = c*G + r while the routed
            # outer product flattens (r, c) row-major; permute the readout.
            j = torch.arange(12)
            routed_pos = (j % 3) * 4 + (j // 3)
            routed.down_proj.weight[:, routed_pos] = plain.down_proj.weight
        x = torch.randn(6, HIDDEN)
        self.assertTrue(torch.allclose(routed(x), plain(x), atol=1e-6))

    def test_parameter_budget(self):
        mlp = self._make(4, 3, 4)
        total = sum(p.numel() for p in mlp.parameters())
        self.assertEqual(
            total, HIDDEN * (4 * (3 + 4) + 4) + HIDDEN * (3 * 4)
        )

    def test_forward_rides_the_module_dtype(self):
        mlp = self._make(2, 3, 4, router="gate_slot").to(torch.bfloat16)
        x = torch.randn(4, HIDDEN, dtype=torch.bfloat16)
        self.assertEqual(mlp(x).dtype, torch.bfloat16)

    def test_qat_quantizes_both_projections(self):
        from specforge.layers.wxay import (
            QuantizedLinear,
            replace_linear_with_quantized,
        )

        mlp = self._make(4, 3, 4)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            replace_linear_with_quantized(mlp, w_bit=4, a_bit=8)
        self.assertIsInstance(mlp.gate_up_proj, QuantizedLinear)
        self.assertIsInstance(mlp.down_proj, QuantizedLinear)


if __name__ == "__main__":
    unittest.main()
