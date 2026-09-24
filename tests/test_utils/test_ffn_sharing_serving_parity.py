# coding=utf-8
"""Serving parity for the shared gate/up SwiGLU MLP and its routed variant.

One artifact, two implementations: SpecForge's training module
(``specforge/modeling/draft/dflash_kernels.py``) and vLLM's serving module
(``vllm/model_executor/models/qwen3_domino.py``).  Both consume the same
input and the same weights and must produce the same output — a divergent
pairing index map, expert slice order or router axis is otherwise silent at
load time and only shows up as a worse acceptance length.

vLLM is imported when it is installed.  On machines without it the test
executes the classes straight out of a vLLM checkout
(``VLLM_QWEN3_DOMINO_PATH`` or the sibling ``../vllm`` directory), so the
parity check also runs on training-only boxes; it skips when neither is
available.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import pathlib
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from specforge.modeling.draft.dflash_kernels import (
    DEFAULT_DFLASH_KERNELS,
    RoutedOuterMLP as TrainingRoutedOuterMLP,
    SharedGLUMLP as TrainingSharedGLUMLP,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SIBLING_VLLM = (
    REPO_ROOT.parent
    / "vllm"
    / "vllm"
    / "model_executor"
    / "models"
    / "qwen3_domino.py"
)
VLLM_SOURCE_ENV = "VLLM_QWEN3_DOMINO_PATH"

HIDDEN = 32

# LATTICE_CASES: (pairing, gate_groups, up_groups, intermediate).
LATTICE_CASES = [
    ("nested", 64, 128, 128),  # pure gate sharing, k=2
    ("outer", 8, 16, 128),  # exactly-once lattice 8 x 16
]
# ROUTED_CASES: (experts, gate_groups, up_groups, router); the virtual
# intermediate is experts * gate_groups * up_groups.
ROUTED_CASES = [
    (1, 8, 16, "expert"),  # single expert == plain outer lattice
    (4, 2, 4, "expert"),  # one softmax scalar per expert
    (2, 4, 4, "gate_slot"),  # per-gate-slot blend across experts
    (4, 2, 4, "gate_slot"),
]


class _StubRowParallelLinear(nn.Module):
    """Stand-in for vLLM's ``RowParallelLinear`` (no distributed group needed)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.uniform_(self.weight, -0.05, 0.05)

    def forward(self, hidden_states: torch.Tensor):
        return nn.functional.linear(hidden_states, self.weight), None


class _StubMergedColumnParallelLinear(nn.Module):
    """Stand-in for vLLM's ``MergedColumnParallelLinear``.

    Mirrors the fused ``[sum(output_sizes), in]`` weight whose shard ``i``
    starts at ``sum(output_sizes[:i])`` — the layout the real stacked weight
    loader places separate checkpoint tensors into.
    """

    def __init__(
        self,
        hidden_size: int,
        output_sizes: list,
        bias: bool = False,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.output_sizes = output_sizes
        self.weight = nn.Parameter(torch.empty(sum(output_sizes), hidden_size))
        nn.init.uniform_(self.weight, -0.05, 0.05)

    def forward(self, hidden_states: torch.Tensor):
        return nn.functional.linear(hidden_states, self.weight), None


def _load_vllm_classes():
    """Return ``(SharedGLUMLP, RoutedOuterMLP, _sharing_indices, description)``
    from vLLM, or all-``None`` when unavailable."""

    if importlib.util.find_spec("vllm") is not None:
        try:
            from vllm.model_executor.models.qwen3_domino import (
                RoutedOuterMLP as VllmRoutedOuterMLP,
                SharedGLUMLP as VllmSharedGLUMLP,
                _sharing_indices as vllm_sharing_indices,
            )

            return (
                VllmSharedGLUMLP,
                VllmRoutedOuterMLP,
                vllm_sharing_indices,
                "imported vllm",
            )
        except Exception:  # noqa: BLE001 - fall back to the checkout below
            pass

    path = pathlib.Path(os.environ.get(VLLM_SOURCE_ENV, SIBLING_VLLM))
    if not path.is_file():
        return None, None, None, None
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    wanted = {"SharedGLUMLP", "RoutedOuterMLP", "_sharing_indices"}
    segments = {
        node.name: ast.get_source_segment(source, node)
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in wanted
    }
    if "SharedGLUMLP" not in segments or "RoutedOuterMLP" not in segments:
        return None, None, None, None
    namespace = {
        "nn": nn,
        "torch": torch,
        "F": nn.functional,
        "MergedColumnParallelLinear": _StubMergedColumnParallelLinear,
        "RowParallelLinear": _StubRowParallelLinear,
        "get_tensor_model_parallel_world_size": lambda: 1,
        "QuantizationConfig": object,
    }
    if "_sharing_indices" in segments:
        exec(
            compile(
                "from __future__ import annotations\n"
                + segments["_sharing_indices"]
                + "\n",
                str(path),
                "exec",
            ),
            namespace,
        )
    for name in ("SharedGLUMLP", "RoutedOuterMLP"):
        exec(
            compile(
                "from __future__ import annotations\n" + segments[name] + "\n",
                str(path),
                "exec",
            ),
            namespace,
        )
    return (
        namespace["SharedGLUMLP"],
        namespace["RoutedOuterMLP"],
        namespace.get("_sharing_indices"),
        f"vLLM source at {path}",
    )


def _draft_config(sharing, intermediate_size):
    return SimpleNamespace(
        hidden_size=HIDDEN,
        intermediate_size=intermediate_size,
        hidden_act="silu",
        dflash_config={"ffn_sharing": dict(sharing)},
    )


class TestFfnSharingServingParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (
            cls.shared_cls,
            cls.routed_cls,
            cls.sharing_indices,
            cls.origin,
        ) = _load_vllm_classes()
        if cls.shared_cls is None or cls.routed_cls is None:
            raise unittest.SkipTest(
                "needs vllm installed or a vLLM checkout with "
                f"qwen3_domino.py ({VLLM_SOURCE_ENV})"
            )
        # Plain functions stored on a class bind like methods when accessed
        # through ``self``; wrap so attribute access stays a function.
        if cls.sharing_indices is not None and not isinstance(
            cls.sharing_indices, staticmethod
        ):
            cls.sharing_indices = staticmethod(cls.sharing_indices)
        print(f"[ffn-sharing parity] serving modules from: {cls.origin}")

    def _serving_shared(self, training: TrainingSharedGLUMLP, dtype):
        serving = object.__new__(self.shared_cls)
        nn.Module.__init__(serving)
        serving.intermediate_size = training.intermediate_size
        serving.gate_groups = training.gate_groups
        serving.up_groups = training.up_groups
        serving.pairing = training.pairing
        serving.gate_up_proj = _StubMergedColumnParallelLinear(
            HIDDEN, [training.gate_groups, training.up_groups]
        ).to(dtype)
        with torch.no_grad():
            serving.gate_up_proj.weight.copy_(
                torch.cat(
                    [training.gate_proj.weight, training.up_proj.weight]
                ).to(dtype)
            )
        serving.down_proj = _StubRowParallelLinear(
            training.intermediate_size, HIDDEN
        ).to(dtype)
        with torch.no_grad():
            serving.down_proj.weight.copy_(training.down_proj.weight)
        gate_idx, up_idx = self.sharing_indices(
            training.intermediate_size,
            training.gate_groups,
            training.up_groups,
            training.pairing,
        )
        serving.register_buffer("gate_idx", gate_idx, persistent=False)
        serving.register_buffer("up_idx", up_idx, persistent=False)
        return serving

    def _serving_routed(self, training: TrainingRoutedOuterMLP, dtype):
        serving = object.__new__(self.routed_cls)
        nn.Module.__init__(serving)
        serving.intermediate_size = training.intermediate_size
        serving.experts = training.experts
        serving.gate_groups = training.gate_groups
        serving.up_groups = training.up_groups
        serving.router = training.router
        serving.router_width = training.router_width
        serving.gate_up_proj = _StubMergedColumnParallelLinear(
            HIDDEN,
            [training.gate_groups, training.up_groups] * training.experts
            + [training.router_width],
        ).to(dtype)
        with torch.no_grad():
            serving.gate_up_proj.weight.copy_(
                training.gate_up_proj.weight.to(dtype)
            )
        serving.down_proj = _StubRowParallelLinear(
            training.gate_groups * training.up_groups, HIDDEN
        ).to(dtype)
        with torch.no_grad():
            serving.down_proj.weight.copy_(training.down_proj.weight)
        return serving

    def test_lattice_outputs_match(self):
        for pairing, gate_groups, up_groups, intermediate in LATTICE_CASES:
            for dtype in (torch.float32, torch.bfloat16):
                with self.subTest(
                    pairing=pairing,
                    gate_groups=gate_groups,
                    up_groups=up_groups,
                    dtype=dtype,
                ):
                    torch.manual_seed(gate_groups * 1009 + up_groups)
                    training = DEFAULT_DFLASH_KERNELS.make_mlp(
                        _draft_config(
                            {
                                "pairing": pairing,
                                "gate_groups": gate_groups,
                                "up_groups": up_groups,
                            },
                            intermediate,
                        )
                    ).to(dtype)
                    serving = self._serving_shared(training, dtype)
                    x = torch.randn(2, 5, HIDDEN, dtype=dtype)
                    served = serving(x)
                    self.assertEqual(served.dtype, dtype)
                    self.assertTrue(
                        torch.allclose(served, training(x), atol=1e-3, rtol=1e-3)
                    )
                    self.assertTrue(
                        torch.allclose(
                            serving(x[0, 0]), training(x[0, 0]), atol=1e-3, rtol=1e-3
                        )
                    )

    def test_routed_outputs_match(self):
        for experts, gate_groups, up_groups, router in ROUTED_CASES:
            for dtype in (torch.float32, torch.bfloat16):
                with self.subTest(experts=experts, router=router, dtype=dtype):
                    torch.manual_seed(experts * 977 + gate_groups)
                    training = DEFAULT_DFLASH_KERNELS.make_mlp(
                        _draft_config(
                            {
                                "mode": "routed_outer",
                                "experts": experts,
                                "gate_groups": gate_groups,
                                "up_groups": up_groups,
                                "router": router,
                            },
                            experts * gate_groups * up_groups,
                        )
                    ).to(dtype)
                    serving = self._serving_routed(training, dtype)
                    x = torch.randn(2, 5, HIDDEN, dtype=dtype)
                    served = serving(x)
                    self.assertEqual(served.dtype, dtype)
                    self.assertTrue(
                        torch.allclose(served, training(x), atol=1e-3, rtol=1e-3)
                    )
                    self.assertTrue(
                        torch.allclose(
                            serving(x[0, 0]), training(x[0, 0]), atol=1e-3, rtol=1e-3
                        )
                    )

    def test_exported_keys_are_what_serving_expects(self):
        routed = DEFAULT_DFLASH_KERNELS.make_mlp(
            _draft_config(
                {
                    "mode": "routed_outer",
                    "experts": 4,
                    "gate_groups": 3,
                    "up_groups": 4,
                },
                48,
            )
        )
        # Training fuses experts + router into gate_up_proj and keeps a plain
        # dense down_proj — exactly the two tensors the serving loader wants.
        self.assertEqual(
            set(routed.state_dict()),
            {"gate_up_proj.weight", "down_proj.weight"},
        )
        self.assertEqual(
            routed.gate_up_proj.weight.shape, (4 * (3 + 4) + 4, HIDDEN)
        )


if __name__ == "__main__":
    unittest.main()
