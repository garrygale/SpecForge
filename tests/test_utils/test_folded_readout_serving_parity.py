# coding=utf-8
"""Serving parity for the folded softmax FFN readout.

One artifact, two implementations: SpecForge's training module
(``specforge/modeling/draft/dflash_kernels.py``) and vLLM's serving module
(``vllm/model_executor/models/qwen3_domino.py``).  Both consume the same gated
hidden and the same weights, and must produce the same output — a divergent
chunk layout, softmax axis, granularity wrap or projection transpose is
otherwise silent at load time and only shows up as a worse acceptance length.

vLLM is imported when it is installed.  On machines without it the test
executes the readout class straight out of a vLLM checkout
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
    FoldedSoftmaxMLP,
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

# (hidden, intermediate, branches, granularity)
LAYOUTS = (
    (4, 4, 1, 2),  # degenerate: dense readout
    (8, 24, 3, 4),  # the 3N -> N -> N shape of the design note
    (32, 128, 4, 8),  # 4 chunks, four repetitions of the shared pattern
    (16, 32, 2, 16),  # a single repetition: catches off-by-one wraps
)


class _StubRowParallelLinear(nn.Module):
    """Stand-in for vLLM's ``RowParallelLinear`` (no distributed group needed).

    Mirrors the parts the readout relies on: a ``[out, in]`` weight and a
    ``(output, bias)`` return value.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if bias:
            raise NotImplementedError("the folded readout is bias-free")
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.uniform_(self.weight, -0.05, 0.05)

    def forward(self, hidden_states: torch.Tensor):
        return nn.functional.linear(hidden_states, self.weight), None


def _load_vllm_readout_class():
    """Return ``(class, description)`` for vLLM's folded readout, or ``(None, None)``."""

    if importlib.util.find_spec("vllm") is not None:
        try:
            from vllm.model_executor.models.qwen3_domino import (
                FoldedSoftmaxReadout as VllmFoldedSoftmaxReadout,
            )

            return VllmFoldedSoftmaxReadout, "imported vllm"
        except Exception:  # noqa: BLE001 - fall back to the checkout below
            pass

    path = pathlib.Path(os.environ.get(VLLM_SOURCE_ENV, SIBLING_VLLM))
    if not path.is_file():
        return None, None
    source = path.read_text(encoding="utf-8")
    class_def = next(
        (
            node
            for node in ast.parse(source).body
            if isinstance(node, ast.ClassDef) and node.name == "FoldedSoftmaxReadout"
        ),
        None,
    )
    if class_def is None:
        return None, None
    namespace = {
        "nn": nn,
        "torch": torch,
        "RowParallelLinear": _StubRowParallelLinear,
        "get_tensor_model_parallel_world_size": lambda: 1,
        "QuantizationConfig": object,
    }
    exec(
        compile(
            "from __future__ import annotations\n"
            + ast.get_source_segment(source, class_def)
            + "\n",
            str(path),
            "exec",
        ),
        namespace,
    )
    return namespace["FoldedSoftmaxReadout"], f"vLLM source at {path}"


def _draft_config(
    hidden_size: int, intermediate_size: int, branches: int, granularity: int
):
    return SimpleNamespace(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        hidden_act="silu",
        dflash_config={
            "ffn_readout": {
                "mode": "folded_softmax",
                "branches": branches,
                "granularity": granularity,
            }
        },
    )


def _serving_readout(
    readout_cls,
    *,
    hidden_size: int,
    intermediate_size: int,
    branches: int,
    granularity: int,
    weight: torch.Tensor,
    logits: torch.Tensor,
):
    """Build vLLM's readout with the exported weights, bypassing ``__init__``.

    The real ``__init__`` builds a ``RowParallelLinear``, which needs an
    initialized tensor-parallel group; attribute wiring here matches it 1:1 and
    leaves vLLM's ``forward`` — the code under test — untouched.
    """

    readout = object.__new__(readout_cls)
    # ``object.__new__`` skips ``nn.Module.__init__``, so set up the parameter
    # registries before assigning modules/parameters.
    nn.Module.__init__(readout)
    readout.hidden_size = hidden_size
    readout.intermediate_size = intermediate_size
    readout.branches = branches
    readout.granularity = granularity
    readout.folded_size = intermediate_size // branches
    readout.repeats = readout.folded_size // granularity
    readout.proj = _StubRowParallelLinear(readout.folded_size, hidden_size).to(
        weight.dtype
    )
    with torch.no_grad():
        readout.proj.weight.copy_(weight)
    readout.fold_logits = nn.Parameter(logits.clone())
    return readout


class TestFoldedReadoutServingParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.readout_cls, cls.origin = _load_vllm_readout_class()
        if cls.readout_cls is None:
            raise unittest.SkipTest(
                "needs vllm installed or a vLLM checkout with "
                f"qwen3_domino.py ({VLLM_SOURCE_ENV})"
            )
        print(f"[folded-readout parity] serving module from: {cls.origin}")

    def test_outputs_match_for_every_layout(self):
        for hidden, intermediate, branches, granularity in LAYOUTS:
            # Serving runs bf16, training builds fp32 first; both must agree.
            for dtype in (torch.float32, torch.bfloat16):
                with self.subTest(
                    hidden=hidden,
                    intermediate=intermediate,
                    branches=branches,
                    granularity=granularity,
                    dtype=dtype,
                ):
                    torch.manual_seed(hidden * 1000 + intermediate)
                    mlp = DEFAULT_DFLASH_KERNELS.make_mlp(
                        _draft_config(hidden, intermediate, branches, granularity)
                    )
                    self.assertIsInstance(mlp, FoldedSoftmaxMLP)
                    with torch.no_grad():
                        # Non-uniform mixture: a wrong softmax axis or chunk
                        # stride cannot hide behind a uniform average.
                        mlp.down_proj.fold_logits.normal_(std=1.5)
                    mlp = mlp.to(dtype)

                    readout = mlp.down_proj
                    serving = _serving_readout(
                        self.readout_cls,
                        hidden_size=hidden,
                        intermediate_size=intermediate,
                        branches=branches,
                        granularity=granularity,
                        weight=readout.proj.weight.detach(),
                        logits=readout.fold_logits.detach(),
                    )
                    self.assertEqual(
                        set(dict(serving.named_parameters())),
                        {"proj.weight", "fold_logits"},
                    )

                    gated = torch.randn(2, 5, intermediate, dtype=dtype)
                    served = serving(gated)
                    self.assertEqual(served.dtype, dtype)
                    self.assertTrue(
                        torch.allclose(
                            served, readout(gated), atol=1e-3, rtol=1e-3
                        )
                    )
                    # A single-token step must agree too.
                    self.assertTrue(
                        torch.allclose(
                            serving(gated[0, 0]),
                            readout(gated[0, 0]),
                            atol=1e-3,
                            rtol=1e-3,
                        )
                    )

    def test_exported_keys_are_what_serving_expects(self):
        mlp = DEFAULT_DFLASH_KERNELS.make_mlp(_draft_config(8, 24, 3, 4))
        keys = set(mlp.state_dict())
        self.assertIn("down_proj.proj.weight", keys)
        self.assertIn("down_proj.fold_logits", keys)

        serving = _serving_readout(
            self.readout_cls,
            hidden_size=8,
            intermediate_size=24,
            branches=3,
            granularity=4,
            weight=mlp.down_proj.proj.weight.detach(),
            logits=mlp.down_proj.fold_logits.detach(),
        )
        # vLLM exposes these two names under the same ``mlp.down_proj.`` path,
        # so the exported artifact loads without a new weight mapper.
        self.assertEqual(
            {name for name, _ in serving.named_parameters()},
            {"proj.weight", "fold_logits"},
        )


if __name__ == "__main__":
    unittest.main()
