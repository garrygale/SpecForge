# coding=utf-8
"""Serving parity for the shared gate/up SwiGLU MLP.

One artifact, two implementations: SpecForge's training module
(``specforge/modeling/draft/dflash_kernels.py``) and vLLM's serving module
(``vllm/model_executor/models/qwen3_domino.py``).  Both consume the same input
and the same exported weights — separate ``gate_proj``/``up_proj`` tensors on
the training side, one fused ``gate_up_proj`` on the serving side — and must
produce the same output: a divergent pairing index map or gate/up half order
is otherwise silent at load time and only shows up as a worse acceptance
length.

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
INTERMEDIATE = 128

# (pairing, gate_groups, up_groups, readout) against INTERMEDIATE=128.
# readout=None is a dense down_proj; (4, 8) is the folded softmax readout.
SHARING_LAYOUTS = (
    ("nested", 64, 128, None),  # pure gate sharing, k=2
    ("nested", 128, 64, None),  # pure up sharing, k=2
    ("nested", 64, 32, None),  # hierarchical mix (misaligned depths)
    ("outer", 8, 16, None),  # exactly-once lattice 8 x 16 = 128
    ("outer", 8, 8, None),  # lattice with one repetition (m=2)
    ("nested", 64, 128, (4, 8)),  # pure gate sharing + folded readout
    ("outer", 8, 16, (4, 8)),  # outer lattice + folded readout
)


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
        if bias:
            raise NotImplementedError("the shared MLP is bias-free")
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.uniform_(self.weight, -0.05, 0.05)

    def forward(self, hidden_states: torch.Tensor):
        return nn.functional.linear(hidden_states, self.weight), None


class _StubMergedColumnParallelLinear(nn.Module):
    """Stand-in for vLLM's ``MergedColumnParallelLinear``.

    Mirrors the parts the shared MLP relies on: a fused
    ``[sum(output_sizes), in]`` weight whose shard ``i`` starts at
    ``sum(output_sizes[:i])`` — the layout the real stacked weight loader
    places separate gate/up checkpoint tensors into.
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
        if bias:
            raise NotImplementedError("the shared MLP is bias-free")
        total = sum(output_sizes)
        self.weight = nn.Parameter(torch.empty(total, hidden_size))
        nn.init.uniform_(self.weight, -0.05, 0.05)

    def forward(self, hidden_states: torch.Tensor):
        return nn.functional.linear(hidden_states, self.weight), None


def _load_vllm_sharing_classes():
    """Return ``(SharedGLUMLP, _sharing_indices, FoldedSoftmaxReadout|None,
    description)`` from vLLM, or all-``None`` when unavailable."""

    if importlib.util.find_spec("vllm") is not None:
        try:
            from vllm.model_executor.models.qwen3_domino import (
                FoldedSoftmaxReadout as VllmFoldedSoftmaxReadout,
                SharedGLUMLP as VllmSharedGLUMLP,
                _sharing_indices as vllm_sharing_indices,
            )

            return (
                VllmSharedGLUMLP,
                vllm_sharing_indices,
                VllmFoldedSoftmaxReadout,
                "imported vllm",
            )
        except Exception:  # noqa: BLE001 - fall back to the checkout below
            pass

    path = pathlib.Path(os.environ.get(VLLM_SOURCE_ENV, SIBLING_VLLM))
    if not path.is_file():
        return None, None, None, None
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    wanted = {"SharedGLUMLP", "_sharing_indices", "FoldedSoftmaxReadout"}
    segments = {
        node.name: ast.get_source_segment(source, node)
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in wanted
    }
    if "SharedGLUMLP" not in segments or "_sharing_indices" not in segments:
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
    if "FoldedSoftmaxReadout" in segments:
        exec(
            compile(
                "from __future__ import annotations\n"
                + segments["FoldedSoftmaxReadout"]
                + "\n",
                str(path),
                "exec",
            ),
            namespace,
        )
    exec(
        compile(
            "from __future__ import annotations\n"
            + segments["_sharing_indices"]
            + "\n"
            + segments["SharedGLUMLP"]
            + "\n",
            str(path),
            "exec",
        ),
        namespace,
    )
    return (
        namespace["SharedGLUMLP"],
        namespace["_sharing_indices"],
        namespace.get("FoldedSoftmaxReadout"),
        f"vLLM source at {path}",
    )


def _draft_config(sharing, readout):
    dflash_config = {"ffn_sharing": dict(sharing)}
    if readout is not None:
        dflash_config["ffn_readout"] = {
            "mode": "folded_softmax",
            "branches": readout[0],
            "granularity": readout[1],
        }
    return SimpleNamespace(
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        hidden_act="silu",
        dflash_config=dflash_config,
    )


def _serving_mlp(
    sharing_cls,
    sharing_indices,
    readout_cls,
    *,
    pairing,
    gate_groups,
    up_groups,
    readout,
    training: TrainingSharedGLUMLP,
    dtype: torch.dtype,
):
    """Build vLLM's shared MLP around the training weights, bypassing init.

    The real ``__init__`` builds vLLM parallel linears, which need an
    initialized tensor-parallel group; the wiring here matches it 1:1 (fused
    ``[gate | up]`` halves, the serving index maps, the chosen down_proj)
    and leaves vLLM's ``shared_act``/``forward`` — the code under test —
    untouched.
    """

    serving = object.__new__(sharing_cls)
    nn.Module.__init__(serving)
    serving.intermediate_size = INTERMEDIATE
    serving.gate_groups = gate_groups
    serving.up_groups = up_groups
    serving.pairing = pairing
    serving.gate_up_proj = _StubMergedColumnParallelLinear(
        HIDDEN, [gate_groups, up_groups]
    ).to(dtype)
    with torch.no_grad():
        serving.gate_up_proj.weight.copy_(
            torch.cat(
                [training.gate_proj.weight, training.up_proj.weight]
            ).to(dtype)
        )
    if readout is None:
        serving.down_proj = _StubRowParallelLinear(
            INTERMEDIATE, HIDDEN
        ).to(dtype)
        with torch.no_grad():
            serving.down_proj.weight.copy_(training.down_proj.weight)
    else:
        branches, granularity = readout
        serving.down_proj = readout_cls(
            hidden_size=HIDDEN,
            intermediate_size=INTERMEDIATE,
            branches=branches,
            granularity=granularity,
        ).to(dtype)
        with torch.no_grad():
            serving.down_proj.proj.weight.copy_(training.down_proj.proj.weight)
            serving.down_proj.fold_logits.copy_(training.down_proj.fold_logits)
    gate_idx, up_idx = sharing_indices(INTERMEDIATE, gate_groups, up_groups, pairing)
    serving.register_buffer("gate_idx", gate_idx, persistent=False)
    serving.register_buffer("up_idx", up_idx, persistent=False)
    return serving


class TestFfnSharingServingParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (
            cls.sharing_cls,
            sharing_indices,
            cls.readout_cls,
            cls.origin,
        ) = _load_vllm_sharing_classes()
        if cls.sharing_cls is None:
            raise unittest.SkipTest(
                "needs vllm installed or a vLLM checkout with "
                f"qwen3_domino.py ({VLLM_SOURCE_ENV})"
            )
        # Plain functions stored on a class bind like methods when accessed
        # through ``self``; wrap so ``self.sharing_indices`` stays a function.
        cls.sharing_indices = staticmethod(sharing_indices)
        print(f"[ffn-sharing parity] serving module from: {cls.origin}")

    def test_outputs_match_for_every_layout(self):
        for pairing, gate_groups, up_groups, readout in SHARING_LAYOUTS:
            if readout is not None and self.readout_cls is None:
                self.skipTest("serving FoldedSoftmaxReadout unavailable")
            for dtype in (torch.float32, torch.bfloat16):
                with self.subTest(
                    pairing=pairing,
                    gate_groups=gate_groups,
                    up_groups=up_groups,
                    readout=readout,
                    dtype=dtype,
                ):
                    torch.manual_seed(
                        gate_groups * 1009 + up_groups + (readout[0] if readout else 0)
                    )
                    sharing = {
                        "pairing": pairing,
                        "gate_groups": gate_groups,
                        "up_groups": up_groups,
                    }
                    training = DEFAULT_DFLASH_KERNELS.make_mlp(
                        _draft_config(sharing, readout)
                    )
                    self.assertIsInstance(training, TrainingSharedGLUMLP)
                    if readout is not None:
                        # Non-uniform mixture: a wrong softmax axis cannot
                        # hide behind a uniform average.
                        with torch.no_grad():
                            training.down_proj.fold_logits.normal_(std=1.5)
                    training = training.to(dtype)
                    serving = _serving_mlp(
                        self.sharing_cls,
                        self.sharing_indices,
                        self.readout_cls,
                        pairing=pairing,
                        gate_groups=gate_groups,
                        up_groups=up_groups,
                        readout=readout,
                        training=training,
                        dtype=dtype,
                    )

                    x = torch.randn(2, 5, HIDDEN, dtype=dtype)
                    served = serving(x)
                    self.assertEqual(served.dtype, dtype)
                    self.assertTrue(
                        torch.allclose(
                            served, training(x), atol=1e-3, rtol=1e-3
                        )
                    )
                    # A single-token step must agree too.
                    self.assertTrue(
                        torch.allclose(
                            serving(x[0, 0]), training(x[0, 0]), atol=1e-3, rtol=1e-3
                        )
                    )

    def test_exported_keys_are_what_serving_expects(self):
        training = DEFAULT_DFLASH_KERNELS.make_mlp(
            _draft_config({"gate_groups": 64}, None)
        )
        keys = set(training.state_dict())
        self.assertEqual(
            keys, {"gate_proj.weight", "up_proj.weight", "down_proj.weight"}
        )

        serving = _serving_mlp(
            self.sharing_cls,
            self.sharing_indices,
            self.readout_cls,
            pairing="nested",
            gate_groups=64,
            up_groups=INTERMEDIATE,
            readout=None,
            training=training,
            dtype=torch.float32,
        )
        # vLLM fuses the halves into one gate_up_proj whose stacked weight
        # loader places the separate exported tensors by their own sizes;
        # the index maps are derived, not checkpointed.
        self.assertEqual(
            {name for name, _ in serving.named_parameters()},
            {"gate_up_proj.weight", "down_proj.weight"},
        )
        self.assertEqual(set(serving.state_dict()), {"gate_up_proj.weight", "down_proj.weight"})

        folded = DEFAULT_DFLASH_KERNELS.make_mlp(
            _draft_config({"gate_groups": 64}, (4, 8))
        )
        serving_folded = _serving_mlp(
            self.sharing_cls,
            self.sharing_indices,
            self.readout_cls,
            pairing="nested",
            gate_groups=64,
            up_groups=INTERMEDIATE,
            readout=(4, 8),
            training=folded,
            dtype=torch.float32,
        )
        self.assertEqual(
            {name for name, _ in serving_folded.named_parameters()},
            {
                "gate_up_proj.weight",
                "down_proj.proj.weight",
                "down_proj.fold_logits",
            },
        )


if __name__ == "__main__":
    unittest.main()
