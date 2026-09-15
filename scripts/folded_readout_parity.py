# coding=utf-8
"""Cross-container serving parity for the folded softmax FFN readout.

SpecForge (training) and vLLM (serving) often live in separate images, so the
in-process check in ``tests/test_utils/test_folded_readout_serving_parity.py``
cannot always import both.  This script splits that check in two:

    # SpecForge container - the authoritative reference
    python scripts/folded_readout_parity.py emit --out /work/folded_parity.pt

    # vLLM container - copy this file in, or mount it read-only
    python folded_readout_parity.py check --fixture /work/folded_parity.pt

``check`` needs only torch plus vLLM's ``qwen3_domino.py`` (imported when vLLM
is installed, otherwise read from ``--vllm-source`` /
``VLLM_QWEN3_DOMINO_PATH`` / the sibling checkout).  Both sides print a sha256
of the implementation they used, so a passing run proves the two images agree
on the same revision.

``check`` also simulates tensor parallelism (``--tp-sim 2,4`` by default): it
builds one readout per rank, sums the per-rank partial mixtures the way the
module's all-reduce does, projects each rank's slice and compares against the
reference output.  fp32 reconstructions match to float32 rounding; bf16 ones
match to summation-order noise (relative error ~5e-3), which is the price of
splitting the accumulation across ranks.

Exit codes: 0 = parity, 1 = mismatch, 2 = serving implementation not found.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import pathlib
import sys
from types import SimpleNamespace

import torch
from torch import nn

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    # ``python scripts/folded_readout_parity.py`` puts scripts/ on sys.path,
    # not the repo root; the emit side imports specforge.
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_CONFIG = (
    REPO_ROOT / "configs" / "qwen3.6-35b-a3b-domino-dflare-verifiedBase.json"
)
KERNEL_FILE = REPO_ROOT / "specforge" / "modeling" / "draft" / "dflash_kernels.py"
SIBLING_VLLM = (
    REPO_ROOT.parent
    / "vllm"
    / "vllm"
    / "model_executor"
    / "models"
    / "qwen3_domino.py"
)
VLLM_SOURCE_ENV = "VLLM_QWEN3_DOMINO_PATH"
VLLM_RELATIVE = pathlib.Path("vllm/model_executor/models/qwen3_domino.py")

# Small structural cases that pin the chunk layout independently of the real
# draft widths: dense, the 3N -> N -> N shape, and a single repetition.
SMALL_CASES = (
    {"name": "dense", "hidden": 4, "intermediate": 4, "branches": 1, "granularity": 2},
    {"name": "3n", "hidden": 8, "intermediate": 24, "branches": 3, "granularity": 4},
    {
        "name": "single-repetition",
        "hidden": 16,
        "intermediate": 32,
        "branches": 2,
        "granularity": 16,
    },
)

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}

# bf16 partial sums are reordered by the sharded mixture (per-rank partials
# plus one extra all-reduce), so the reconstruction is expected to differ from
# the single-rank reference by summation noise rather than bit-exactly.  fp32
# must still match tightly.
DEFAULT_TOLERANCES = {
    "fp32": (1e-4, 1e-4),
    "bf16": (2e-2, 2e-2),
}


def _tolerances(args, dtype_name: str):
    atol, rtol = DEFAULT_TOLERANCES[dtype_name]
    return (
        atol if args.atol is None else args.atol,
        rtol if args.rtol is None else args.rtol,
    )


def _sha256(path: pathlib.Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    except OSError:
        return "unavailable"


def _torch_load(path: pathlib.Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # torch < 1.13 has no weights_only
        return torch.load(path, map_location="cpu")


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
            raise NotImplementedError("the folded readout is bias-free")
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.uniform_(self.weight, -0.05, 0.05)

    def forward(self, hidden_states: torch.Tensor):
        return nn.functional.linear(hidden_states, self.weight), None


def _resolve_source(source: "str | None") -> "pathlib.Path | None":
    candidate = source or os.environ.get(VLLM_SOURCE_ENV)
    if not candidate:
        candidate = SIBLING_VLLM
    path = pathlib.Path(candidate)
    if path.is_dir():
        path = path / VLLM_RELATIVE
    return path if path.is_file() else None


def _load_vllm_readout_class(source: "str | None"):
    """Return ``(class, origin, sha256)``; ``(None, reason, None)`` when missing."""

    if source is None and importlib.util.find_spec("vllm") is not None:
        try:
            from vllm.model_executor.models.qwen3_domino import (
                FoldedSoftmaxReadout as VllmFoldedSoftmaxReadout,
            )

            module = sys.modules.get(VllmFoldedSoftmaxReadout.__module__)
            source_file = getattr(module, "__file__", None)
            sha = _sha256(pathlib.Path(source_file)) if source_file else None
            return VllmFoldedSoftmaxReadout, f"imported vllm ({source_file})", sha
        except Exception:  # noqa: BLE001 - fall back to a checkout on disk
            pass

    path = _resolve_source(source)
    if path is None:
        return None, f"no vLLM readout found ({VLLM_SOURCE_ENV} unset)", None
    text = path.read_text(encoding="utf-8")
    class_def = next(
        (
            node
            for node in ast.parse(text).body
            if isinstance(node, ast.ClassDef) and node.name == "FoldedSoftmaxReadout"
        ),
        None,
    )
    if class_def is None:
        return None, f"{path} has no FoldedSoftmaxReadout class", None
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
            + ast.get_source_segment(text, class_def)
            + "\n",
            str(path),
            "exec",
        ),
        namespace,
    )
    return namespace["FoldedSoftmaxReadout"], f"source at {path}", _sha256(path)


def _build_serving_readout(readout_cls, case, dtype):
    readout = object.__new__(readout_cls)
    nn.Module.__init__(readout)
    readout.hidden_size = case["hidden_size"]
    readout.intermediate_size = case["intermediate_size"]
    readout.branches = case["branches"]
    readout.granularity = case["granularity"]
    readout.folded_size = case["intermediate_size"] // case["branches"]
    readout.repeats = readout.folded_size // case["granularity"]
    readout.tp_size = 1
    readout.tp_rank = 0
    readout.local_hidden_size = case["intermediate_size"]
    readout.local_input_size = readout.folded_size
    readout.proj = _StubRowParallelLinear(readout.folded_size, case["hidden_size"])
    readout.proj = readout.proj.to(dtype)
    with torch.no_grad():
        readout.proj.weight.copy_(case["proj_weight"].to(dtype))
    readout.fold_logits = nn.Parameter(case["fold_logits"].to(dtype).clone())
    return readout


def _build_rank_readout(readout_cls, case, dtype, rank: int, tp_size: int):
    """One tensor-parallel rank's readout, with the distributed bits by hand."""

    readout = _build_serving_readout(readout_cls, case, dtype)
    readout.tp_size = tp_size
    readout.tp_rank = rank
    readout.local_hidden_size = case["intermediate_size"] // tp_size
    readout.local_input_size = readout.folded_size // tp_size
    readout._init_tp_mixture()
    return readout


def _check_simulated_tp(readout_cls, fixture, args) -> int:
    """Reconstruct the unsharded mixture from per-rank partials.

    The module all-reduces its per-rank partial mixture, so summing the
    partials of every rank has to reproduce the reference output exactly.  This
    needs no distributed group: the ranks are built by hand and the all-reduce
    is the sum below.
    """

    if not hasattr(readout_cls, "_local_mixture"):
        print("tp simulation  : skipped (serving revision predates sharded mixture)")
        return 0

    failures = 0
    for tp_size in args.tp_sim:
        for case in fixture["cases"]:
            folded_size = case["intermediate_size"] // case["branches"]
            if (
                case["intermediate_size"] % tp_size
                or folded_size % tp_size
            ):
                continue
            dtype = DTYPES[case["dtype"]]
            gated = case["gated"].to(dtype)
            lead = gated.shape[:-1]
            local_hidden = case["intermediate_size"] // tp_size
            local_slot = folded_size // tp_size
            weight = case["proj_weight"].to(dtype)
            partials = [
                _build_rank_readout(
                    readout_cls, case, dtype, rank, tp_size
                )
                ._local_mixture(
                    gated[..., rank * local_hidden : (rank + 1) * local_hidden]
                )
                .reshape(*lead, folded_size)
                for rank in range(tp_size)
            ]
            # 1) the mixture partials are all-reduced into the full folded
            #    vector, 2) every rank projects its own input slice, 3) the
            #    row-parallel output is all-reduced, here just summed.
            mixed = torch.stack(partials).sum(dim=0)
            projected = torch.zeros(
                *lead, case["hidden_size"], dtype=weight.dtype
            )
            for rank in range(tp_size):
                slot = rank * local_slot
                projected += torch.nn.functional.linear(
                    mixed[..., slot : slot + local_slot].reshape(-1, local_slot),
                    weight[:, slot : slot + local_slot],
                ).reshape(*lead, case["hidden_size"])
            expected = case["expected"].to(dtype)
            diff = (projected.float() - expected.float()).abs().max().item()
            scale = max(expected.float().abs().max().item(), 1e-6)
            atol, rtol = _tolerances(args, case["dtype"])
            ok = torch.allclose(projected, expected, atol=atol, rtol=rtol)
            failures += 0 if ok else 1
            print(
                f"{'PASS' if ok else 'FAIL'} tp={tp_size} {case['name']:<28} "
                f"dtype={case['dtype']} max|diff|={diff:.3e} rel={diff / scale:.2e}"
            )
    return failures


def _real_case(args) -> dict:
    from specforge.modeling.draft.dflash_kernels import resolve_folded_readout

    raw = json.loads(pathlib.Path(args.config).read_text(encoding="utf-8"))
    config = SimpleNamespace(
        hidden_size=int(args.hidden_size or raw["hidden_size"]),
        intermediate_size=int(args.intermediate_size or raw["intermediate_size"]),
        dflash_config={"ffn_readout": "folded_softmax"},
    )
    if not args.branches and not args.granularity:
        # Exercise the training-side resolver so the fixture carries exactly
        # the shapes a real run would have trained with.
        resolved = resolve_folded_readout(config)
        branches = resolved["branches"] if resolved else 1
        granularity = resolved["granularity"] if resolved else 1
    else:
        branches = int(args.branches or 1)
        granularity = int(args.granularity or 1)
    return {
        "name": pathlib.Path(args.config).stem,
        "hidden": config.hidden_size,
        "intermediate": config.intermediate_size,
        "branches": branches,
        "granularity": granularity,
    }


def emit(args) -> int:
    from specforge.modeling.draft.dflash_kernels import FoldedSoftmaxReadout

    specs = [_real_case(args)]
    if not args.real_only:
        specs.extend(SMALL_CASES)

    cases = []
    for spec in specs:
        for dtype_name in args.dtypes:
            dtype = DTYPES[dtype_name]
            torch.manual_seed(args.seed)
            readout = FoldedSoftmaxReadout(
                spec["hidden"],
                spec["intermediate"],
                branches=spec["branches"],
                granularity=spec["granularity"],
            ).to(dtype)
            with torch.no_grad():
                readout.proj.weight.normal_(0.0, 0.02)
                # Non-uniform mixture: a wrong softmax axis or chunk stride
                # cannot hide behind a uniform average.
                readout.fold_logits.normal_(0.0, 1.5)
            gated = torch.randn(
                2, args.tokens, spec["intermediate"], dtype=dtype
            )
            with torch.no_grad():
                expected = readout(gated)
            cases.append(
                {
                    "name": f"{spec['name']}-{dtype_name}",
                    "dtype": dtype_name,
                    "hidden_size": spec["hidden"],
                    "intermediate_size": spec["intermediate"],
                    "branches": spec["branches"],
                    "granularity": spec["granularity"],
                    "proj_weight": readout.proj.weight.detach().cpu(),
                    "fold_logits": readout.fold_logits.detach().cpu(),
                    "gated": gated.cpu(),
                    "expected": expected.cpu(),
                }
            )
            print(
                f"emit {cases[-1]['name']:<28} "
                f"hidden={spec['hidden']} intermediate={spec['intermediate']} "
                f"branches={spec['branches']} granularity={spec['granularity']} "
                f"folded={spec['intermediate'] // spec['branches']}"
            )

    torch.save(
        {
            "format": 1,
            "kernel_sha256": _sha256(KERNEL_FILE),
            "config": str(args.config),
            "cases": cases,
        },
        args.out,
    )
    print(
        f"wrote {args.out} ({len(cases)} cases, "
        f"reference kernel sha256={_sha256(KERNEL_FILE)})"
    )
    return 0


def check(args) -> int:
    readout_cls, origin, sha = _load_vllm_readout_class(args.vllm_source)
    if readout_cls is None:
        print(f"FAILED: {origin}")
        return 2

    fixture = _torch_load(pathlib.Path(args.fixture))
    print(f"reference kernel : sha256={fixture.get('kernel_sha256')} "
          f"({fixture.get('config')})")
    print(f"serving readout  : {origin}" + (f" sha256={sha}" if sha else ""))

    failures = 0
    for case in fixture["cases"]:
        dtype = DTYPES[case["dtype"]]
        serving = _build_serving_readout(readout_cls, case, dtype)
        expected = case["expected"].to(dtype)
        gated = case["gated"].to(dtype)
        with torch.no_grad():
            got = serving(gated)
            got_token = serving(gated[0, 0])
        diff = (got.float() - expected.float()).abs().max().item()
        scale = max(expected.float().abs().max().item(), 1e-6)
        atol, rtol = _tolerances(args, case["dtype"])
        ok = torch.allclose(got, expected, atol=atol, rtol=rtol)
        ok_token = torch.allclose(
            got_token, expected[0, 0], atol=atol, rtol=rtol
        )
        failures += 0 if (ok and ok_token) else 1
        print(
            f"{'PASS' if ok and ok_token else 'FAIL'} {case['name']:<28} "
            f"dtype={case['dtype']} max|diff|={diff:.3e} rel={diff / scale:.2e}"
        )

    if args.tp_sim:
        failures += _check_simulated_tp(readout_cls, fixture, args)
    if failures:
        print(f"FAILED: {failures} parity check(s) disagree")
        return 1
    print(f"OK: {len(fixture['cases'])} cases match the training module")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    emit_parser = sub.add_parser("emit", help="write a parity fixture (training side)")
    emit_parser.add_argument("--out", type=pathlib.Path, required=True)
    emit_parser.add_argument("--config", type=pathlib.Path, default=DEFAULT_CONFIG)
    emit_parser.add_argument("--hidden-size", type=int, default=None)
    emit_parser.add_argument("--intermediate-size", type=int, default=None)
    emit_parser.add_argument("--branches", type=int, default=None)
    emit_parser.add_argument("--granularity", type=int, default=None)
    emit_parser.add_argument("--tokens", type=int, default=5)
    emit_parser.add_argument("--dtypes", default="fp32,bf16")
    emit_parser.add_argument("--seed", type=int, default=0)
    emit_parser.add_argument(
        "--real-only", action="store_true", help="skip the small structural cases"
    )
    emit_parser.set_defaults(func=emit)

    check_parser = sub.add_parser("check", help="verify a fixture (serving side)")
    check_parser.add_argument("--fixture", type=pathlib.Path, required=True)
    check_parser.add_argument("--vllm-source", default=None)
    check_parser.add_argument("--atol", type=float, default=None)
    check_parser.add_argument("--rtol", type=float, default=None)
    check_parser.add_argument(
        "--tp-sim",
        default="2,4",
        help="tensor-parallel sizes to simulate; empty string disables",
    )
    check_parser.set_defaults(func=check)

    args = parser.parse_args(argv)
    if args.command == "emit":
        args.dtypes = [item.strip() for item in args.dtypes.split(",") if item.strip()]
        unknown = [item for item in args.dtypes if item not in DTYPES]
        if unknown:
            parser.error(f"unknown dtypes {unknown}; expected fp32/bf16")
    else:
        try:
            args.tp_sim = [
                int(item) for item in str(args.tp_sim).split(",") if item.strip()
            ]
        except ValueError:
            parser.error("--tp-sim expects comma-separated integers")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
