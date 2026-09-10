#!/usr/bin/env python3
"""Acceptance-length evaluation for DFlash/Domino draft models.

This is the standalone in-process evaluator used to measure draft acceptance
without launching a vLLM service. It supports HumanEval, GSM8K, MATH-500,
MBPP, data/tensor parallelism across NPUs, and optional NPU W8A8/W4A4/mixed
inference quantization.

Parallelism
-----------
``--dp`` (data parallel) shards the benchmark problems, ``--tp`` (tensor
parallel) shards the target model, and the job runs on ``dp * tp`` NPUs:

* global rank ``r`` binds NPU ``r`` and maps to DP rank ``r // tp`` and TP
  rank ``r % tp``;
* every DP rank loads a private draft model and evaluates the problems with
  ``index % dp == dp_rank``;
* the ``tp`` ranks of a DP group cooperate on every target forward, so they
  evaluate the same problems in lockstep and only TP rank 0 writes results.

``--dp``/``--tp`` values above one make the first process launch the
``dp * tp`` workers itself unless the job was started by ``torchrun`` (that
is, ``RANK`` is in the environment). Quantization stays a draft-model feature,
so ``--quantize`` composes with a tensor-parallel target.

Usage::

    # 8 NPUs, eight independent data-parallel workers
    python scripts/check_acceptance.py --draft-path D --target-path T --dp 8

    # 4 NPUs, target sharded 2-way inside two data-parallel groups
    python scripts/check_acceptance.py --draft-path D --target-path T \\
        --dp 2 --tp 2 --quantize w8a8

    # merge the per-DP logs written by the run above
    python inference/merge_results.py <timestamp>
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
from datetime import datetime
from typing import Any, Optional

import torch
import torch.distributed as dist
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from specforge.modeling.auto import AutoDraftModel

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_humaneval(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_gsm8k(path: str) -> list[dict[str, Any]]:
    return load_humaneval(path)


def load_math500(path: str) -> list[dict[str, Any]]:
    return load_humaneval(path)


def build_mbpp_prompt(text: str, test_list: list[str]) -> str:
    tests = "\n".join(test_list)
    return (
        "You are an expert Python programmer, and here is your task: "
        f"{text} Your code should pass these tests:\n\n{tests}\n\n[BEGIN]\n"
    )


def load_mbpp(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)
    problems = []
    for item in items:
        text = item.get("prompt") or item.get("text") or ""
        test_list = item.get("test_list") or []
        problems.append({**item, "prompt": build_mbpp_prompt(text, test_list)})
    return problems


def aggregate_stats(
    per_problem_results: list[dict[str, Any]],
) -> tuple[Optional[float], Optional[float]]:
    valid = [
        r
        for r in per_problem_results
        if r.get("mean_acceptance_length") is not None
        and r.get("num_complete_blocks", 0) > 0
    ]
    if not valid:
        return None, None
    simple_mean = sum(r["mean_acceptance_length"] for r in valid) / len(valid)
    weights = [r["num_complete_blocks"] for r in valid]
    weighted_mean = (
        sum(r["mean_acceptance_length"] * w for r, w in zip(valid, weights))
        / sum(weights)
    )
    return simple_mean, weighted_mean


def _import_torch_npu():
    """Import the Ascend adapter on demand so ``torch.npu`` becomes visible."""
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        return None
    return getattr(torch, "npu", None)


def _ensure_accelerator_import() -> None:
    """Register the Ascend adapter before device detection or dist startup."""
    if not hasattr(torch, "npu"):
        _import_torch_npu()


def _npu_available() -> bool:
    npu = getattr(torch, "npu", None) or _import_torch_npu()
    return bool(npu is not None and npu.is_available())


def _resolve_device(npu_id: Optional[int] = None) -> str:
    if _npu_available():
        if npu_id is not None:
            torch.npu.set_device(npu_id)
            return f"npu:{npu_id}"
        # Distributed startup already bound this process to its local NPU.
        return "npu"
    if torch.cuda.is_available():
        if npu_id is None:
            return "cuda"
        device_index = npu_id % torch.cuda.device_count()
        torch.cuda.set_device(device_index)
        return f"cuda:{device_index}"
    if npu_id is not None:
        print(
            f"[check_acceptance] warning: no accelerator available, ignoring "
            f"requested device index {npu_id}",
            flush=True,
        )
    return "cpu"


def _dtensor_types():
    """Return ``(DTensor, Replicate)`` when this torch build ships DTensor."""
    try:
        from torch.distributed.tensor import DTensor, Replicate
    except Exception:  # torch without torch.distributed.tensor
        return None, None
    return DTensor, Replicate


def _is_dtensor(value: Any) -> bool:
    dtensor_cls, _ = _dtensor_types()
    return dtensor_cls is not None and isinstance(value, dtensor_cls)


_gathered_placement_warnings: set[str] = set()


def _to_local_tensor(value: Any) -> Any:
    """Return a plain, whole tensor for a (possibly sharded) DTensor."""
    placements = tuple(getattr(value, "placements", ()) or ())
    _, replicate_cls = _dtensor_types()
    if replicate_cls is None or all(
        isinstance(placement, replicate_cls) for placement in placements
    ):
        # Replicated already: ``to_local`` is a view and needs no collective.
        return value.to_local()
    signature = ", ".join(str(placement) for placement in placements)
    if signature not in _gathered_placement_warnings:
        _gathered_placement_warnings.add(signature)
        print(
            "[check_acceptance] warning: gathering a tensor-parallel output "
            f"with placements [{signature}]; verify this output layout is the "
            "whole tensor the evaluator needs.",
            flush=True,
        )
    return value.full_tensor()


def _localize_outputs(value: Any) -> Any:
    """Replace DTensors inside a module output with plain local tensors."""
    if _is_dtensor(value):
        return _to_local_tensor(value)
    fields = getattr(value, "__dataclass_fields__", None)
    if fields is not None:
        # ModelOutput instances: mutate in place so the model keeps returning
        # its own output type (``output.logits`` and friends stay available).
        for field in fields:
            try:
                setattr(value, field, _localize_outputs(getattr(value, field)))
            except Exception:  # read-only or unavailable field
                pass
        return value
    if isinstance(value, dict):
        return {key: _localize_outputs(item) for key, item in value.items()}
    if isinstance(value, tuple):
        items = [_localize_outputs(item) for item in value]
        try:
            return type(value)(*items)
        except Exception:
            return tuple(items)
    if isinstance(value, list):
        return [_localize_outputs(item) for item in value]
    return value


def _install_output_localizers(target) -> list[str]:
    """Keep DTensors out of the draft model's forward path.

    Tensor-parallel layers return ``DTensor`` outputs. The acceptance loop
    indexes, concatenates and samples those tensors, and feeds them to the
    draft model that is replicated on every rank, so every target output the
    loop touches has to be a plain local tensor. Forward hooks convert them as
    the target produces them, including the direct embedding and ``lm_head``
    calls that ``spec_generate`` performs.
    """

    def _hook(module, args, output):
        return _localize_outputs(output)

    hooked: list[str] = []
    seen: set[int] = set()
    candidates = (
        ("target", lambda: target),
        ("text_model", lambda: _target_text_model(target)),
        ("embed_tokens", lambda: _target_embed_tokens(target)),
        ("lm_head", lambda: _target_lm_head(target)),
    )
    for name, resolve in candidates:
        try:
            module = resolve()
        except Exception:
            continue
        if not isinstance(module, torch.nn.Module) or id(module) in seen:
            continue
        seen.add(id(module))
        module.register_forward_hook(_hook)
        hooked.append(name)
    return hooked


def _tp_device_mesh(tp_size: int):
    """Return the 1-D ``tp`` mesh that the target model is sharded over.

    The mesh comes from the process group ``specforge.distributed`` already
    created for the ``tp`` dimension, so the target's collectives and the
    framework's own tensor-parallel layers share one group.
    """
    from specforge.distributed import get_tp_device_mesh, get_tp_group
    from specforge.utils import get_device_type

    tp_group = get_tp_group()
    device_type = get_device_type()
    if tp_group is not None:
        try:
            return dist.DeviceMesh.from_group(
                tp_group, device_type=device_type, mesh_dim_names=("tp",)
            )
        except TypeError:
            # torch builds whose ``from_group`` predates mesh dimension names.
            return dist.DeviceMesh.from_group(tp_group, device_type=device_type)
    mesh = get_tp_device_mesh()
    if mesh is None or _mesh_size(mesh) != tp_size:
        raise RuntimeError(
            "tensor parallel requires an initialized tp process group; start "
            "the job with --dp/--tp (or torchrun) instead of a bare worker"
        )
    return mesh


def _load_target_model_tp(target_path: str, dp_size: int, tp_size: int, torch_dtype):
    """Load the target sharded across the ``tp`` ranks of one DP group.

    ``tp_plan="auto"`` lets transformers shard the checkpoint while loading, so
    a target that does not fit on a single NPU is never materialized in full.
    """
    device_mesh = _tp_device_mesh(tp_size)
    attempts = [
        ("tp_plan with an explicit tp submesh", {"device_mesh": device_mesh}),
    ]
    if dp_size == 1:
        # With a single DP group the tp submesh is the whole job, so a build
        # that derives the mesh on its own still shards correctly.
        attempts.append(("tp_plan with an implicit mesh", {}))

    errors: list[str] = []
    last_error: Optional[BaseException] = None
    for label, extra in attempts:
        try:
            return AutoModelForCausalLM.from_pretrained(
                target_path,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                tp_plan="auto",
                **extra,
            )
        except TypeError as exc:
            last_error = exc
            errors.append(f"{label} (AutoModelForCausalLM): {exc}")
            if "unexpected keyword argument" not in str(exc):
                # A real failure inside the model, not a kwarg miss.
                break
        except Exception as exc:
            last_error = exc
            errors.append(f"{label} (AutoModelForCausalLM): {exc}")
            try:
                return AutoModel.from_pretrained(
                    target_path,
                    torch_dtype=torch_dtype,
                    trust_remote_code=True,
                    tp_plan="auto",
                    **extra,
                )
            except Exception as fallback_exc:
                last_error = fallback_exc
                errors.append(f"{label} (AutoModel): {fallback_exc}")

    detail = "\n".join(f"  - {error}" for error in errors)
    raise RuntimeError(
        f"could not load the target with tensor parallel size {tp_size} "
        f"(dp={dp_size}):\n{detail}\n"
        "  hint: --tp > 1 needs a transformers build that supports tensor "
        "parallelism (tp_plan) for this architecture; on an out-of-memory "
        "failure raise --tp (more NPUs per target copy) or lower --dp, "
        "otherwise fall back to --tp 1."
    ) from last_error


def _mesh_size(mesh) -> Optional[int]:
    """Return the number of ranks in a device mesh, when it is readable."""
    try:
        return int(mesh.mesh.numel())
    except Exception:
        return None


def _assert_tp_sharded(model, tp_size: int) -> None:
    """Fail loudly when tensor-parallel loading did not shard as requested."""
    dtensor_cls, replicate_cls = _dtensor_types()
    if dtensor_cls is None:
        raise RuntimeError(
            "tensor parallel requires torch.distributed.tensor (DTensor), which "
            "this torch build does not provide"
        )
    for parameter in model.parameters():
        if not isinstance(parameter, dtensor_cls):
            continue
        if all(
            isinstance(placement, replicate_cls) for placement in parameter.placements
        ):
            continue
        sharded_over = _mesh_size(getattr(parameter, "device_mesh", None))
        if sharded_over is not None and sharded_over != tp_size:
            raise RuntimeError(
                f"the target is sharded over {sharded_over} ranks instead of "
                f"--tp {tp_size}: this transformers build ignored the (dp, tp) "
                "device mesh, so a dp > 1 run would produce wrong results. Use "
                f"--dp 1 --tp {tp_size} (the implicit mesh then matches), or "
                "use a transformers build that accepts an explicit device mesh."
            )
        return
    print(
        "[check_acceptance] warning: no DTensor parameters were found after "
        f"loading with tp_plan='auto'; the target may be replicated instead of "
        f"sharded over {tp_size} ranks, which saves no memory. Verify the "
        "per-rank memory usage and transformers support for this architecture.",
        flush=True,
    )


def _target_vocab_size(target) -> Optional[int]:
    """Return the language-model vocabulary size, if the config exposes it."""
    config = getattr(target, "config", None)
    vocab_size = getattr(config, "vocab_size", None)
    if vocab_size is None:
        text_config = getattr(config, "text_config", None)
        vocab_size = getattr(text_config, "vocab_size", None)
    return int(vocab_size) if isinstance(vocab_size, int) else None


def _verify_tp_outputs(target) -> None:
    """Probe one token to catch tensor-parallel outputs the loop cannot use.

    Sampling and hidden-state extraction need *whole* tensors: a plan that
    leaves the vocabulary sharded, or an output the localizers cannot reach,
    would silently change the accepted-token statistics. Every rank of a TP
    group runs this probe once, in lockstep.
    """
    try:
        text_model = _target_text_model(target)
        device = _target_device(target)
        probe_ids = torch.ones((1, 1), dtype=torch.long, device=device)
        with torch.inference_mode():
            output = text_model(probe_ids, logits_to_keep=1)
    except Exception as exc:
        print(
            f"[check_acceptance] warning: could not probe the tensor-parallel "
            f"target ({exc}); continuing.",
            flush=True,
        )
        return

    logits = getattr(output, "logits", None)
    if logits is None:
        print(
            "[check_acceptance] warning: the tensor-parallel target forward "
            "returned no logits for a one-token probe; continuing.",
            flush=True,
        )
        return
    if _is_dtensor(logits):
        raise RuntimeError(
            "the tensor-parallel target still returns DTensor logits after the "
            "output localizers were installed; the acceptance loop needs plain "
            "tensors. Report this model/transformers combination."
        )
    vocab_size = _target_vocab_size(target)
    if vocab_size is not None and logits.shape[-1] != vocab_size:
        raise RuntimeError(
            f"tensor-parallel logits are not gathered: last dimension is "
            f"{logits.shape[-1]} but the target vocabulary has {vocab_size} "
            "tokens. Sampling a sharded vocabulary would corrupt the acceptance "
            "statistics, so this run is stopped; use --tp 1 for this model."
        )


def _load_target_model(
    target_path: str,
    device: str,
    torch_dtype,
    *,
    dp_size: int = 1,
    tp_size: int = 1,
):
    if tp_size > 1:
        model = _load_target_model_tp(target_path, dp_size, tp_size, torch_dtype)
        _assert_tp_sharded(model, tp_size)
        hooked = _install_output_localizers(model)
        print(
            f"Target loaded with tensor parallel size {tp_size} "
            f"(output localizers attached to: {', '.join(hooked) or 'none'})",
            flush=True,
        )
        _verify_tp_outputs(model)
    else:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                target_path,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )
        except Exception:
            model = AutoModel.from_pretrained(
                target_path,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )
        model.to(device)
    model.eval()
    return model


def _target_text_model(target):
    get_language_model = getattr(target, "get_language_model", None)
    if callable(get_language_model):
        language_model = get_language_model()
        if language_model is not None:
            return language_model
    language_model = getattr(target, "language_model", None)
    if language_model is not None:
        return language_model
    candidate = getattr(target, "model", None)
    if candidate is not None and getattr(candidate, "language_model", None) is not None:
        return candidate.language_model
    if (
        candidate is not None
        and getattr(candidate, "embed_tokens", None) is not None
        and getattr(candidate, "lm_head", None) is not None
    ):
        return candidate
    return target


def _target_embed_tokens(target):
    """Resolve the embedding module ``spec_generate`` calls directly."""
    text_model = _target_text_model(target)
    embed_tokens = getattr(text_model, "embed_tokens", None)
    if embed_tokens is not None:
        return embed_tokens
    return getattr(text_model.model, "embed_tokens")


def _target_lm_head(target):
    """Resolve the LM head ``spec_generate`` calls directly."""
    text_model = _target_text_model(target)
    lm_head = getattr(text_model, "lm_head", None)
    if lm_head is not None:
        return lm_head
    return getattr(target, "lm_head")


def _target_device(target):
    """Return the device holding the target's parameters."""
    try:
        return next(target.parameters()).device
    except (AttributeError, StopIteration, TypeError):
        return "cpu"


def _apply_quantize(model, quantize: Optional[str]) -> None:
    if quantize not in ("w8a8", "w4a4", "mixed"):
        return
    method_config = getattr(model.config, "dflash_config", None) or {}
    qat_exclude = list(method_config.get("qat_exclude", []) or [])
    w4a4_layers = set(method_config.get("qat_w4a4_layers", []) or [])
    stochastic_weight = bool(method_config.get("stochastic_weight", False))
    if quantize == "w8a8":
        from specforge.layers.npu_w8a8 import replace_linear_with_npu_w8a8

        replace_linear_with_npu_w8a8(
            model,
            w_bit=8,
            exclude_names=qat_exclude,
            stochastic_weight=stochastic_weight,
        )
    elif quantize == "w4a4":
        from specforge.layers.npu_w4a4 import replace_linear_with_npu_w4a4

        replace_linear_with_npu_w4a4(
            model,
            w_bit=4,
            exclude_names=qat_exclude + list(w4a4_layers),
            stochastic_weight=stochastic_weight,
        )
        if w4a4_layers:
            replace_linear_with_npu_w4a4(
                model,
                w_bit=4,
                include_only=w4a4_layers,
                stochastic_weight=stochastic_weight,
            )
    elif quantize == "mixed":
        from specforge.layers.npu_w4a4 import replace_linear_with_npu_w4a4
        from specforge.layers.npu_w8a8 import replace_linear_with_npu_w8a8

        replace_linear_with_npu_w8a8(
            model,
            w_bit=8,
            exclude_names=qat_exclude + list(w4a4_layers),
            stochastic_weight=stochastic_weight,
        )
        if w4a4_layers:
            replace_linear_with_npu_w4a4(
                model,
                include_only=w4a4_layers,
                stochastic_weight=stochastic_weight,
            )


def run_acceptance_check(
    draft_path: str,
    target_path: str,
    humaneval_path: str,
    *,
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
    use_chat_template: bool = True,
    universal_prefix: str = "",
    include_incomplete_blocks: bool = False,
    npu_id: Optional[int] = None,
    dp_size: int = 1,
    tp_size: int = 1,
    dp_rank: int = 0,
    tp_rank: int = 0,
    timestamp: Optional[str] = None,
    quantize: Optional[str] = None,
    benchmark: str = "humaneval",
    gsm8k_path: Optional[str] = None,
    math500_path: Optional[str] = None,
    mbpp_path: Optional[str] = None,
) -> dict[str, Any]:
    world_size = dp_size * tp_size
    is_writer = tp_rank == 0
    rank = dp_rank * tp_size + tp_rank
    device = _resolve_device(npu_id)
    torch_dtype = torch.bfloat16 if device != "cpu" else torch.float32

    draft_model = AutoDraftModel.from_pretrained(
        draft_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    draft_model.to(device)
    draft_model.eval()
    print(f"[rank {rank}] Draft loaded: block_size={draft_model.block_size}")
    if quantize:
        _apply_quantize(draft_model, quantize)
        print(f"[rank {rank}] Applied {quantize} NPU quantization to draft model.")

    target_model = _load_target_model(
        target_path,
        device,
        torch_dtype,
        dp_size=dp_size,
        tp_size=tp_size,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        target_path, trust_remote_code=True
    )
    stop_token_ids = [tokenizer.eos_token_id]
    if tokenizer.pad_token_id is not None:
        stop_token_ids.append(tokenizer.pad_token_id)

    if benchmark == "humaneval":
        problems = load_humaneval(humaneval_path)
    elif benchmark == "gsm8k":
        problems = load_gsm8k(gsm8k_path or humaneval_path)
    elif benchmark == "math500":
        problems = load_math500(math500_path or humaneval_path)
    elif benchmark == "mbpp":
        problems = load_mbpp(mbpp_path or humaneval_path)
    else:
        raise ValueError(f"unknown benchmark {benchmark}")

    per_problem = []
    for idx, prob in enumerate(problems):
        if dp_size > 1 and idx % dp_size != dp_rank:
            continue
        if tp_size > 1:
            # Tensor-parallel ranks evaluate the same problem redundantly.
            # Identical seeds keep their sampling streams identical so the
            # target's collectives stay in lockstep.
            _seed_sampling(1234 + idx)
        if benchmark == "humaneval":
            task_id = prob["task_id"]
            prompt = prob["prompt"]
        elif benchmark == "gsm8k":
            task_id = f"gsm8k_{idx}"
            prompt = f"Question: {prob['question']}\nAnswer:"
        elif benchmark == "math500":
            task_id = prob.get("unique_id") or f"math500_{idx}"
            prompt = prob["problem"]
        else:
            task_id = f"mbpp_{prob.get('task_id', idx)}"
            prompt = prob["prompt"]
        if universal_prefix:
            prompt = universal_prefix + prompt
        if use_chat_template:
            input_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            input_ids = tokenizer(
                input_text,
                return_tensors="pt",
                add_special_tokens=False,
            ).input_ids
        else:
            input_ids = tokenizer.encode(prompt, return_tensors="pt")
        input_ids = input_ids.to(device)

        try:
            _, stats = draft_model.spec_generate(
                target=target_model,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                stop_token_ids=stop_token_ids,
                temperature=temperature,
                return_acceptance_stats=True,
            )
            lengths = stats.get("acceptance_lengths", [])
            if include_incomplete_blocks:
                mean_accept = sum(lengths) / len(lengths) if lengths else None
                num_blocks = len(lengths)
            else:
                mean_accept = stats["mean_acceptance_length"]
                num_blocks = stats["num_complete_blocks"]
            result = {
                "task_id": task_id,
                "mean_acceptance_length": mean_accept,
                "num_complete_blocks": num_blocks,
                "num_incomplete_blocks": stats["num_incomplete_blocks"],
                "per_position_accuracy": stats.get("per_position_accuracy", []),
            }
        except Exception as exc:
            result = {
                "task_id": task_id,
                "mean_acceptance_length": None,
                "num_complete_blocks": 0,
                "num_incomplete_blocks": 0,
                "error": str(exc),
            }
        per_problem.append(result)

    overall_simple, overall_weighted = aggregate_stats(per_problem)
    valid_count = sum(
        1 for r in per_problem if r.get("mean_acceptance_length") is not None
    )
    per_pos = [
        r["per_position_accuracy"]
        for r in per_problem
        if r.get("per_position_accuracy")
    ]
    per_pos_avg = torch.tensor(per_pos).mean(dim=0).tolist() if per_pos else None

    timestamp = timestamp or datetime.now().strftime("%y%m%d%H%M%S")
    log_file = None
    if is_writer:
        log_file = _log_path(timestamp, world_size, dp_rank)
        with open(log_file, "w", encoding="utf-8") as f:
            for entry in per_problem:
                f.write(json.dumps(entry) + "\n")
        print(f"[rank {rank}] Wrote per-DP results to {log_file}")
    return {
        "per_problem": per_problem,
        "overall_simple": overall_simple,
        "overall_weighted": overall_weighted,
        "valid_count": valid_count,
        "total_problems": len(problems),
        "num_evaluated": len(per_problem),
        "log_file": log_file,
        "per_position_accuracy": per_pos_avg,
    }


def _seed_sampling(seed: int) -> None:
    """Give every tensor-parallel rank the same sampling stream."""
    torch.manual_seed(seed)
    for device_type in ("npu", "cuda"):
        module = getattr(torch, device_type, None)
        manual_seed_all = getattr(module, "manual_seed_all", None)
        if callable(manual_seed_all):
            manual_seed_all(seed)


def _split_world(dp_size: int, tp_size: int, rank: int) -> tuple[int, int]:
    """Map a global rank onto ``(dp_rank, tp_rank)``.

    Tensor-parallel ranks are contiguous, matching the trailing ``tp``
    dimension of the ``(dp, tp)`` device mesh.
    """
    if dp_size < 1 or tp_size < 1:
        raise ValueError(f"dp and tp must be >= 1, got dp={dp_size}, tp={tp_size}")
    world_size = dp_size * tp_size
    if not 0 <= rank < world_size:
        raise ValueError(f"rank {rank} is outside [0, {world_size})")
    return rank // tp_size, rank % tp_size


def _log_path(timestamp: str, world_size: int, dp_rank: int) -> str:
    """Return the per-DP log path written by TP rank 0.

    Single-process runs keep the historical unsuffixed name so existing
    workflows (and ``merge_results.py --files``) keep working.
    """
    suffix = f"_dp{dp_rank}" if world_size > 1 else ""
    return os.path.join(_SCRIPT_DIR, f"{timestamp}_acceptance_lengths{suffix}.jsonl")


def _parallel_sizes(args) -> tuple[int, int]:
    """Resolve ``--dp``/``--tp``, honouring the deprecated ``--num-npus``."""
    if args.dp is not None and args.num_npus is not None:
        raise SystemExit(
            "--dp and its deprecated alias --num-npus are mutually exclusive; "
            "pass --dp"
        )
    dp_size = 1 if args.dp is None else args.dp
    if args.num_npus is not None:
        print(
            "warning: --num-npus is deprecated; use --dp instead.",
            flush=True,
        )
        dp_size = args.num_npus
    tp_size = 1 if args.tp is None else args.tp
    if dp_size < 1 or tp_size < 1:
        raise SystemExit(
            f"--dp and --tp must be >= 1, got dp={dp_size}, tp={tp_size}"
        )
    return dp_size, tp_size


def _inside_job() -> bool:
    """Return whether this process already belongs to a distributed job."""
    if dist.is_available() and dist.is_initialized():
        return True
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _worker_command(
    args, rank: int, timestamp: str, dp_size: int, tp_size: int
) -> list[str]:
    """Return the command line for one worker of the ``dp * tp`` job."""
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--draft-path",
        args.draft_path,
        "--target-path",
        args.target_path,
        "--benchmark",
        args.benchmark,
        "--dp",
        str(dp_size),
        "--tp",
        str(tp_size),
        "--npu-id",
        str(rank),
        "--timestamp",
        timestamp,
        "--dist-timeout-minutes",
        str(args.dist_timeout_minutes),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--temperature",
        str(args.temperature),
    ]
    for flag, value in (
        ("--humaneval-path", args.humaneval_path),
        ("--gsm8k-path", args.gsm8k_path),
        ("--math500-path", args.math500_path),
        ("--mbpp-path", args.mbpp_path),
    ):
        if value:
            cmd += [flag, value]
    if args.quantize:
        cmd += ["--quantize", args.quantize]
    if args.universal_prefix:
        cmd += ["--universal-prefix", args.universal_prefix]
    cmd.append(
        "--use-chat-template" if args.use_chat_template else "--no-use-chat-template"
    )
    if args.include_incomplete_blocks:
        cmd.append("--include-incomplete-blocks")
    return cmd


def _launch_workers(args, dp_size: int, tp_size: int, timestamp: str) -> int:
    """Spawn one worker per NPU and wait for the ``dp * tp`` job to finish."""
    world_size = dp_size * tp_size
    env = os.environ.copy()
    env["WORLD_SIZE"] = str(world_size)
    env["MASTER_ADDR"] = env.get("MASTER_ADDR") or "127.0.0.1"
    env["MASTER_PORT"] = env.get("MASTER_PORT") or str(_free_tcp_port())
    repo_root = os.path.dirname(_SCRIPT_DIR)
    env["PYTHONPATH"] = os.pathsep.join(
        entry for entry in (repo_root, env.get("PYTHONPATH", "")) if entry
    )
    print(
        f"Launching dp={dp_size} x tp={tp_size} = {world_size} workers "
        f"(rendezvous {env['MASTER_ADDR']}:{env['MASTER_PORT']}, "
        f"timestamp {timestamp})",
        flush=True,
    )

    processes = []
    for rank in range(world_size):
        child_env = {**env, "RANK": str(rank), "LOCAL_RANK": str(rank)}
        command = _worker_command(args, rank, timestamp, dp_size, tp_size)
        processes.append(subprocess.Popen(command, env=child_env))

    try:
        return_codes = [process.wait() for process in processes]
    except KeyboardInterrupt:
        for process in processes:
            process.terminate()
        raise

    failures = [(rank, code) for rank, code in enumerate(return_codes) if code != 0]
    if failures:
        print(f"worker failures (rank, exit code): {failures}", flush=True)
        return 1
    print(
        f"All {world_size} workers finished; merge the per-DP logs with "
        f"`python inference/merge_results.py {timestamp}`",
        flush=True,
    )
    return 0


def _resolve_rank(args, world_size: int) -> int:
    """Resolve this worker's global rank from the job or the environment."""
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world = dist.get_world_size()
    elif "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world = int(os.environ.get("WORLD_SIZE", world_size))
    else:
        rank = 0 if args.npu_id is None else int(args.npu_id)
        world = world_size
    if world != world_size:
        raise SystemExit(
            f"job world size {world} does not match --dp {args.dp} x "
            f"--tp {args.tp} = {world_size}"
        )
    if not 0 <= rank < world_size:
        raise SystemExit(f"rank {rank} is outside [0, {world_size})")
    return rank


def _resolve_local_device(args, rank: int) -> Optional[int]:
    """Return the device index this worker must bind, if any."""
    _ensure_accelerator_import()
    if args.npu_id is not None:
        return int(args.npu_id)
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None:
        return int(local_rank)
    if _npu_available() or torch.cuda.is_available():
        return rank
    return None


def _prepare_distributed_env(rank: int, world_size: int) -> None:
    """Fill in rendezvous defaults for hand-launched tensor-parallel workers."""
    os.environ.setdefault("RANK", str(rank))
    os.environ.setdefault("WORLD_SIZE", str(world_size))
    os.environ.setdefault("LOCAL_RANK", str(rank))
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")


def _init_tp_process_group(args, dp_size: int, tp_size: int) -> None:
    """Initialize the ``(dp, tp)`` process groups used by the target."""
    from specforge.distributed import init_distributed

    init_distributed(timeout=args.dist_timeout_minutes, tp_size=tp_size)


def _destroy_process_group() -> None:
    """Tear down the tensor-parallel groups without masking a failure."""
    try:
        from specforge.distributed import destroy_distributed

        destroy_distributed()
    except Exception as exc:
        print(
            f"[check_acceptance] warning: process group teardown failed: {exc}",
            flush=True,
        )


def _run_worker(args, world_size: int, dp_size: int, tp_size: int) -> int:
    """Evaluate this worker's share of the problems and return an exit code."""
    rank = _resolve_rank(args, world_size)
    dp_rank, tp_rank = _split_world(dp_size, tp_size, rank)
    npu_id = _resolve_local_device(args, rank)
    if tp_size > 1:
        _prepare_distributed_env(rank, world_size)
        # HCCL requires the Ascend adapter to be registered (and the local
        # device to be bound) before the process group is created.
        _ensure_accelerator_import()
        _init_tp_process_group(args, dp_size, tp_size)

    try:
        stats = run_acceptance_check(
            draft_path=args.draft_path,
            target_path=args.target_path,
            humaneval_path=args.humaneval_path,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            use_chat_template=args.use_chat_template,
            universal_prefix=args.universal_prefix,
            include_incomplete_blocks=args.include_incomplete_blocks,
            npu_id=npu_id,
            dp_size=dp_size,
            tp_size=tp_size,
            dp_rank=dp_rank,
            tp_rank=tp_rank,
            timestamp=args.timestamp,
            quantize=args.quantize,
            benchmark=args.benchmark,
            gsm8k_path=args.gsm8k_path,
            math500_path=args.math500_path,
            mbpp_path=args.mbpp_path,
        )
    finally:
        if tp_size > 1:
            _destroy_process_group()

    if tp_rank != 0:
        # Only TP rank 0 owns results; the other ranks just mirrored the loop.
        return 0
    print(
        f"[rank {rank}] dp_rank={dp_rank} tp_rank={tp_rank} "
        f"evaluated {stats['num_evaluated']}/{stats['total_problems']} problems"
    )
    print(f"[rank {rank}] Mean acceptance length (simple):   {stats['overall_simple']}")
    print(f"[rank {rank}] Mean acceptance length (weighted): {stats['overall_weighted']}")
    failures = [entry for entry in stats["per_problem"] if entry.get("error")]
    if stats["valid_count"] == 0 and failures:
        print(
            f"[rank {rank}] every problem failed, first error: "
            f"{failures[0]['error']}"
        )
        return 1
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--draft-path", required=True)
    parser.add_argument("--target-path", required=True)
    parser.add_argument(
        "--humaneval-path",
        default=os.path.join(_SCRIPT_DIR, "human-eval-v2-20210705.jsonl"),
    )
    parser.add_argument(
        "--gsm8k-path",
        default=os.path.join(_SCRIPT_DIR, "gsm8k_test.jsonl"),
    )
    parser.add_argument(
        "--math500-path",
        default=os.path.join(_SCRIPT_DIR, "math500-test.jsonl"),
    )
    parser.add_argument(
        "--mbpp-path",
        default=os.path.join(_SCRIPT_DIR, "sanitized-mbpp.json"),
    )
    parser.add_argument("--benchmark", default="humaneval", choices=["humaneval", "gsm8k", "math500", "mbpp"])
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--use-chat-template", action="store_true", default=True)
    parser.add_argument("--no-use-chat-template", action="store_false", dest="use_chat_template")
    parser.add_argument("--universal-prefix", default="")
    parser.add_argument("--include-incomplete-blocks", action="store_true")
    parser.add_argument(
        "--dp",
        type=int,
        default=None,
        help="data parallel size: problems are sharded over dp ranks "
        "(default 1)",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=None,
        help="tensor parallel size for the target model; the job uses "
        "dp * tp NPUs (default 1)",
    )
    parser.add_argument(
        "--num-npus",
        type=int,
        default=None,
        help="deprecated alias for --dp",
    )
    parser.add_argument(
        "--npu-id",
        type=int,
        default=None,
        help="NPU index this worker binds; defaults to LOCAL_RANK/RANK and is "
        "set automatically when --dp * --tp workers are launched",
    )
    parser.add_argument(
        "--dist-timeout-minutes",
        type=float,
        default=30.0,
        help="collective timeout used when --tp > 1 (default 30)",
    )
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--quantize", choices=["w8a8", "w4a4", "mixed"], default=None)
    args = parser.parse_args(argv)

    dp_size, tp_size = _parallel_sizes(args)
    args.dp, args.tp = dp_size, tp_size
    world_size = dp_size * tp_size

    if world_size > 1 and args.npu_id is None and not _inside_job():
        timestamp = args.timestamp or datetime.now().strftime("%y%m%d%H%M%S")
        return _launch_workers(args, dp_size, tp_size, timestamp)
    return _run_worker(args, world_size, dp_size, tp_size)


if __name__ == "__main__":
    raise SystemExit(main())
