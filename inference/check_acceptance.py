#!/usr/bin/env python3
"""Acceptance-length evaluation for DFlash/Domino draft models.

This is the standalone in-process evaluator used to measure draft acceptance
without launching a vLLM service. It supports HumanEval, GSM8K, MATH-500,
MBPP, multi-NPU sharding/merging, and optional NPU W8A8/W4A4/mixed inference
quantization.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from typing import Any, Optional

import torch
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


def _npu_available() -> bool:
    npu = getattr(torch, "npu", None)
    return bool(npu is not None and npu.is_available())


def _resolve_device(npu_id: Optional[int] = None) -> str:
    if npu_id is not None:
        torch.npu.set_device(npu_id)
        return f"npu:{npu_id}"
    if _npu_available():
        return "npu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_target_model(target_path: str, device: str, torch_dtype):
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
    num_npus: Optional[int] = None,
    timestamp: Optional[str] = None,
    quantize: Optional[str] = None,
    benchmark: str = "humaneval",
    gsm8k_path: Optional[str] = None,
    math500_path: Optional[str] = None,
    mbpp_path: Optional[str] = None,
) -> dict[str, Any]:
    device = _resolve_device(npu_id)
    torch_dtype = torch.bfloat16 if device != "cpu" else torch.float32

    draft_model = AutoDraftModel.from_pretrained(
        draft_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    draft_model.to(device)
    draft_model.eval()
    print(f"Draft loaded: block_size={draft_model.block_size}")
    if quantize:
        _apply_quantize(draft_model, quantize)
        print(f"  Applied {quantize} NPU quantization to draft model.")

    target_model = _load_target_model(target_path, device, torch_dtype)
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
        if num_npus is not None and npu_id is not None and idx % num_npus != npu_id:
            continue
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
    npu_suffix = f"_npu{npu_id}" if npu_id is not None else ""
    log_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"{timestamp}_acceptance_lengths{npu_suffix}.jsonl",
    )
    with open(log_file, "w", encoding="utf-8") as f:
        for entry in per_problem:
            f.write(json.dumps(entry) + "\n")

    print(f"Wrote per-NPU results to {log_file}")
    return {
        "per_problem": per_problem,
        "overall_simple": overall_simple,
        "overall_weighted": overall_weighted,
        "valid_count": valid_count,
        "total_problems": len(problems),
        "log_file": log_file,
        "per_position_accuracy": per_pos_avg,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--npu-id", type=int, default=None)
    parser.add_argument("--num-npus", type=int, default=None)
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--quantize", choices=["w8a8", "w4a4", "mixed"], default=None)
    args = parser.parse_args(argv)

    if args.num_npus and args.num_npus > 1 and args.npu_id is None:
        timestamp = datetime.now().strftime("%y%m%d%H%M%S")
        procs = []
        for i in range(args.num_npus):
            cmd = [sys.executable, __file__]
            cmd += ["--draft-path", args.draft_path, "--target-path", args.target_path]
            cmd += ["--humaneval-path", args.humaneval_path]
            cmd += ["--benchmark", args.benchmark, "--npu-id", str(i), "--num-npus", str(args.num_npus), "--timestamp", timestamp]
            if args.gsm8k_path:
                cmd += ["--gsm8k-path", args.gsm8k_path]
            if args.math500_path:
                cmd += ["--math500-path", args.math500_path]
            if args.mbpp_path:
                cmd += ["--mbpp-path", args.mbpp_path]
            if args.quantize:
                cmd += ["--quantize", args.quantize]
            cmd += ["--max-new-tokens", str(args.max_new_tokens)]
            cmd += ["--temperature", str(args.temperature)]
            if args.universal_prefix:
                cmd += ["--universal-prefix", args.universal_prefix]
            if args.use_chat_template:
                cmd.append("--use-chat-template")
            else:
                cmd.append("--no-use-chat-template")
            if args.include_incomplete_blocks:
                cmd.append("--include-incomplete-blocks")
            procs.append(subprocess.Popen(cmd))
        for proc in procs:
            proc.wait()
        return 0

    run_acceptance_check(
        draft_path=args.draft_path,
        target_path=args.target_path,
        humaneval_path=args.humaneval_path,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        use_chat_template=args.use_chat_template,
        universal_prefix=args.universal_prefix,
        include_incomplete_blocks=args.include_incomplete_blocks,
        npu_id=args.npu_id,
        num_npus=args.num_npus,
        timestamp=args.timestamp,
        quantize=args.quantize,
        benchmark=args.benchmark,
        gsm8k_path=args.gsm8k_path,
        math500_path=args.math500_path,
        mbpp_path=args.mbpp_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
