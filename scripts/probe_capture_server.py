# coding=utf-8
"""Probe one spec-capture request against a running SGLang capture server.

An online run publishes nothing when every capture call fails, which shows up
as an empty per-rank inbox long before any error surfaces. This probe issues
exactly one capture with the same schema the producer would build, and prints
either the artifacts the server returned or the client-side reason it refused
them.

Example:

    python scripts/probe_capture_server.py \
        --server-url http://127.0.0.1:30000 \
        --target-model-path Qwen/Qwen3.6-35B-A3B \
        --draft-model-config configs/qwen3.6-35b-a3b-domino-dflare-verifiedBase.json

Add ``--l1`` to probe the Domino CE+L1 request, which also asks the server for
the ``last_hidden`` artifact.
"""

from __future__ import annotations

import argparse
import sys


class _NullStore:
    """Feature-store subset: the server owns the write, the probe only reads."""

    store_id = "capture-probe"

    def __init__(self) -> None:
        self.attempts = {}
        self.adopted = []

    def adopt(self, ref) -> None:
        self.adopted.append(ref.sample_id)

    def abort(self, sample_id: str, *, reason: str) -> None:
        print(f"  (cleanup) would abort {sample_id}: {reason}")

    def discard_external_attempts(self, *, reason: str = "probe") -> None:
        pass

    def track_external_attempt(self, sample_id, *, generation, feature_names):
        self.attempts[sample_id] = (generation, tuple(feature_names))


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", required=True)
    parser.add_argument(
        "--strategy",
        default="domino",
        help="training.strategy of the run being prepared (default: domino)",
    )
    parser.add_argument("--target-model-path", required=True)
    parser.add_argument("--draft-model-config", required=True)
    parser.add_argument(
        "--prompt-tokens",
        type=int,
        default=16,
        help="length of the synthetic probe prompt (default: 16)",
    )
    parser.add_argument(
        "--l1",
        action="store_true",
        help="enable the Domino L1/TV objective, which needs the teacher state",
    )
    parser.add_argument("--timeout", type=float, default=300.0)
    return parser.parse_args(argv)


def _payload(args: argparse.Namespace) -> dict:
    training = {
        "strategy": args.strategy,
        "max_steps": 1,
        "batch_size": 1,
        "accumulation_steps": 1,
    }
    if args.strategy == "domino" and args.l1:
        training["domino_l1_loss_alpha"] = 0.9
    return {
        "model": {
            "target_model_path": args.target_model_path,
            "draft_model_config": args.draft_model_config,
            "target_backend": "sglang",
        },
        "data": {"train_data_path": "probe.jsonl"},
        "training": training,
        "deployment": {
            "mode": "disaggregated",
            "disaggregated": {
                "control_dir": "probe-control",
                "backend": "mooncake",
                "server_urls": [args.server_url],
            },
        },
    }


def main(argv=None) -> int:
    args = _parse_args(argv)

    from specforge.algorithms.builtin import builtin_algorithm_registry
    from specforge.config import Config
    from specforge.inference.adapters.server_capture import (
        SGLangServerCaptureAdapter,
        ServerCaptureFailure,
        ServerCaptureSchema,
    )
    from specforge.inference.capture import CaptureConfig
    from specforge.runtime.contracts import PromptTask
    from specforge.training.capture_contract import (
        resolve_server_capture_contract,
        resolve_streaming_capture,
    )

    registry = builtin_algorithm_registry()
    if args.strategy not in registry.names:
        print(f"unknown strategy {args.strategy!r}", file=sys.stderr)
        return 2
    algorithm = registry.resolve(args.strategy)
    config = Config.model_validate(_payload(args))

    contract = resolve_server_capture_contract(config, algorithm=algorithm)
    capture_request = resolve_streaming_capture(config, algorithm=algorithm)
    layout = capture_request.layout
    schema = ServerCaptureSchema(
        aux_feature=layout.aux_feature,
        last_hidden_feature=layout.last_hidden_feature,
        passthrough=layout.passthrough,
        attention_mask_feature=layout.attention_mask_feature,
    )
    capture = CaptureConfig.from_strategy(
        required_features=capture_request.required_features,
        aux_hidden_state_layer_ids=contract.aux_layer_ids,
        target_repr=algorithm.providers.server_streaming_for(
            config.model.input_modality
        ).target_representation,
        target_hidden_size=contract.target_hidden_size,
        target_vocab_size=contract.target_vocab_size,
        draft_vocab_size=contract.draft_vocab_size,
    )

    print(f"strategy:             {args.strategy}")
    print(f"capture method:       {contract.method}")
    print(f"aux capture layers:   {list(contract.aux_layer_ids)}")
    print(f"target hidden size:   {contract.target_hidden_size}")
    print(f"required features:    {sorted(capture_request.required_features)}")
    print(f"aux artifact:         {layout.aux_feature}")
    print(f"last_hidden artifact: {layout.last_hidden_feature}")
    print(f"server:               {args.server_url}")

    store = _NullStore()
    adapter = SGLangServerCaptureAdapter(
        args.server_url,
        store,
        run_id="capture-probe",
        algorithm=args.strategy,
        schema=schema,
        timeout_s=args.timeout,
    )
    length = max(2, int(args.prompt_tokens))
    task = PromptTask(
        run_id="capture-probe",
        task_id="probe-0",
        source_id="capture-probe",
        payload={
            "input_ids": list(range(1, length + 1)),
            "loss_mask": [1] * length,
        },
        max_length=length,
        metadata={"num_tokens": length},
    )
    try:
        results = adapter.produce_refs([task], capture=capture)
    except Exception as exc:  # transport failure: server unreachable / erroring
        print(f"\nPROBE FAILED (transport): {type(exc).__name__}: {exc}")
        return 1

    result = results[0]
    if isinstance(result, ServerCaptureFailure):
        print(f"\nPROBE FAILED (client rejected the capture): {result.reason}")
        print(f"retryable: {result.retryable}")
        _print_hint(result.reason)
        return 1

    print("\nPROBE OK: the server returned a capture this run accepts.")
    for name, spec in sorted(result.feature_specs.items()):
        print(f"  {name}: shape={tuple(spec.shape)} dtype={spec.dtype}")
    print(f"  feature keys under mooncake://{store.store_id}/{result.sample_id}")
    return 0


def _print_hint(reason: str) -> None:
    hints = (
        (
            "no spec_capture result",
            "The server answered /generate without a spec_capture payload: it is "
            "not running patches/sglang/v0.5.18/spec-capture.patch, or it was "
            "started without --enable-spec-capture.",
        ),
        (
            "missing features",
            "The server accepted the request but returned fewer artifacts than "
            "this run needs. If the missing feature is target_last_hidden_states "
            "the capture build predates the last-hidden plumbing: re-apply the "
            "patch and restart the server.",
        ),
        (
            "seq len",
            "The capture did not cover the whole prompt: start the server with "
            "--chunked-prefill-size -1.",
        ),
        (
            "aux-layer ids",
            "Start the server with --spec-capture-aux-layer-ids matching the "
            "draft config's target_layer_ids.",
        ),
    )
    for needle, hint in hints:
        if needle in reason:
            print(f"hint: {hint}")
            return
    print(
        "hint: read the server's scheduler log for the matching spec_capture "
        "error; the client message above is the reason it rejected the result."
    )


if __name__ == "__main__":
    raise SystemExit(main())
