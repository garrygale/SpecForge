#!/usr/bin/env python3
# coding=utf-8
"""Re-generate assistant responses for a Nemotron JSONL subset using a local vLLM service.

Sends each sample's user prompt(s) to a vLLM server (OpenAI-compatible API) and
replaces the original assistant responses with the model's output.  All metadata
fields (``idx``, ``id``, ``source``, ``source_id``) are preserved.

Multi-turn conversations are handled sequentially: each user -> assistant pair is
regenerated in turn, with the regenerated assistant response fed into the context
for the next turn.

When the model returns ``reasoning_content`` (e.g. with ``--enable-reasoning``
and a vLLM reasoning parser), it is stored on the regenerated assistant message
so the thinking content is preserved for training.

Use ``--feed-reasoning`` to also feed the regenerated ``reasoning_content`` back
into the request history for subsequent turns (off by default).

The vLLM server must be started with a reasoning parser (e.g.
``--reasoning-parser qwen3``) for the API to return thinking text in
``message.reasoning``. Without it, vLLM leaves ``reasoning`` empty and keeps
all tokens in ``content``, so no ``reasoning_content`` can be saved. The
script reads ``reasoning`` first and falls back to ``reasoning_content`` for
older vLLM releases.

Responses cut off by ``max_tokens`` (``finish_reason == "length"``) are kept,
not discarded: the assistant message gets a ``"truncated": true`` flag, and if
the cap was hit while the model was still thinking (no answer text emitted),
the partial reasoning becomes the message content so the sample still counts
as a success.

Use ``--retry-over-tokens N`` to raise the cap later without redoing
everything: a sample is regenerated only when its existing response looks at
risk -- any assistant turn whose saved tokens (``reasoning_content`` +
``content``) exceed N, or a ``"truncated"`` flag, or no previous response
(old errors included). Everything else is copied from the previous output
file (``--prev-output``, defaults to the output path) into the new output.
Counts come from the server's ``POST /tokenize`` endpoint, so the served
model's own tokenizer is used. vLLM's ``max_tokens`` caps generated tokens
only -- prompt and chat-template tokens are charged to the context window
instead -- so N compares against the completion budget alone, minus slack
for the handful of <think>/</think>/EOS tokens the text count misses (e.g.
8000 for a previous 8192 cap). Default -1 regenerates every sample.

Usage::

    python scripts/regen_responses.py
    python scripts/regen_responses.py --input ./subsets/CodeInstruct.jsonl --temperature 0.9 --top-k 50 --top-p 0.95 --enable-reasoning --reasoning-effort high --max-reasoning-tokens 2048 --num-workers 64
    python scripts/regen_responses.py --enable-reasoning --feed-reasoning
    python scripts/regen_responses.py --max-tokens 16384 --retry-over-tokens 8000

Edit the defaults in the ``Configuration`` block at the top of this file,
or override them via CLI arguments.
"""

# ---- Configuration ---------------------------------------------------------
# Override any of these via CLI arguments (see --help).

DATASET_PATH = "./nemotron_subsets/CodeInstruct.jsonl"
VLLM_BASE_URL = "http://localhost:8000/v1"
MODEL_NAME = "default"          # model name as registered in vLLM
TEMPERATURE = 0.7
TOP_K = 0                       # top-k sampling; 0 disables it (vLLM default)
TOP_P = 1.0                     # top-p / nucleus sampling; 1.0 disables it
ENABLE_REASONING = False        # toggle thinking-model controls on/off
REASONING_EFFORT = "medium"     # none|minimal|low|medium|high|xhigh|max
MAX_REASONING_TOKENS = None     # None = no cap; int = thinking_token_budget
FEED_REASONING = False          # include regenerated reasoning_content in later-turn requests
MAX_TOKENS = 4096
NUM_WORKERS = 32
OUTPUT_DIR = "./regenerated"
RETRY_OVER_TOKENS = -1          # >0: only regenerate samples whose saved response exceeds this many tokens
PREV_OUTPUT = None              # previous regenerated JSONL to read existing responses from
# ----------------------------------------------------------------------------

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import httpx
from openai import OpenAI
from tqdm import tqdm


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Re-generate Nemotron dataset responses using a local vLLM service"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=DATASET_PATH,
        help=f"Path to input JSONL subset (default: {DATASET_PATH})",
    )
    parser.add_argument(
        "--vllm-url",
        type=str,
        default=VLLM_BASE_URL,
        help=f"vLLM OpenAI-compatible base URL (default: {VLLM_BASE_URL})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_NAME,
        help=f"Model name (default: {MODEL_NAME})",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=TEMPERATURE,
        help=f"Sampling temperature (default: {TEMPERATURE})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help=f"Top-k sampling; 0 disables it (default: {TOP_K})",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=TOP_P,
        help=f"Top-p (nucleus) sampling, 0.0-1.0 (default: {TOP_P})",
    )
    parser.add_argument(
        "--enable-reasoning",
        dest="enable_reasoning",
        action=argparse.BooleanOptionalAction,
        default=ENABLE_REASONING,
        help=(
            "Send reasoning controls (reasoning_effort and thinking_token_budget) "
            "with each request. Use --no-enable-reasoning for instruct models "
            f"(default: {ENABLE_REASONING})"
        ),
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        choices=["none", "minimal", "low", "medium", "high", "xhigh", "max"],
        default=REASONING_EFFORT,
        help=(
            "Reasoning effort level, sent only when --enable-reasoning is set "
            f"(default: {REASONING_EFFORT})"
        ),
    )
    parser.add_argument(
        "--max-reasoning-tokens",
        type=int,
        default=MAX_REASONING_TOKENS,
        help=(
            "Max reasoning tokens (thinking_token_budget); None sends no limit. "
            "Only sent when --enable-reasoning is set. Requires the vLLM server "
            "to be started with --reasoning-parser / --reasoning-config "
            f"(default: {MAX_REASONING_TOKENS})"
        ),
    )
    parser.add_argument(
        "--feed-reasoning",
        dest="feed_reasoning",
        action=argparse.BooleanOptionalAction,
        default=FEED_REASONING,
        help=(
            "Include regenerated reasoning_content in the request history for "
            "subsequent turns of multi-turn conversations. Use "
            "--no-feed-reasoning to keep sending only role/content "
            f"(default: {FEED_REASONING})"
        ),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=MAX_TOKENS,
        help=f"Max generation tokens (default: {MAX_TOKENS})",
    )
    parser.add_argument(
        "--retry-over-tokens",
        type=int,
        default=RETRY_OVER_TOKENS,
        help=(
            "Only regenerate samples whose existing response is at risk of "
            "truncation: any assistant turn (reasoning_content + content) "
            "over this many tokens, a truncated flag, or no previous "
            "response. Other samples are copied from --prev-output into the "
            "new output. vLLM max_tokens caps generated tokens only (the "
            "chat template does not count), so leave slack for special "
            f"tokens (default: {RETRY_OVER_TOKENS} = regenerate all)"
        ),
    )
    parser.add_argument(
        "--prev-output",
        type=str,
        default=PREV_OUTPUT,
        help=(
            "Previous regenerated JSONL to read existing responses from when "
            "--retry-over-tokens is set (default: the output file itself)"
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=NUM_WORKERS,
        help=f"Concurrent worker threads (default: {NUM_WORKERS})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip samples already present in output + error files",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Regenerate only the first N samples (for testing)",
    )
    args = parser.parse_args()
    if args.resume and args.retry_over_tokens >= 0:
        parser.error("--resume cannot be combined with --retry-over-tokens")
    return args


# ---------------------------------------------------------------------------
#  Regeneration logic (runs in worker threads -- do NOT write to files here)
# ---------------------------------------------------------------------------

def regenerate_one(
    sample: Dict[str, Any],
    vllm_url: str,
    model: str,
    temperature: float,
    max_tokens: int,
    top_k: int,
    top_p: float,
    enable_reasoning: bool,
    reasoning_effort: str,
    max_reasoning_tokens: Optional[int],
    feed_reasoning: bool,
) -> Dict[str, Any]:
    """Regenerate all assistant responses in a single sample.

    Returns the updated sample dict with a ``status`` key set to ``"success"``
    or ``"error"``.  On error, an ``error`` key with the message is added.
    On success, each regenerated assistant message carries the model's
    ``reasoning_content`` when the API returns it.
    """
    messages = sample.get("conversations")
    if not isinstance(messages, list) or len(messages) == 0:
        sample["status"] = "error"
        sample["error"] = "Sample has no 'conversations' list or list is empty"
        return sample

    if messages[0].get("role") == "assistant":
        sample["status"] = "error"
        sample["error"] = "Conversation starts with an assistant message"
        return sample

    # Create a client per sample to avoid cross-thread issues.
    # Matches the pattern in scripts/regenerate_train_data.py.
    client = OpenAI(base_url=vllm_url, api_key="not-needed")

    regenerated: List[Dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        if role == "system":
            regenerated.append(msg)

        elif role == "user":
            regenerated.append(msg)
            # Build conversation-so-far and send to vLLM
            chat_messages = _to_openai_messages(
                regenerated, include_reasoning=feed_reasoning
            )
            try:
                extra_body = {"top_k": top_k}
                if enable_reasoning:
                    extra_body["reasoning_effort"] = reasoning_effort
                    if max_reasoning_tokens is not None:
                        extra_body["thinking_token_budget"] = max_reasoning_tokens

                resp = client.chat.completions.create(
                    model=model,
                    messages=chat_messages,
                    temperature=temperature,
                    top_p=top_p,
                    extra_body=extra_body,
                    max_tokens=max_tokens,
                    stream=False,
                )
            except Exception as exc:
                sample["status"] = "error"
                sample["error"] = f"vLLM request failed: {exc}"
                return sample

            choice = resp.choices[0]
            new_content = choice.message.content
            reasoning_content = _extract_reasoning_content(choice.message)
            truncated = choice.finish_reason == "length"
            if new_content is None:
                if truncated and reasoning_content:
                    # Cap hit while still thinking: keep the partial reasoning
                    # as the answer instead of dropping the sample.
                    new_content = reasoning_content
                    reasoning_content = None
                else:
                    sample["status"] = "error"
                    sample["error"] = "vLLM returned empty content"
                    return sample

            resp_msg: Dict[str, Any] = {
                "role": "assistant",
                "content": new_content,
            }
            if reasoning_content is not None:
                resp_msg["reasoning_content"] = reasoning_content
            if truncated:
                resp_msg["truncated"] = True
            regenerated.append(resp_msg)

        elif role == "assistant":
            # Skip original assistant messages -- they are replaced by the
            # regenerated ones appended after each user turn above.
            continue

        else:
            sample["status"] = "error"
            sample["error"] = f"Unknown message role: {role!r}"
            return sample

    sample["conversations"] = regenerated
    sample["status"] = "success"
    return sample


def _extract_reasoning_content(message: Any) -> Optional[str]:
    """Return the assistant message's reasoning text from a vLLM response.

    vLLM >= 0.26 exposes it as ``message.reasoning``; older releases used
    ``message.reasoning_content``. Some OpenAI SDK versions only surface
    non-standard fields via ``model_extra``, so fall back to the raw dump.
    """
    for key in ("reasoning", "reasoning_content"):
        value = getattr(message, key, None)
        if value is not None:
            return value
    extras = getattr(message, "model_extra", None) or {}
    for key in ("reasoning", "reasoning_content"):
        value = extras.get(key)
        if value is not None:
            return value
    try:
        dumped = message.model_dump()
    except Exception:
        dumped = {}
    for key in ("reasoning", "reasoning_content"):
        value = dumped.get(key)
        if value is not None:
            return value
    return None


def _to_openai_messages(
    conversations: List[Dict[str, Any]],
    include_reasoning: bool = False,
) -> List[Dict[str, str]]:
    """Convert internal conversation format to OpenAI chat messages.

    By default only keeps ``role`` and ``content`` keys -- extra fields like
    ``reasoning_content`` that some Nemotron samples may carry are stripped.
    With ``include_reasoning=True``, a non-None ``reasoning_content`` on an
    assistant message is preserved so the model sees its own previous thinking.
    """
    openai_messages = []
    for msg in conversations:
        if msg.get("role") not in ("system", "user", "assistant"):
            continue
        openai_msg: Dict[str, str] = {
            "role": msg["role"],
            "content": msg["content"],
        }
        if include_reasoning:
            reasoning_content = msg.get("reasoning_content")
            if reasoning_content is not None:
                openai_msg["reasoning_content"] = reasoning_content
        openai_messages.append(openai_msg)
    return openai_messages


def _sample_key(sample: Dict[str, Any]) -> Any:
    return sample.get("idx", sample.get("id"))


def _load_prev_records(path: str) -> Dict[Any, Dict[str, Any]]:
    records: Dict[Any, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = _sample_key(rec)
            if key is not None and key not in records:
                records[key] = rec
    return records


def _server_root(vllm_url: str) -> str:
    root = vllm_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    return root


def _tokenize_count(server_root: str, model: str, text: str) -> Optional[int]:
    """Token count of *text* from the server's /tokenize endpoint, or None."""
    try:
        resp = httpx.post(
            f"{server_root}/tokenize",
            json={"model": model, "prompt": text, "add_special_tokens": False},
            timeout=60.0,
        )
        resp.raise_for_status()
        return int(resp.json()["count"])
    except Exception:
        return None


def _count_record_tokens(
    server_root: str, model: str, rec: Dict[str, Any]
) -> Optional[int]:
    """Max per-turn token count of a regenerated record; None on failure.

    Each assistant turn got its own max_tokens budget, so only the longest
    turn decides whether the sample could have been cut short.
    """
    max_turn = 0
    for msg in rec.get("conversations", []):
        if msg.get("role") != "assistant":
            continue
        turn = 0
        for key in ("reasoning_content", "content"):
            text = msg.get(key)
            if text:
                count = _tokenize_count(server_root, model, text)
                if count is None:
                    return None
                turn += count
        max_turn = max(max_turn, turn)
    return max_turn


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    os.makedirs(args.output_dir, exist_ok=True)

    stem = os.path.splitext(os.path.basename(args.input))[0]
    output_path = os.path.join(args.output_dir, f"{stem}_regen.jsonl")
    error_path = os.path.join(args.output_dir, f"{stem}_regen_error.jsonl")

    # --- Load all samples into memory (JSONL lines are small, this is fine) ---
    print(f"Loading samples from {args.input} ...")
    samples: List[Dict[str, Any]] = []
    with open(args.input, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    total = len(samples)
    print(f"  {total} samples loaded")

    if args.num_samples is not None and args.num_samples < total:
        samples = samples[: args.num_samples]
        total = len(samples)
        print(f"  Limited to first {total} samples (--num-samples)")

    # --- Resume: skip already-processed samples ---
    skip_lines = 0
    if args.resume:
        skip_lines += _count_lines(output_path)
        skip_lines += _count_lines(error_path)
        if skip_lines > 0:
            print(f"Resume: skipping {skip_lines} already-processed samples")
            if skip_lines >= total:
                print("All samples already processed. Nothing to do.")
                return
            samples = samples[skip_lines:]
            total = len(samples)

    # --- Test connectivity ---
    print(f"Testing connection to vLLM at {args.vllm_url} ...")
    test_client = OpenAI(base_url=args.vllm_url, api_key="not-needed")
    try:
        test_extra_body = {"top_k": args.top_k}
        if args.enable_reasoning:
            test_extra_body["reasoning_effort"] = args.reasoning_effort
            if args.max_reasoning_tokens is not None:
                test_extra_body["thinking_token_budget"] = args.max_reasoning_tokens

        test_resp = test_client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=1,
            temperature=0.0,
            top_p=args.top_p,
            extra_body=test_extra_body,
        )
        if test_resp.choices[0].message.content is not None:
            print("  Connection OK")
        else:
            print("  WARNING: test request returned empty content -- continuing anyway")
    except Exception as e:
        raise ConnectionError(
            f"Failed to connect to vLLM at {args.vllm_url}: {e}"
        ) from e

    # --- Retry selection: keep samples whose saved response is safely short ---
    kept_records: List[Dict[str, Any]] = []
    if args.retry_over_tokens >= 0:
        prev_path = args.prev_output or output_path
        if not os.path.exists(prev_path):
            raise FileNotFoundError(
                f"--retry-over-tokens needs the previous output at {prev_path}; "
                "pass --prev-output if it lives elsewhere"
            )
        server_root = _server_root(args.vllm_url)
        if _tokenize_count(server_root, args.model, "hello") is None:
            raise ConnectionError(f"/tokenize endpoint not usable at {server_root}")
        prev_by_key = _load_prev_records(prev_path)
        print(f"Loaded {len(prev_by_key)} previous records from {prev_path}")

        selected: List[Dict[str, Any]] = []
        pending: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        n_missing = n_trunc = 0
        for sample in samples:
            rec = prev_by_key.get(_sample_key(sample))
            if rec is None or rec.get("status") != "success":
                n_missing += 1
                selected.append(sample)
            elif any(
                msg.get("truncated") for msg in rec.get("conversations", [])
            ):
                n_trunc += 1
                selected.append(sample)
            else:
                pending.append((sample, rec))

        n_over = n_count_fail = 0
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {}
            for sample, rec in pending:
                fut = executor.submit(
                    _count_record_tokens, server_root, args.model, rec
                )
                futures[fut] = (sample, rec)
            for f in tqdm(
                as_completed(futures), total=len(futures), desc="Counting tokens"
            ):
                sample, rec = futures[f]
                max_turn = f.result()
                if max_turn is None:
                    n_count_fail += 1
                    selected.append(sample)
                elif max_turn > args.retry_over_tokens:
                    n_over += 1
                    selected.append(sample)
                else:
                    kept_records.append(rec)

        samples = selected
        total = len(samples)
        print(
            f"Retry > {args.retry_over_tokens} tokens: regenerating {total} "
            f"(over threshold: {n_over}, truncated flag: {n_trunc}, "
            f"no previous response: {n_missing}, counting failed: {n_count_fail}); "
            f"keeping {len(kept_records)}"
        )

    # --- Print configuration ---
    print("-" * 50)
    print(f"Configuration:")
    print(f"  Input:        {args.input}")
    print(f"  vLLM URL:     {args.vllm_url}")
    print(f"  Model:        {args.model}")
    print(f"  Temperature:  {args.temperature}")
    print(f"  Top-k:        {args.top_k}")
    print(f"  Top-p:        {args.top_p}")
    print(f"  Reasoning:    {'on' if args.enable_reasoning else 'off'}")
    print(f"  Effort:       {args.reasoning_effort}")
    print(f"  Think cap:    {args.max_reasoning_tokens}")
    print(f"  Feed back:    {'on' if args.feed_reasoning else 'off'}")
    print(f"  Max tokens:   {args.max_tokens}")
    print(f"  Workers:      {args.num_workers}")
    print(f"  Output:       {output_path}")
    print(f"  Errors:       {error_path}")
    print(f"  Resume:       {args.resume}")
    print(f"  Samples:      {total}")
    print("-" * 50)

    # --- Regenerate ---
    write_lock = Lock()
    success_count = len(kept_records)
    error_count = 0
    reasoning_saved = 0
    truncated_count = 0

    start_time = time.time()

    with (
        open(output_path, "a" if args.resume and skip_lines > 0 else "w", encoding="utf-8") as output_fh,
        open(error_path, "a" if args.resume and skip_lines > 0 else "w", encoding="utf-8") as error_fh,
        ThreadPoolExecutor(max_workers=args.num_workers) as executor,
    ):
        # Carried-over samples are written first so the output is complete
        # even if the run dies mid-way through regeneration.
        for rec in kept_records:
            output_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if any(
                msg.get("role") == "assistant"
                and msg.get("reasoning_content") is not None
                for msg in rec.get("conversations", [])
            ):
                reasoning_saved += 1
        if kept_records:
            output_fh.flush()

        futures = {
            executor.submit(
                regenerate_one,
                sample,
                args.vllm_url,
                args.model,
                args.temperature,
                args.max_tokens,
                args.top_k,
                args.top_p,
                args.enable_reasoning,
                args.reasoning_effort,
                args.max_reasoning_tokens,
                args.feed_reasoning,
            ): i
            for i, sample in enumerate(samples)
        }

        for f in tqdm(as_completed(futures), total=len(futures), desc="Regenerating"):
            result = f.result()
            with write_lock:
                if result.get("status") == "success":
                    output_fh.write(json.dumps(result, ensure_ascii=False) + "\n")
                    output_fh.flush()
                    if any(
                        msg.get("role") == "assistant"
                        and msg.get("reasoning_content") is not None
                        for msg in result.get("conversations", [])
                    ):
                        reasoning_saved += 1
                    if any(
                        msg.get("truncated")
                        for msg in result.get("conversations", [])
                    ):
                        truncated_count += 1
                    success_count += 1
                else:
                    error_fh.write(json.dumps(result, ensure_ascii=False) + "\n")
                    error_fh.flush()
                    error_count += 1

    elapsed = time.time() - start_time
    print("-" * 50)
    print(f"Done in {elapsed:.1f}s")
    if kept_records:
        print(
            f"  Success: {success_count} "
            f"(regenerated {success_count - len(kept_records)}, "
            f"kept {len(kept_records)})"
        )
    else:
        print(f"  Success: {success_count}")
    print(f"  Errors:  {error_count}")
    print(f"  Truncated (kept): {truncated_count}")
    if args.enable_reasoning:
        print(f"  With reasoning: {reasoning_saved}")
    print(f"  Output:  {output_path}")
    if error_count > 0:
        print(f"  Errors:  {error_path}")
    if args.enable_reasoning and success_count > 0 and reasoning_saved == 0:
        print(
            "WARNING: reasoning was enabled but no reasoning_content was saved. "
            "Make sure the vLLM server is started with --reasoning-parser "
            "(e.g. --reasoning-parser qwen3); otherwise vLLM keeps thinking "
            "tokens inside content and message.reasoning stays empty."
        )


def _count_lines(path: str) -> int:
    """Return the number of lines in *path*, or 0 if the file doesn't exist."""
    if not os.path.exists(path):
        return 0
    count = 0
    with open(path, "r", encoding="utf-8") as fh:
        for _ in fh:
            count += 1
    return count


if __name__ == "__main__":
    main()
