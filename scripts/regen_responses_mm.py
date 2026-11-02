#!/usr/bin/env python3
# coding=utf-8
"""Re-generate assistant responses for image-grounded JSONL datasets (llava-665k
style) using a local vLLM service.

Mirrors ``scripts/regen_responses.py``; the only additions are multimodal:

* records carry ``"images": ["relpath.jpg", ...]`` (empty list = text-only),
  resolved against ``--image-root``;
* the images are attached to the **first user turn** as OpenAI content parts
  (``image_url`` with a base64 data URI) — later turns stay plain text, the
  model has already seen the image;
* image bytes are read from disk only when a request is built, so the input
  file stays small and images are never duplicated into any output.

Everything else keeps the original semantics: multi-turn conversations are
regenerated sequentially with responses fed forward; ``reasoning_content`` is
saved when the server returns it (``--enable-reasoning``); responses cut off
by ``max_tokens`` are kept with a ``"truncated"`` flag; ``--retry-over-tokens``
reads its state from the input file so a previous run's output can be fed back
as ``--input`` to redo only at-risk samples (image paths ride along unchanged,
no re-touching of the arrow source).

Usage::

    python scripts/regen_responses_mm.py --input data/llava665k.jsonl \
        --image-dir data/llava665k_images --vllm-url http://localhost:8000/v1
    python scripts/regen_responses_mm.py --input ./regenerated/llava665k_regen.jsonl \
        --image-dir data/llava665k_images --max-tokens 8192 --retry-over-tokens 4000
"""

# ---- Configuration ---------------------------------------------------------
# Override any of these via CLI arguments (see --help).

INPUT_PATH = "./data/llava665k.jsonl"
IMAGE_ROOT = "./data/llava665k_images"
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
REQUEST_TIMEOUT = 600.0         # per-request timeout in seconds; raise for long generations
OUTPUT_DIR = "./regenerated"
RETRY_OVER_TOKENS = -1          # >0: only regenerate samples whose saved response exceeds this many tokens
# ----------------------------------------------------------------------------

import argparse
import base64
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI
from tqdm import tqdm

_MIME = {
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "webp": "image/webp",
    "gif": "image/gif",
    "bmp": "image/bmp",
}


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Re-generate image-grounded dataset responses using a local vLLM service"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=INPUT_PATH,
        help=f"Path to input JSONL (default: {INPUT_PATH})",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default=IMAGE_ROOT,
        help=(
            "Root directory the records' image paths are relative to "
            f"(default: {IMAGE_ROOT})"
        ),
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
            "Only regenerate samples whose existing responses are at risk "
            "of truncation: any assistant turn (reasoning_content + content) "
            "over this many tokens, a truncated flag, or a user turn "
            "without a response. Other samples are written to the output "
            "unchanged, keeping the responses already in the input file. "
            "vLLM max_tokens caps generated tokens only (the chat template "
            "does not count), so leave slack for special tokens "
            f"(default: {RETRY_OVER_TOKENS} = regenerate all)"
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=NUM_WORKERS,
        help=f"Concurrent worker threads (default: {NUM_WORKERS})",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=REQUEST_TIMEOUT,
        help=(
            "Per-request timeout in seconds for chat completion calls. "
            "The OpenAI SDK default is 600s, too short for very long "
            f"generations (default: {REQUEST_TIMEOUT})"
        ),
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
#  Image helpers (bytes are read from disk only when a request is built)
# ---------------------------------------------------------------------------

def _image_data_uri(rel_path: str, image_root: str) -> str:
    path = rel_path if os.path.isabs(rel_path) else os.path.join(image_root, rel_path)
    with open(path, "rb") as fh:
        data = fh.read()
    ext = os.path.splitext(rel_path)[1].lstrip(".").lower()
    mime = _MIME.get(ext, "image/jpeg")
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def _strip_image_marker(text: str) -> str:
    # Canonical llava-665k texts are already clean; this only guards against
    # records converted from the original json where the first user turn
    # begins with the "<image>" placeholder.
    for marker in ("<image>\n", "<image>"):
        if text.startswith(marker):
            return text[len(marker):]
    return text


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
    timeout: float,
    image_root: str,
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

    images = sample.get("images") or []
    if not isinstance(images, list):
        sample["status"] = "error"
        sample["error"] = "'images' must be a list of paths when present"
        return sample
    try:
        # Resolve the data URIs once per sample; images attach to the first
        # user turn only, matching the llava convention.
        image_uris = [_image_data_uri(rel, image_root) for rel in images] or None
    except OSError as exc:
        sample["status"] = "error"
        sample["error"] = f"Cannot read image: {exc}"
        return sample

    # Create a client per sample to avoid cross-thread issues.
    # Matches the pattern in scripts/regenerate_train_data.py.
    client = OpenAI(base_url=vllm_url, api_key="not-needed", timeout=timeout)

    regenerated: List[Dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        if role == "system":
            regenerated.append(msg)

        elif role == "user":
            regenerated.append(msg)
            # Build conversation-so-far and send to vLLM
            chat_messages = _to_openai_messages(
                regenerated,
                include_reasoning=feed_reasoning,
                image_uris=image_uris,
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
    image_uris: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Convert internal conversation format to OpenAI chat messages.

    Text-only messages become ``{"role", "content": str}`` exactly like the
    language-only regenerator.  When ``image_uris`` is given (image-grounded
    sample), the FIRST user message becomes multimodal content parts: the
    images followed by the text; later user turns stay plain text.
    """
    openai_messages = []
    attached = False
    for msg in conversations:
        if msg.get("role") not in ("system", "user", "assistant"):
            continue
        if image_uris is not None and msg.get("role") == "user" and not attached:
            attached = True
            content: Any = [
                {"type": "image_url", "image_url": {"url": uri}}
                for uri in image_uris
            ]
            content.append(
                {"type": "text", "text": _strip_image_marker(msg["content"])}
            )
            openai_msg: Dict[str, Any] = {"role": "user", "content": content}
        else:
            openai_msg = {"role": msg["role"], "content": msg["content"]}
        if include_reasoning:
            reasoning_content = msg.get("reasoning_content")
            if reasoning_content is not None:
                openai_msg["reasoning_content"] = reasoning_content
        openai_messages.append(openai_msg)
    return openai_messages


def _has_complete_responses(conversations: List[Dict[str, Any]]) -> bool:
    """True when every user turn has an assistant response with text."""
    n_user = 0
    n_resp = 0
    for msg in conversations:
        role = msg.get("role")
        if role == "user":
            n_user += 1
        elif role == "assistant" and (
            msg.get("content") or msg.get("reasoning_content")
        ):
            n_resp += 1
    return n_user > 0 and n_resp >= n_user


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


def _count_sample_tokens(
    server_root: str, model: str, sample: Dict[str, Any]
) -> Optional[int]:
    """Max per-turn token count of a sample's existing responses; None on
    failure.

    Each assistant turn got its own max_tokens budget, so only the longest
    turn decides whether the sample could have been cut short.  Image tokens
    are prompt-side and never part of a completion budget.
    """
    max_turn = 0
    for msg in sample.get("conversations", []):
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
    n_image = sum(1 for s in samples if s.get("images"))
    print(f"  {total} samples loaded ({n_image} image-grounded, {total - n_image} text-only)")

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
    test_client = OpenAI(base_url=args.vllm_url, api_key="not-needed", timeout=args.timeout)
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

    # --- Retry selection: keep samples whose existing responses are safely short ---
    kept_records: List[Dict[str, Any]] = []
    if args.retry_over_tokens >= 0:
        server_root = _server_root(args.vllm_url)
        if _tokenize_count(server_root, args.model, "hello") is None:
            raise ConnectionError(f"/tokenize endpoint not usable at {server_root}")

        selected: List[Dict[str, Any]] = []
        pending: List[Dict[str, Any]] = []
        n_missing = n_trunc = 0
        for sample in samples:
            conversations = sample.get("conversations", [])
            if not _has_complete_responses(conversations):
                n_missing += 1
                selected.append(sample)
            elif any(
                msg.get("truncated")
                for msg in conversations
                if msg.get("role") == "assistant"
            ):
                n_trunc += 1
                selected.append(sample)
            else:
                pending.append(sample)

        n_over = n_count_fail = 0
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {}
            for sample in pending:
                fut = executor.submit(
                    _count_sample_tokens, server_root, args.model, sample
                )
                futures[fut] = sample
            for f in tqdm(
                as_completed(futures), total=len(futures), desc="Counting tokens"
            ):
                sample = futures[f]
                max_turn = f.result()
                if max_turn is None:
                    n_count_fail += 1
                    selected.append(sample)
                elif max_turn > args.retry_over_tokens:
                    n_over += 1
                    selected.append(sample)
                else:
                    kept = dict(sample)
                    kept["status"] = "success"
                    kept_records.append(kept)

        samples = selected
        total = len(samples)
        print(
            f"Retry > {args.retry_over_tokens} tokens: regenerating {total} "
            f"(over threshold: {n_over}, truncated flag: {n_trunc}, "
            f"incomplete responses: {n_missing}, counting failed: {n_count_fail}); "
            f"keeping {len(kept_records)}"
        )

    # --- Print configuration ---
    print("-" * 50)
    print(f"Configuration:")
    print(f"  Input:        {args.input}")
    print(f"  Image dir:    {args.image_dir}")
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
    print(f"  Timeout:      {args.timeout:g}s")
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
        # Kept samples (responses already in the input) are written first so
        # the output is complete even if the run dies mid-way through
        # regeneration.
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
                args.timeout,
                args.image_dir,
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
