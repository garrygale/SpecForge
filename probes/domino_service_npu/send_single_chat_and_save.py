#!/usr/bin/env python3
"""Send one prompt and save both text and timing for triage.

Use this during a degraded acceptance window to keep a concrete output sample
next to the server's ``[DOMINO_DEBUG]`` logs.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

import requests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-ip", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=4144)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Keep enable_thinking=True. Some Qwen responses then return no "
        "final content and only reasoning_content when max_tokens is small.",
    )
    parser.add_argument("--output", default="results/domino_acceptance/degraded_sample.json")
    args = parser.parse_args()

    payload = {
        "model": args.served_model_name,
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": args.enable_thinking},
    }
    url = f"http://{args.server_ip}:{args.server_port}/v1/chat/completions"
    start = time.monotonic()
    resp = requests.post(url, json=payload, timeout=1800)
    elapsed = time.monotonic() - start
    resp.raise_for_status()
    data = resp.json()
    message = data["choices"][0]["message"]
    content = message.get("content") or message.get("reasoning_content") or ""
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    usage = data.get("usage", {})

    record = {
        "elapsed_s": round(elapsed, 2),
        "completion_tokens": usage.get("completion_tokens"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "generated_text": content,
        "thinking_enabled": args.enable_thinking,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    import os

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    print(f"Saved degraded output sample to {args.output}")
    print(
        "First 500 chars:",
        (content[:500] if content else "<no content, only reasoning/thinking>").replace(
            "\n", "\\n"
        ),
    )


if __name__ == "__main__":
    main()
