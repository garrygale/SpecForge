#!/usr/bin/env python3
"""Watch Domino acceptance over a single long generation.

This probe starts one chat-completion request and polls the vLLM Prometheus
spec-decode counters while that request is running.  It prints per-interval
and cumulative acceptance deltas so you can see whether acceptance collapses
after a certain generation length/context size, or only when many requests are
sharing the server.

Requires only ``requests``; no torch/torch_npu/sglang.

Example:
    python probes/domino_service_npu/probe_acceptance_over_generation.py \
        --server-port 4144 \
        --served-model-name qwen3.6-35b \
        --prompt "Solve this step by step: ..." \
        --max-tokens 2048
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from datetime import datetime, timezone

import requests


def fetch_spec_decode_metrics(base_url: str) -> dict | None:
    resp = requests.get(f"{base_url}/metrics", timeout=30)
    resp.raise_for_status()

    num_drafts = 0
    num_draft_tokens = 0
    num_accepted_tokens = 0
    accepted_per_pos: dict[int, int] = {}
    found = False

    for line in resp.text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or not line.startswith("vllm:spec_decode"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        metric_name = parts[0].split("{")[0]
        if not metric_name.endswith("_total"):
            continue
        try:
            value = int(float(parts[-1]))
        except ValueError:
            continue
        found = True
        if metric_name == "vllm:spec_decode_num_drafts_total":
            num_drafts += value
        elif metric_name == "vllm:spec_decode_num_draft_tokens_total":
            num_draft_tokens += value
        elif metric_name == "vllm:spec_decode_num_accepted_tokens_per_pos_total":
            label = 'position="'
            if label in line:
                start = line.index(label) + len(label)
                end = line.index('"', start)
                try:
                    pos = int(line[start:end])
                except ValueError:
                    continue
                accepted_per_pos[pos] = accepted_per_pos.get(pos, 0) + value
        elif metric_name == "vllm:spec_decode_num_accepted_tokens_total":
            num_accepted_tokens += value

    if not found:
        return None
    return {
        "num_drafts": num_drafts,
        "num_draft_tokens": num_draft_tokens,
        "num_accepted_tokens": num_accepted_tokens,
        "accepted_per_pos": accepted_per_pos,
    }


def metric_delta(before: dict, after: dict) -> dict:
    accepted_per_pos: dict[int, int] = {}
    for pos, val in after["accepted_per_pos"].items():
        accepted_per_pos[pos] = val - before["accepted_per_pos"].get(pos, 0)
    return {
        "num_drafts": after["num_drafts"] - before["num_drafts"],
        "num_draft_tokens": (
            after["num_draft_tokens"] - before["num_draft_tokens"]
        ),
        "num_accepted_tokens": (
            after["num_accepted_tokens"] - before["num_accepted_tokens"]
        ),
        "accepted_per_pos": accepted_per_pos,
    }


def per_pos_rates(accepted_per_pos: dict[int, int], total_drafts: int) -> list[float]:
    max_pos = max(accepted_per_pos, default=-1)
    return [
        accepted_per_pos.get(pos, 0) / total_drafts if total_drafts > 0 else 0.0
        for pos in range(max_pos + 1)
    ]


def send_chat(
    base_url: str,
    model_name: str,
    prompt: str,
    max_tokens: int,
    result: list,
) -> None:
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    try:
        resp = requests.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            timeout=3600,
        )
        resp.raise_for_status()
        result.append(("ok", resp.json()))
    except Exception as exc:  # noqa: BLE001
        result.append(("error", str(exc)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-ip", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=4144)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--poll-interval", type=float, default=2.0)
    args = parser.parse_args()

    base_url = f"http://{args.server_ip}:{args.server_port}"
    print(
        f"Server: {base_url}  model={args.served_model_name}  "
        f"max_tokens={args.max_tokens}"
    )
    print("Waiting for metrics before starting...")
    baseline = fetch_spec_decode_metrics(base_url)
    if baseline is None:
        print("(spec-decode counters not exported yet; using first poll as baseline)")

    result: list = []
    thread = threading.Thread(
        target=send_chat,
        args=(base_url, args.served_model_name, args.prompt, args.max_tokens, result),
        daemon=True,
    )
    start = time.monotonic()
    thread.start()

    totals = {
        "num_drafts": 0,
        "num_draft_tokens": 0,
        "num_accepted_tokens": 0,
        "accepted_per_pos": {},
    }
    samples: list[dict] = []

    while thread.is_alive():
        time.sleep(args.poll_interval)
        after = fetch_spec_decode_metrics(base_url)
        if after is None:
            continue
        if baseline is None:
            baseline = after
            continue
        delta = metric_delta(baseline, after)
        baseline = after
        for key in ("num_drafts", "num_draft_tokens", "num_accepted_tokens"):
            totals[key] += delta[key]
        for pos, val in delta["accepted_per_pos"].items():
            totals["accepted_per_pos"][pos] = (
                totals["accepted_per_pos"].get(pos, 0) + val
            )
        elapsed = time.monotonic() - start
        rates = per_pos_rates(delta["accepted_per_pos"], delta["num_drafts"])
        rates_str = ", ".join(f"{r:.3f}" for r in rates) or "-"
        mean_len = (
            1 + delta["num_accepted_tokens"] / delta["num_drafts"]
            if delta["num_drafts"] > 0
            else float("nan")
        )
        total_len = (
            1 + totals["num_accepted_tokens"] / totals["num_drafts"]
            if totals["num_drafts"] > 0
            else float("nan")
        )
        print(
            f"[t={elapsed:7.1f}s] drafts={delta['num_drafts']:6d} "
            f"accepted={delta['num_accepted_tokens']:6d} "
            f"len={mean_len:5.2f} cum_len={total_len:5.2f} "
            f"per_pos=[{rates_str}]",
            flush=True,
        )
        samples.append(
            {
                "elapsed_s": round(elapsed, 2),
                "delta": delta,
                "cumulative": {
                    k: (dict(v) if isinstance(v, dict) else v)
                    for k, v in totals.items()
                },
            }
        )

    thread.join()
    time.sleep(0.5)
    after = fetch_spec_decode_metrics(base_url)
    if after is not None and baseline is not None:
        delta = metric_delta(baseline, after)
        for key in ("num_drafts", "num_draft_tokens", "num_accepted_tokens"):
            totals[key] += delta[key]
        for pos, val in delta["accepted_per_pos"].items():
            totals["accepted_per_pos"][pos] = (
                totals["accepted_per_pos"].get(pos, 0) + val
            )
        baseline = after

    elapsed = time.monotonic() - start
    if result:
        status, payload = result[0]
        if status == "ok":
            usage = payload.get("usage", {})
            print(
                f"\nRequest finished after {elapsed:.1f}s, "
                f"completion_tokens={usage.get('completion_tokens')}, "
                f"prompt_tokens={usage.get('prompt_tokens')}"
            )
        else:
            print(f"\nRequest failed after {elapsed:.1f}s: {payload}")
    else:
        print(f"\nRequest did not finish in {elapsed:.1f}s")

    total_len = (
        1 + totals["num_accepted_tokens"] / totals["num_drafts"]
        if totals["num_drafts"] > 0
        else float("nan")
    )
    rates = per_pos_rates(totals["accepted_per_pos"], totals["num_drafts"])
    rates_str = ", ".join(f"{r:.3f}" for r in rates) or "-"
    print(f"Cumulative drafts={totals['num_drafts']} "
          f"accepted={totals['num_accepted_tokens']} len={total_len:.3f} "
          f"per_pos=[{rates_str}]")

    summary = {
        "server": base_url,
        "served_model_name": args.served_model_name,
        "max_tokens": args.max_tokens,
        "totals": {
            k: (dict(v) if isinstance(v, dict) else v)
            for k, v in totals.items()
        },
        "mean_acceptance_length": total_len,
        "per_position_acceptance_rates": rates,
        "num_samples": len(samples),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    stamp = datetime.now().strftime("%y%m%d_%H%M%S")
    path = f"results/domino_acceptance/acceptance_trace_{stamp}.json"
    import os

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(f"Trace written to {path}")


if __name__ == "__main__":
    main()
