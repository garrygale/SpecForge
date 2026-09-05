#!/usr/bin/env python3
"""Measure Domino acceptance while slowing request replacement.

This is the "fixed concurrency, slower finish/reuse rate" probe. It keeps
``NUM_WORKERS`` clients in flight and inserts an optional delay after each
request completes before the next prompt is sent. If the acceptance decay
disappears/weakens as the replacement delay grows, the collapse is driven by
the finished-request/block-reuse path; if it still decays, it is a
steady-state batch-size bug.

Example (dp=1, 32 clients, full eager):
    python probes/domino_service_npu/check_acceptance_replacement_rate.py \
        --server-port 4144 --served-model-name qwen3.6-35b \
        --dataset humaneval --dataset-path /path/human-eval-v2-20210705.jsonl \
        --num-workers 32 --replacement-delay-ms 100 \
        --monitor-interval 10
"""

from __future__ import annotations

import argparse
import json
import queue
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import requests


def load_prompts(path: str, dataset: str) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    if dataset == "mbpp":
        with open(path, encoding="utf-8") as f:
            items = json.load(f)
        for idx, item in enumerate(items):
            text = item.get("prompt") or item.get("text") or ""
            task_id = item.get("task_id", idx)
            rows.append((f"mbpp_{task_id}", text))
        return rows
    with open(path, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if dataset == "humaneval":
                rows.append((item["task_id"], item["prompt"]))
            elif dataset == "gsm8k":
                rows.append((f"gsm8k_{idx}", item["question"]))
            elif dataset == "math500":
                rows.append((str(item.get("unique_id") or idx), item["problem"]))
            else:
                raise ValueError(f"unknown dataset {dataset}")
    return rows


def fetch_metrics(base_url: str) -> dict | None:
    resp = requests.get(f"{base_url}/metrics", timeout=30)
    resp.raise_for_status()
    num_drafts = 0
    num_draft_tokens = 0
    num_accepted = 0
    accepted_per_pos: dict[int, int] = {}
    found = False
    for line in resp.text.splitlines():
        line = line.strip()
        if not line or not line.startswith("vllm:spec_decode"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        name = parts[0].split("{")[0]
        if not name.endswith("_total"):
            continue
        try:
            value = int(float(parts[-1]))
        except ValueError:
            continue
        found = True
        if name == "vllm:spec_decode_num_drafts_total":
            num_drafts += value
        elif name == "vllm:spec_decode_num_draft_tokens_total":
            num_draft_tokens += value
        elif name == "vllm:spec_decode_num_accepted_tokens_per_pos_total":
            label = 'position="'
            if label in line:
                start = line.index(label) + len(label)
                end = line.index('"', start)
                accepted_per_pos[int(line[start:end])] = (
                    accepted_per_pos.get(int(line[start:end]), 0) + value
                )
        elif name == "vllm:spec_decode_num_accepted_tokens_total":
            num_accepted += value
    if not found:
        return None
    return {
        "num_drafts": num_drafts,
        "num_draft_tokens": num_draft_tokens,
        "num_accepted_tokens": num_accepted,
        "accepted_per_pos": accepted_per_pos,
    }


def delta(before: dict, after: dict) -> dict:
    accepted_per_pos = {}
    for pos, val in after["accepted_per_pos"].items():
        accepted_per_pos[pos] = val - before["accepted_per_pos"].get(pos, 0)
    return {
        "num_drafts": after["num_drafts"] - before["num_drafts"],
        "num_draft_tokens": after["num_draft_tokens"] - before["num_draft_tokens"],
        "num_accepted_tokens": (
            after["num_accepted_tokens"] - before["num_accepted_tokens"]
        ),
        "accepted_per_pos": accepted_per_pos,
    }


def mean_and_rates(m: dict) -> tuple[float | None, list[float]]:
    if m["num_drafts"] <= 0:
        return None, []
    mean = 1 + m["num_accepted_tokens"] / m["num_drafts"]
    max_pos = max(m["accepted_per_pos"], default=-1)
    rates = [
        m["accepted_per_pos"].get(pos, 0) / m["num_drafts"]
        for pos in range(max_pos + 1)
    ]
    return mean, rates


def send_chat(base_url: str, model: str, prompt: str, max_tokens: int) -> int:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    resp = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=1800)
    resp.raise_for_status()
    return int(resp.json()["usage"]["completion_tokens"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-ip", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=4144)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--dataset", choices=["humaneval", "gsm8k", "math500", "mbpp"], default="humaneval")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--num-prompts", type=int, default=-1)
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--replacement-delay-ms", type=float, default=0.0)
    parser.add_argument("--monitor-interval", type=float, default=10.0)
    parser.add_argument("--output-dir", default="results/domino_acceptance")
    args = parser.parse_args()

    base_url = f"http://{args.server_ip}:{args.server_port}"
    prompts = load_prompts(args.dataset_path, args.dataset)
    if args.num_prompts and args.num_prompts > 0:
        prompts = prompts[: args.num_prompts]

    before = fetch_metrics(base_url)
    if before is None:
        raise SystemExit("spec-decode metrics not available; enable log stats")

    work: queue.Queue = queue.Queue()
    for idx, (task_id, prompt) in enumerate(prompts, start=1):
        work.put((idx, task_id, prompt))
    lock = threading.Lock()
    done = {"count": 0}

    def worker() -> None:
        while True:
            try:
                idx, task_id, prompt = work.get_nowait()
            except queue.Empty:
                return
            try:
                tokens = send_chat(
                    base_url, args.served_model_name, prompt, args.max_tokens
                )
                with lock:
                    done["count"] += 1
                    print(
                        f"[w{idx % args.num_workers}] [{idx}/{len(prompts)}] "
                        f"{task_id} tokens={tokens}",
                        flush=True,
                    )
            finally:
                if args.replacement_delay_ms > 0:
                    time.sleep(args.replacement_delay_ms / 1000.0)

    monitor_stop = threading.Event()

    def monitor() -> None:
        last = before
        while not monitor_stop.is_set():
            monitor_stop.wait(args.monitor_interval)
            if monitor_stop.is_set():
                return
            try:
                now = fetch_metrics(base_url)
            except Exception:
                continue
            if now is None:
                continue
            d = delta(last, now)
            last = now
            mean, rates = mean_and_rates(d)
            with lock:
                print(
                    f"[interval] drafts={d['num_drafts']} "
                    f"accept_len={mean if mean is not None else float('nan'):.2f} "
                    f"per_pos={[round(r, 3) for r in rates]}",
                    flush=True,
                )

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
        futures = [pool.submit(worker) for _ in range(args.num_workers)]
        for future in futures:
            future.result()
    monitor_stop.set()
    monitor_thread.join()
    time.sleep(0.5)
    after = fetch_metrics(base_url)
    if after is None:
        raise SystemExit("metrics disappeared during run")
    d = delta(before, after)
    mean, rates = mean_and_rates(d)
    print(
        f"\nTotal drafts={d['num_drafts']} mean_acceptance_length={mean:.3f} "
        f"per_pos={[round(r, 3) for r in rates]}",
        flush=True,
    )

    import os

    os.makedirs(args.output_dir, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%y%m%d_%H%M%S")
    summary = {
        "num_workers": args.num_workers,
        "replacement_delay_ms": args.replacement_delay_ms,
        "num_requests": len(prompts),
        "completed": done["count"],
        "elapsed_s": round(time.monotonic() - start, 2),
        "mean_acceptance_length": mean,
        "per_position_acceptance_rates": rates,
        "num_drafts": d["num_drafts"],
        "num_accepted_tokens": d["num_accepted_tokens"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    path = os.path.join(args.output_dir, f"replacement_delay_{stamp}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: {path}")


if __name__ == "__main__":
    main()
