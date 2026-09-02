#!/usr/bin/env python3
"""Merge per-NPU acceptance-length logs into one JSON result."""

import glob
import json
import os
import sys
from typing import Any, Optional


def aggregate_stats(
    per_problem_results: list[dict[str, Any]],
) -> tuple[Optional[float], Optional[float]]:
    valid = [
        entry
        for entry in per_problem_results
        if entry.get("mean_acceptance_length") is not None
        and entry.get("num_complete_blocks", 0) > 0
    ]
    if not valid:
        return None, None
    simple_mean = sum(
        entry["mean_acceptance_length"] for entry in valid
    ) / len(valid)
    weights = [entry["num_complete_blocks"] for entry in valid]
    weighted_mean = (
        sum(
            entry["mean_acceptance_length"] * weight
            for entry, weight in zip(valid, weights)
        )
        / sum(weights)
    )
    return simple_mean, weighted_mean


def merge_logs(file_paths: list[str]) -> list[dict[str, Any]]:
    all_entries = []
    seen = set()
    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                task_id = entry["task_id"]
                if task_id in seen:
                    print(f"Warning: duplicate task_id {task_id!r}, skipping")
                    continue
                seen.add(task_id)
                all_entries.append(entry)

    def sort_key(entry: dict) -> int:
        digits = "".join(c for c in entry["task_id"].split("/")[-1] if c.isdigit())
        return int(digits) if digits else 0

    all_entries.sort(key=sort_key)
    return all_entries


def main() -> int:
    if "--files" in sys.argv:
        index = sys.argv.index("--files")
        file_paths = sys.argv[index + 1 :]
    elif len(sys.argv) == 2:
        timestamp = sys.argv[1]
        script_dir = os.path.dirname(os.path.abspath(__file__))
        pattern = os.path.join(
            script_dir, f"{timestamp}_acceptance_lengths_npu*.jsonl"
        )
        file_paths = sorted(glob.glob(pattern))
        if not file_paths:
            print(f"No log files found matching: {pattern}")
            return 1
    else:
        print(__doc__)
        return 1

    entries = merge_logs(file_paths)
    simple, weighted = aggregate_stats(entries)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = "merged"
    if file_paths:
        timestamp = os.path.basename(file_paths[0]).split("_")[0]
    output_path = os.path.join(
        script_dir, f"{timestamp}_acceptance_lengths_merged.json"
    )
    merged = {
        "summary": {
            "overall_mean_acceptance_length_simple": simple,
            "overall_mean_acceptance_length_weighted": weighted,
            "num_problems": len(entries),
            "num_valid": sum(
                1 for entry in entries if entry.get("mean_acceptance_length") is not None
            ),
        },
        "results": entries,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)
        f.write("\n")
    print(f"Merged {len(file_paths)} log(s) into {output_path}")
    print(f"Mean acceptance length (simple):   {simple}")
    print(f"Mean acceptance length (weighted): {weighted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
