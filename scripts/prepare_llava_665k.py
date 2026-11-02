#!/usr/bin/env python3
# coding=utf-8
"""Convert Ethlake/llava-665k arrow shards into the canonical conversation JSONL.

Each output row::

    {"id": ..., "subset": ...,
     "images": ["images/ab/<sha1>.jpg"],   # [] for text-only rows
     "conversations": [{"role": "user"|"assistant", "content": ...}]}

Images are extracted once into ``--image-dir`` keyed by sha1 of the encoded
bytes, so re-running over overlapping shards never rewrites a file and the
JSONL only carries relative paths -- image bytes never enter any training
cache downstream. Original encodings are kept (no re-encode, no PIL decode).

Reads individual ``.arrow`` shard files (works without the full 54-shard
dataset directory) or a complete ``load_from_disk`` directory.

Usage::

    python scripts/prepare_llava_665k.py --shards "tmp/shards/*.arrow" \
        --output data/llava665k.jsonl --image-dir data/llava665k_images
    python scripts/prepare_llava_665k.py --dataset-dir /data/llava-665k \
        --output data/llava665k.jsonl --image-dir data/llava665k_images
"""

import argparse
import glob
import hashlib
import json
import os
import sys
import time
from collections import Counter

from datasets import Dataset, Image, load_from_disk


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert llava-665k arrow shards to canonical JSONL + deduped image dir"
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--shards", nargs="+", help="arrow shard files / glob patterns")
    src.add_argument("--dataset-dir", help="complete save_to_disk directory")
    parser.add_argument("--output", required=True, help="output JSONL path")
    parser.add_argument("--image-dir", required=True, help="image extraction directory")
    parser.add_argument("--subsets", default=None, help="comma-separated subset filter")
    parser.add_argument("--num-samples", type=int, default=None, help="stop after N rows")
    parser.add_argument("--report-every", type=int, default=5000)
    return parser.parse_args()


def sniff_fmt(head):
    if head.startswith(b"\xff\xd8"):
        return "jpg"
    if head.startswith(b"\x89PNG"):
        return "png"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "webp"
    if head.startswith(b"GIF8"):
        return "gif"
    return "bin"


def load_datasets(args):
    if args.dataset_dir:
        ds = load_from_disk(args.dataset_dir)
        ds = ds.cast_column("images", Image(decode=False))
        yield os.path.basename(args.dataset_dir), ds
        return
    paths = []
    for pattern in args.shards:
        expanded = glob.glob(pattern)
        paths.extend(expanded if expanded else [pattern])
    for path in sorted(paths):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"shard not found: {path}")
        ds = Dataset.from_file(path).cast_column("images", Image(decode=False))
        yield path, ds


def save_image(image_dir, data):
    digest = hashlib.sha1(data).hexdigest()
    rel = f"{digest[:2]}/{digest}.{sniff_fmt(data[:16])}"
    path = os.path.join(image_dir, rel)
    existed = os.path.isfile(path)
    if not existed:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    return rel, existed


def main():
    args = parse_args()
    subsets = set(args.subsets.split(",")) if args.subsets else None
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    os.makedirs(args.image_dir, exist_ok=True)

    stats = Counter()
    turns = Counter()
    written_images = set()
    image_bytes = 0
    n = 0
    started = time.time()

    with open(args.output, "w", encoding="utf-8") as out:
        for shard, ds in load_datasets(args):
            for i in range(len(ds)):
                if args.num_samples is not None and n >= args.num_samples:
                    break
                row = ds[i]
                subset = row["subset"]
                if subsets is not None and subset not in subsets:
                    continue
                images = []
                img = row["images"]
                if img is not None:
                    data = img["bytes"]
                    if data is None and img.get("path"):
                        with open(img["path"], "rb") as fh:
                            data = fh.read()
                    if data is None:
                        raise ValueError(f"row {row['id']!r} has an image without bytes")
                    rel, existed = save_image(args.image_dir, data)
                    images.append(rel)
                    if not existed:
                        stats["image_files_written"] += 1
                        image_bytes += len(data)
                    else:
                        stats["image_files_reused"] += 1
                    written_images.add(rel)
                conversations = []
                for pair in row["texts"]:
                    conversations.append({"role": "user", "content": pair["user"]})
                    conversations.append(
                        {"role": "assistant", "content": pair["assistant"]}
                    )
                if not conversations or conversations[0]["role"] != "user":
                    raise ValueError(f"row {row['id']!r} does not start with a user turn")
                rec = {
                    "id": row["id"],
                    "subset": subset,
                    "images": images,
                    "conversations": conversations,
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                stats[f"subset:{subset}"] += 1
                stats["rows_with_image" if images else "rows_text_only"] += 1
                turns[len(row["texts"])] += 1
                n += 1
                if n % args.report_every == 0:
                    print(f"  {n} rows, {time.time() - started:.0f}s", flush=True)
            if args.num_samples is not None and n >= args.num_samples:
                break
            print(f"[{shard}] done, cumulative rows={n}", flush=True)
        out.flush()

    multi = sum(c for t, c in turns.items() if t >= 2)
    print("=" * 60)
    print(f"prepare_llava_665k report  ({time.time() - started:.1f}s)")
    print("=" * 60)
    print(f"rows written         : {n}")
    print(f"  with image         : {stats['rows_with_image']}")
    print(f"  text-only          : {stats['rows_text_only']}")
    print(f"subsets              : { {k[len('subset:'):]: v for k, v in stats.items() if k.startswith('subset:')} }")
    print(f"turn histogram       : {dict(sorted(turns.items()))}")
    print(f"multi-turn (>=2)     : {multi} ({multi / max(n, 1):.1%})")
    print(f"unique images        : {len(written_images)}")
    print(f"  newly written      : {stats['image_files_written']} ({image_bytes / 1e9:.2f} GB)")
    print(f"  reused on disk     : {stats['image_files_reused']}")
    print(f"output               : {os.path.abspath(args.output)}")
    print(f"image dir            : {os.path.abspath(args.image_dir)}")


if __name__ == "__main__":
    sys.exit(main())
