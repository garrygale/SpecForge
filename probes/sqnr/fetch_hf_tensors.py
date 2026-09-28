# coding=utf-8
"""Range-fetch individual tensors from a HuggingFace safetensors checkpoint.

The two official drafts are 5.0 GB (shanjiaz/dflash-qwen3-8b) and 2.8 GB
(AngelSlim/Qwen3-8b-dflare), but the SQNR ledger needs only a handful of small
tensors (``fc.weight``, ``layer_fusion_weights``, the norms). This script reads
the safetensors HEADER with two HTTP range requests and then downloads only the
byte ranges of the requested tensors, so nothing large is ever transferred.

Note: ``web_fetch``/curl cannot reach huggingface.co in this environment (it
resolves to a fake-IP proxy and schannel fails), but python's own TLS stack can,
so this uses ``urllib``.

Usage:
    python probes/sqnr/fetch_hf_tensors.py --repo shanjiaz/dflash-qwen3-8b \
        --list
    python probes/sqnr/fetch_hf_tensors.py --repo shanjiaz/dflash-qwen3-8b \
        --tensor fc.weight --tensor hidden_norm.weight --out <dir>
    python probes/sqnr/fetch_hf_tensors.py --repo AngelSlim/Qwen3-8b-dflare \
        --tensor layer_fusion_weights --out <dir>

Each fetched tensor is written as ``<repo>__<tensor>.pt`` (a torch tensor) plus
one ``<repo>__shapes.json`` with the full header, so the ledger can be rerun
without touching the network.
"""

import argparse
import json
import os
import struct
import urllib.request

import torch

_API = "https://huggingface.co/api/models/"


def http_get(url: str, byte_range=None, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url)
    if byte_range is not None:
        req.add_header("Range", "bytes=%d-%d" % byte_range)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def read_header(repo: str, filename: str = "model.safetensors"):
    """Return (url, header, data_start).

    ``data_offsets`` in a safetensors header are relative to the START OF THE
    DATA SECTION, which begins right after the JSON header -- i.e. at
    ``8 + header_len``, NOT at 8. Using 8 silently reads the tail of the header
    JSON as bf16 for the leading tensors (which is how this was caught: a
    ``hidden_norm.weight`` that should be ~1.0 read as 1e37 with mean=inf).
    """
    url = "https://huggingface.co/%s/resolve/main/%s" % (repo, filename)
    n = struct.unpack("<Q", http_get(url, (0, 7)))[0]
    header = json.loads(http_get(url, (8, 8 + n - 1)))
    return url, header, 8 + n


_NP2TORCH = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "I64": torch.int64,
    "I32": torch.int32,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}


def fetch_tensor(url: str, header: dict, name: str, data_start: int):
    """Download and decode one tensor using its declared byte range."""
    if name not in header:
        raise KeyError(name)
    entry = header[name]
    begin, end = entry["data_offsets"]
    raw = http_get(url, (data_start + begin, data_start + end - 1))
    dtype = _NP2TORCH[entry["dtype"]]
    flat = torch.frombuffer(bytearray(raw), dtype=dtype)
    return flat.reshape(entry["shape"]).clone(), len(raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--file", default="model.safetensors")
    ap.add_argument("--tensor", action="append", default=[],
                    help="tensor name to fetch (repeatable)")
    ap.add_argument("--list", action="store_true", help="only list the header")
    ap.add_argument("--out", default=".", help="output directory")
    args = ap.parse_args()

    url, header, data_start = read_header(args.repo, args.file)
    if args.list:
        for k, v in header.items():
            if k == "__metadata__":
                print("metadata:", v)
                continue
            size = v["data_offsets"][1] - v["data_offsets"][0]
            print("%-50s %-6s %-22s %8.2f MB" % (k, v["dtype"], v["shape"], size / 1e6))
        return

    os.makedirs(args.out, exist_ok=True)
    tag = args.repo.split("/")[-1]
    with open(os.path.join(args.out, tag + "__header.json"), "w", encoding="utf-8") as f:
        json.dump(header, f, indent=1)
    for name in args.tensor:
        t, nbytes = fetch_tensor(url, header, name, data_start)
        path = os.path.join(args.out, tag + "__" + name.replace(".", "_") + ".pt")
        torch.save(t, path)
        print("[fetch] %-46s %-10s %-20s %8.2f MB -> %s"
              % (name, t.dtype, list(t.shape), nbytes / 1e6, path))


if __name__ == "__main__":
    main()
