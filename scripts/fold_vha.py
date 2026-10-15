"""Fold a trained VHA draft checkpoint into a standard GQA one.

Virtual Head Attention (arXiv:2610.17959) trains with a per-KV-group premix
on the query path and a low-rank ``I + U V^T`` postmix on the attention
output. Both are linear, so they absorb exactly: premix into q_proj (physical
heads expand to group-major virtual heads), postmix into o_proj. The emitted
checkpoint keeps num_attention_heads at its virtual value and drops every
``vha_*`` key plus ``dflash_config.vha``.

vLLM (>= codex/DRAFTv2 with the load-time fold) serves the *unfolded* bf16
VHA checkpoint directly, so this script is only needed (a) before the QAT
stage — int-packed QAT exports cannot be folded at load — or (b) for serving
stacks without a VHA-aware domino loader:

    python scripts/fold_vha.py --input ckpt_vha/ --output ckpt_folded/

Weights are folded in fp32 and cast back to the stored dtype. The script is
idempotent-safe: it refuses a checkpoint that carries no vha_premix keys.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from specforge.modeling.draft.vha import fold_vha_state_dict, vha_layer_prefixes


def load_state_dict(path: Path) -> tuple[dict, Path | None]:
    index = path / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        state_dict = {}
        for file in sorted(set(weight_map.values())):
            state_dict.update(load_file(path / file))
        return state_dict, index
    single = path / "model.safetensors"
    if not single.exists():
        raise FileNotFoundError(f"no safetensors found under {path}")
    return load_file(single), None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="VHA checkpoint directory")
    parser.add_argument("--output", required=True, help="output directory")
    args = parser.parse_args()

    src = Path(args.input)
    dst = Path(args.output)
    dst.mkdir(parents=True, exist_ok=True)

    config_path = src / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"missing {config_path}")
    config = json.loads(config_path.read_text())
    dflash_config = dict(config.get("dflash_config") or {})
    if not dflash_config.get("vha"):
        raise ValueError(
            "dflash_config.vha not set; refusing to fold a non-VHA checkpoint"
        )

    state_dict, index = load_state_dict(src)
    prefixes = vha_layer_prefixes(state_dict)
    print(f"found {len(prefixes)} VHA layers: {', '.join(p.split('.')[1] for p in prefixes)}")

    folded = fold_vha_state_dict(
        {k: v.to(torch.float32) for k, v in state_dict.items()}
    )
    dtypes = {k: state_dict[k].dtype for k in state_dict}
    folded = {k: v.to(dtypes.get(k, torch.float32)) for k, v in folded.items()}

    for key in ("num_attention_heads", "num_key_value_heads", "head_dim"):
        print(f"  {key}: {config.get(key)}")
    print(
        f"  q_proj folded to virtual width "
        f"{config['num_attention_heads'] * config['head_dim']}"
    )

    del dflash_config["vha"]
    config["dflash_config"] = dflash_config
    (dst / "config.json").write_text(json.dumps(config, indent=2))

    save_file(folded, dst / "model.safetensors", metadata={"format": "pt"})
    skip = {"config.json", "model.safetensors", "model.safetensors.index.json"}
    for extra in src.iterdir():
        if extra.name in skip or extra.name.endswith(".safetensors"):
            continue
        if extra.is_file():
            shutil.copy(extra, dst / extra.name)
    print(f"folded checkpoint written to {dst}")


if __name__ == "__main__":
    main()
