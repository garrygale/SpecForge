# coding=utf-8
"""Probe the trained ``ffn_readout`` fold logits of a draft checkpoint.

Diagnostic for "is the chunk mixture actually being used?": per layer it
prints the fold-logit magnitudes and how far each mixture column drifted
from the uniform init (total-variation distance of the softmax weights from
1/branches), plus an aggregate verdict —

* all layers essentially uniform  -> the mixture never left its init: the
  bottleneck is optimization (give ``fold_logits`` a dedicated higher-LR
  param group), not capacity;
* non-uniform mixtures present    -> the fold is being used; remaining
  accuracy gaps point back at capacity/structure.

Usage::

    python scripts/fold_logits_probe.py --checkpoint /path/to/draft
"""

from __future__ import annotations

import argparse
import pathlib

import torch

# A slot whose softmax weights stay this close to uniform (TV) read as unused.
UNIFORM_TV = 0.01
# Logits this small overall mean the mixture never really moved.
FLAT_LOGIT = 0.05


def _load_state_dict(checkpoint: str) -> dict:
    path = pathlib.Path(checkpoint)
    if path.is_dir():
        shards = sorted(path.glob("*.safetensors")) or [path / "model.safetensors"]
        if shards and shards[0].is_file():
            from safetensors.torch import load_file

            state = {}
            for shard in shards:
                state.update(load_file(str(shard)))
            return state
        candidates = sorted(path.glob("*.bin")) + sorted(path.glob("*.pt"))
        if not candidates:
            raise FileNotFoundError(f"no weight file under {path}")
        path = candidates[0]
    blob = torch.load(str(path), map_location="cpu", weights_only=True)
    if isinstance(blob, dict) and "draft_state_dict" in blob:
        blob = blob["draft_state_dict"]
    return blob


def _normalize_keys(state: dict) -> dict:
    return {
        name[len("model."):] if name.startswith("model.") else name: tensor
        for name, tensor in state.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    state = _normalize_keys(_load_state_dict(args.checkpoint))
    keys = sorted(
        (k for k in state if k.endswith("mlp.down_proj.fold_logits")),
        key=lambda k: int(k.split(".")[1]),
    )
    if not keys:
        raise SystemExit(
            "no layers.*.mlp.down_proj.fold_logits tensors found — the "
            "checkpoint's down_proj is dense or the ffn_readout knob was off"
        )

    print(f"fold-logit probe on {args.checkpoint} ({len(keys)} layers)\n")
    flat_layers = 0
    for key in keys:
        layer = key.split(".")[1]
        logits = state[key].float()
        branches, granularity = logits.shape
        weights = torch.softmax(logits, dim=0)
        tv = 0.5 * (weights - 1.0 / branches).abs().sum(dim=0)

        print(f"layer {layer}: branches={branches} granularity={granularity}")
        print(
            f"  logits: abs-mean {logits.abs().mean():.4f}  "
            f"abs-max {logits.abs().max():.4f}"
        )
        print(
            f"  slot TV from uniform: mean {tv.mean():.4f}  "
            f"median {tv.median():.4f}  max {tv.max():.4f}"
        )
        uniform_share = (tv < UNIFORM_TV).float().mean().item()
        decisive = int((tv > 0.1).sum())
        print(
            f"  essentially-uniform slots (TV<{UNIFORM_TV}): "
            f"{uniform_share:.0%}   decisive slots (TV>0.1): "
            f"{decisive}/{granularity}"
        )
        winners = torch.bincount(
            weights.argmax(dim=0), minlength=branches
        ).tolist()
        print(
            f"  dominant-branch histogram: {winners} "
            f"(uniform expectation ~{granularity / branches:.1f} each)"
        )
        hot = int(tv.argmax())
        print(
            f"  most-decided slot {hot}: weights "
            f"{[f'{v:.2f}' for v in weights[:, hot].tolist()]}"
        )
        if logits.abs().max() < FLAT_LOGIT:
            flat_layers += 1
            print("  -> essentially at the uniform init")
        print()

    if flat_layers == len(keys):
        print(
            "VERDICT: every layer's mixture stayed at the uniform init — an "
            "optimization bottleneck, not capacity. Give fold_logits its own "
            "param group with a higher LR (e.g. 10x the base) and/or warm "
            "start from a checkpoint whose mixture matters."
        )
    elif flat_layers:
        print(
            f"VERDICT: mixed — {flat_layers}/{len(keys)} layers stayed "
            "uniform; look at which (early layers idling is common and "
            "mostly harmless, late layers idling is not)."
        )
    else:
        print(
            "VERDICT: the mixtures are being used. Remaining accuracy gaps "
            "point at capacity/structure (G_g/branches budget, 2D fold), "
            "not at the mixture optimizer."
        )


if __name__ == "__main__":
    main()
