#!/usr/bin/env python3
"""Validate the migrated target/draft configs and expected state-dict keys."""

from __future__ import annotations

import os
import tempfile

import torch

from probes.domino_migration.probe_domino_dflare_cpu import tiny_config
from specforge.modeling.auto import AutoDraftModel, AutoDraftModelConfig

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_EXPECTED_KEYS = {
    "input_proj.weight",
    "output_proj.weight",
    "layer_fusion_weights",
    "hidden_norm.weight",
    "norm.weight",
    "layers.0.self_attn.q_proj.weight",
    "layers.0.self_attn.k_proj.weight",
    "layers.0.self_attn.v_proj.weight",
    "layers.0.self_attn.o_proj.weight",
    "layers.0.self_attn.k_proj_target.weight",
    "layers.0.self_attn.v_proj_target.weight",
    "layers.0.mlp.gate_proj.weight",
    "layers.0.mlp.up_proj.weight",
    "layers.0.mlp.down_proj.weight",
    "prefix_gru.weight_ih_l0",
    "prefix_gru.weight_hh_l0",
    "embed_proj.0.weight",
    "embed_proj.2.weight",
}


def main() -> None:
    for filename in (
        "qwen3-8b-domino-dflare-verifiedBase.json",
        "qwen3.8-27b-domino-dflare-verifiedBase.json",
        "qwen3.6-35b-a3b-domino-dflare-verifiedBase.json",
    ):
        path = os.path.join(_ROOT, "configs", filename)
        config = AutoDraftModelConfig.from_file(path)
        assert config.dflash_config["fusion_mode"] == "flare"
        assert config.dflash_config["heterogeneous_kv"] is True
        assert config.dflash_config["target_hidden_size"] > 0
        print(f"{filename}: ok")

    payload = tiny_config()
    path = os.path.join(tempfile.mkdtemp(), "tiny.json")
    with open(path, "w", encoding="utf-8") as handle:
        import json

        json.dump(payload, handle)
    model = AutoDraftModel.from_config(
        AutoDraftModelConfig.from_file(path),
        torch_dtype=torch.float32,
    )
    keys = set(model.state_dict().keys())
    missing = _EXPECTED_KEYS - keys
    if missing:
        raise AssertionError(f"missing state-dict keys: {sorted(missing)}")
    print("tiny Domino state-dict keys: ok")


if __name__ == "__main__":
    main()
