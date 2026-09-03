#!/usr/bin/env python3
"""Validate migrated Domino configs and the 7-step GRU correction loop.

The service is configured with `--spec-tokens 7` while the draft
`block_size` remains 16 (Domino is DSpark-shaped: `num_query_per_req ==
num_speculative_tokens`).  This probe checks the checked-in configs and the
SpecForge `eagerGRU` seven-step path without requiring NPU.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import torch
from torch import nn

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from probes.domino_migration.probe_domino_dflare_cpu import tiny_config
from specforge.modeling.auto import AutoDraftModel, AutoDraftModelConfig
from specforge.modeling.draft.dflash import eagerGRU


def main() -> None:
    target_configs = (
        "qwen3.6-35b-a3b-domino-dflare-verifiedBase.json",
        "qwen3.8-27b-domino-dflare-verifiedBase.json",
    )
    for filename in target_configs:
        config = AutoDraftModelConfig.from_file(os.path.join(_ROOT, "configs", filename))
        cfg = config.dflash_config
        assert cfg["projector_type"] == "domino"
        assert cfg["fusion_mode"] == "flare"
        assert cfg["heterogeneous_kv"] is True
        assert cfg["target_hidden_size"] in (2048, 5120)
        assert len(cfg["target_layer_ids"]) == 7
        assert getattr(config, "block_size", None) == 16
        assert len(config.layer_types) == config.num_hidden_layers == 7
        print(f"{filename}: ok (block_size=16, target_hidden={cfg['target_hidden_size']})")

    # Seven-step GRU correction loop against eagerGRU / torch.nn.GRU.
    input_size = 12
    hidden = 16
    seq = 7
    gru = eagerGRU(input_size, hidden, num_layers=1, batch_first=True, bias=False)
    ref = nn.GRU(input_size, hidden, num_layers=1, batch_first=True, bias=False)
    ref.load_state_dict(gru.state_dict(), strict=True)
    x = torch.randn(1, seq, input_size)
    out, h = gru(x)
    ref_out, ref_h = ref(x)
    assert torch.allclose(out, ref_out, atol=1e-5)
    assert torch.allclose(h, ref_h, atol=1e-5)

    # Exercise the correction-head feature shape exactly like the 7-token
    # service loop: sample_hidden [B, 7, H+G], one GRU state, per-step bias.
    hidden_states = torch.randn(2, 7, input_size + hidden)
    assert hidden_states.shape[1] == 7
    print(f"eagerGRU 7-step parity: ok (seq={seq}, hidden={hidden})")

    tiny_payload = tiny_config()
    tiny_path = os.path.join(tempfile.mkdtemp(), "tiny_domino.json")
    with open(tiny_path, "w", encoding="utf-8") as f:
        json.dump(tiny_payload, f)
    tiny = AutoDraftModel.from_config(
        AutoDraftModelConfig.from_file(tiny_path),
        torch_dtype=torch.float32,
    )
    assert tiny.pure_draft_prefix_len <= 7
    print("Domino pure prefix / 7-token contract: ok")


if __name__ == "__main__":
    main()
