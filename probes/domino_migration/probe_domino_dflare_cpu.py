#!/usr/bin/env python3
"""CPU smoke probe for the migrated dflare Domino model.

Run from the SpecForge repository root:

    set PYTHONPATH=C:/Users/g/Desktop/codeAgents/SpecForge
    C:/Users/g/Desktop/codeAgents/phi-GNNv2/.venv/Scripts/python.exe probes/domino_migration/probe_domino_dflare_cpu.py
"""

from __future__ import annotations

import json
import os
import tempfile
from types import SimpleNamespace

import torch
from torch import nn

from specforge.algorithms.common.dflash_family_model import OnlineDominoModel
from specforge.layers.wxay import replace_linear_with_quantized
from specforge.modeling.auto import AutoDraftModel, AutoDraftModelConfig
from specforge.modeling.draft.dflash import eagerGRU


def tiny_config() -> dict:
    return {
        "architectures": ["DominoDraftModel"],
        "model_type": "qwen3",
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "num_hidden_layers": 1,
        "layer_types": ["sliding_attention"],
        "num_target_layers": 2,
        "target_hidden_size": 64,
        "block_size": 4,
        "vocab_size": 64,
        "max_position_embeddings": 32,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000,
        "sliding_window": [4],
        "use_sliding_window": True,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "tie_word_embeddings": False,
        "use_cache": True,
        "dflash_config": {
            "mask_token_id": 63,
            "target_layer_ids": [1],
            "projector_type": "domino",
            "fusion_mode": "flare",
            "heterogeneous_kv": True,
            "pure_draft_prefix_len": 1,
            "emb_dim": 8,
            "gru_hidden_dim": 16,
            "shift_label": True,
            "use_hidden_proj": False,
            "hidden_proj_dim": 16,
            "target_hidden_size": 64,
        },
    }


class FakeTarget(nn.Module):
    """Minimal text-causal target used only for acceptance-shape testing."""

    def __init__(self, hidden: int, vocab: int):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab, hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def forward(
        self,
        input_ids,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
        logits_to_keep=None,
        output_hidden_states=False,
        **kwargs,
    ):
        hidden = self.model.embed_tokens(input_ids)
        pos = position_ids if position_ids is not None else torch.zeros_like(input_ids)
        hidden = hidden + pos.float().unsqueeze(-1) * 0.01
        logits = self.lm_head(hidden)
        if logits_to_keep is not None:
            logits = logits[:, -int(logits_to_keep) :]
        return SimpleNamespace(
            logits=logits,
            hidden_states=[hidden, hidden, hidden],
        )


def main() -> None:
    payload = tiny_config()
    path = os.path.join(tempfile.mkdtemp(), "tiny_domino.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    config = AutoDraftModelConfig.from_file(path)
    draft = AutoDraftModel.from_config(config, torch_dtype=torch.float32)
    assert isinstance(draft.prefix_gru, eagerGRU)
    reference_gru = nn.GRU(64, 16, num_layers=1, batch_first=True, bias=False)
    reference_gru.load_state_dict(draft.prefix_gru.state_dict(), strict=True)
    gru_input = torch.randn(1, 4, 64)
    eager_output, eager_hidden = draft.prefix_gru(gru_input)
    reference_output, reference_hidden = reference_gru(gru_input)
    assert torch.allclose(eager_output, reference_output, atol=1e-5)
    assert torch.allclose(eager_hidden, reference_hidden, atol=1e-5)
    print("eagerGRU parity: ok", type(draft.prefix_gru).__name__)

    # Forward / backward through flare, heterogeneous K/V, and dimensions.
    noise = torch.randn(1, 4, 64)
    target_hidden = torch.randn(1, 4, 64)
    out = draft(
        noise_embedding=noise,
        target_hidden=target_hidden,
        position_ids=torch.arange(8).unsqueeze(0),
    )
    assert tuple(out.shape) == (1, 4, 64)
    out.sum().backward()
    print("draft forward/backward: ok", tuple(out.shape))

    # Online training wrapper objective.
    lm = nn.Linear(64, 64, bias=False)
    embed = nn.Embedding(64, 64)
    trainer = OnlineDominoModel(
        draft_model=draft,
        target_lm_head=lm,
        target_embed_tokens=embed,
        mask_token_id=63,
        block_size=4,
        attention_backend="sdpa",
        num_anchors=2,
        loss_decay_gamma=7.0,
        shift_label=True,
    )
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    hidden_states = torch.randn(1, 8, 64)
    loss, accuracy, metrics = trainer(
        ids,
        hidden_states,
        torch.ones(1, 8),
    )
    assert torch.isfinite(loss)
    assert torch.isfinite(accuracy)
    print("online trainer: ok", float(loss), float(accuracy))

    # QAT replacement remains numerically constructible.
    replace_linear_with_quantized(
        draft,
        w_bit=4,
        a_bit=8,
        exclude_names=["embed_proj"],
    )
    qat_out = draft(
        noise_embedding=noise,
        target_hidden=target_hidden,
        position_ids=torch.arange(8).unsqueeze(0),
    )
    assert tuple(qat_out.shape) == (1, 4, 64)
    print("qat forward: ok", type(draft.layers[0].self_attn.q_proj).__name__)

    # Acceptance generation + stats against the fake target.
    fresh = AutoDraftModel.from_config(config, torch_dtype=torch.float32)
    target = FakeTarget(64, 64)
    output_ids, stats = fresh.spec_generate(
        target=target,
        input_ids=ids[:, :6],
        max_new_tokens=12,
        stop_token_ids=[0],
        temperature=0.0,
        return_acceptance_stats=True,
    )
    assert tuple(output_ids.shape)[1] > 6
    assert "mean_acceptance_length" in stats
    print("spec_generate/stats: ok", stats["num_complete_blocks"])


if __name__ == "__main__":
    main()
