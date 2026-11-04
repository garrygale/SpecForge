# coding=utf-8
"""SwiGLU-gated correction bottleneck on a tiny Domino draft."""

import json
import tempfile
import os
import unittest

import torch
import torch.nn.functional as F
from torch import nn

from specforge.algorithms.common.dflash_family_model import OnlineDominoModel
from specforge.modeling.auto import AutoDraftModel, AutoDraftModelConfig


def tiny_payload(**dflash_overrides) -> dict:
    dflash_config = {
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
        "target_hidden_size": 64,
        "out_proj_after_norm": True,
        "shared_emb_readout": False,
        "correction_swiglu": True,
    }
    dflash_config.update(dflash_overrides)
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
        "dflash_config": dflash_config,
    }


def build(**dflash_overrides):
    payload = tiny_payload(**dflash_overrides)
    path = os.path.join(tempfile.mkdtemp(), "tiny_domino.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    config = AutoDraftModelConfig.from_file(path)
    return AutoDraftModel.from_config(config, torch_dtype=torch.float32)


class FakeTarget(nn.Module):
    def __init__(self, hidden: int, vocab: int):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab, hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids, position_ids=None, logits_to_keep=None, **kwargs):
        hidden = self.model.embed_tokens(input_ids)
        hidden = hidden + position_ids.float().unsqueeze(-1) * 0.01
        logits = self.lm_head(hidden)
        if logits_to_keep is not None:
            logits = logits[:, -int(logits_to_keep) :]
        return type(
            "Out", (), {"logits": logits, "hidden_states": [hidden, hidden, hidden]}
        )()


class CorrectionSwiGLUTest(unittest.TestCase):
    def test_module_layout(self):
        draft = build()
        self.assertTrue(draft.correction_swiglu)
        fc1 = draft.embed_proj[0]
        self.assertEqual(fc1.in_features, 32 + 16)
        self.assertEqual(fc1.out_features, 2 * 8)
        self.assertEqual(draft.embed_proj[2].weight.shape, (64, 8))

    def test_default_stays_silu(self):
        draft = build(correction_swiglu=False)
        self.assertFalse(draft.correction_swiglu)
        self.assertIsInstance(draft.embed_proj[1], nn.SiLU)
        self.assertEqual(draft.embed_proj[0].out_features, 8)

    def test_guards(self):
        with self.assertRaisesRegex(ValueError, "shared_emb_readout"):
            build(shared_emb_readout=True)
        with self.assertRaisesRegex(ValueError, "use_embed_proj"):
            build(use_embed_proj=False)
        with self.assertRaisesRegex(ValueError, "use_hidden_proj"):
            build(use_hidden_proj=True)

    def test_correction_matches_manual_swiglu(self):
        draft = build()
        torch.manual_seed(0)
        hidden = torch.randn(2, 3, 4, 32)
        embeds = torch.randn(2, 3, 4, 64)

        with torch.no_grad():
            corr = draft.compute_correction_logits(
                prev_token_embeddings=embeds, hidden_states=hidden
            )
            gru_out = draft._run_gru(embeds.reshape(6, 4, -1)).reshape(2, 3, 4, -1)
            prefix_states = gru_out[:, :, draft.suffix_start :, :]
            z = hidden[:, :, draft.suffix_start :, :]
            feats = torch.cat([z, prefix_states], dim=-1)

            w1 = draft.embed_proj[0].weight  # [2M, H + G]
            gate, up = (
                z @ w1[:, :32].t() + prefix_states @ w1[:, 32:].t()
            ).chunk(2, dim=-1)
            manual = (F.silu(gate) * up) @ draft.embed_proj[2].weight.t()
        torch.testing.assert_close(corr, manual, atol=1e-4, rtol=1e-4)

    def test_gradients_reach_doubled_fc1(self):
        draft = build()
        lm_head = nn.Linear(64, 64, bias=False)
        hidden = torch.randn(2, 3, 4, 32)
        embeds = torch.randn(2, 3, 4, 64)
        base_logits = lm_head(draft.head_hidden(hidden))
        final = draft.apply_logits_head(
            base_logits,
            prev_token_embeddings=embeds,
            hidden_states=hidden,
            lm_head=lm_head,
        )
        final.sum().backward()
        w1 = draft.embed_proj[0].weight
        self.assertIsNotNone(w1.grad)
        self.assertTrue(torch.isfinite(w1.grad).all())
        for name in ("weight_ih_l0", "weight_hh_l0"):
            weight = getattr(draft.prefix_gru, name)
            self.assertIsNotNone(weight.grad, name)

    def test_online_objective(self):
        draft = build()
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
        loss, accuracy, _ = trainer(ids, hidden_states, torch.ones(1, 8))
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(accuracy))

    def test_spec_generate(self):
        draft = build()
        target = FakeTarget(64, 64)
        output_ids, stats = draft.spec_generate(
            target=target,
            input_ids=torch.tensor([[1, 2, 3, 4, 5, 6]]),
            max_new_tokens=12,
            stop_token_ids=[0],
            temperature=0.0,
            return_acceptance_stats=True,
        )
        self.assertGreater(output_ids.shape[1], 6)
        self.assertIn("mean_acceptance_length", stats)


if __name__ == "__main__":
    unittest.main()
