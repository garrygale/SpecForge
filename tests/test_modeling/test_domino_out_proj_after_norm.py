# coding=utf-8
"""out_proj_after_norm ordering on a tiny heterogeneous Domino draft."""

import json
import tempfile
import os
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from specforge.algorithms.common.dflash_family_model import OnlineDominoModel
from specforge.modeling.auto import AutoDraftModel, AutoDraftModelConfig


def tiny_payload(after_norm: bool | None = None) -> dict:
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
    }
    if after_norm is not None:
        dflash_config["out_proj_after_norm"] = after_norm
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


def build(after_norm: bool | None = None):
    payload = tiny_payload(after_norm)
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
        return SimpleNamespace(
            logits=logits,
            hidden_states=[hidden, hidden, hidden],
        )


class OutProjAfterNormTest(unittest.TestCase):
    def test_default_false_shapes(self):
        draft = build()
        self.assertFalse(draft.out_proj_after_norm)
        self.assertEqual(draft.norm.weight.numel(), 64)
        self.assertEqual(draft.embed_proj[0].in_features, 64 + 16)
        self.assertEqual(draft.prefix_gru.input_size, 64)

    def test_true_shapes(self):
        draft = build(True)
        self.assertTrue(draft.out_proj_after_norm)
        self.assertEqual(draft.norm.weight.numel(), 32)
        self.assertEqual(draft.embed_proj[0].in_features, 32 + 16)
        self.assertEqual(draft.prefix_gru.input_size, 64)

    def _forward(self, draft):
        torch.manual_seed(0)
        noise = torch.randn(1, 4, 64)
        target_hidden = torch.randn(1, 4, 64)
        calls = []

        def hook(name):
            def fn(module, inputs, output):
                calls.append((name, tuple(inputs[0].shape)))

            return fn

        draft.output_proj.register_forward_hook(hook("output_proj"))
        draft.norm.register_forward_hook(hook("norm"))
        out = draft(
            noise_embedding=noise,
            target_hidden=target_hidden,
            position_ids=torch.arange(8).unsqueeze(0),
        )
        return out, calls

    def test_false_order_projects_before_norm(self):
        draft = build(False)
        out, calls = self._forward(draft)
        self.assertEqual(tuple(out.shape), (1, 4, 64))
        self.assertEqual(calls[0][0], "output_proj")
        self.assertEqual(calls[0][1], (1, 4, 32))
        self.assertEqual(calls[1][0], "norm")
        self.assertEqual(calls[1][1], (1, 4, 64))
        self.assertEqual(tuple(draft.head_hidden(out).shape), (1, 4, 64))

    def test_true_order_norms_before_projection(self):
        draft = build(True)
        out, calls = self._forward(draft)
        self.assertEqual(tuple(out.shape), (1, 4, 32))
        self.assertEqual(calls, [("norm", (1, 4, 32))])
        self.assertEqual(tuple(draft.head_hidden(out).shape), (1, 4, 64))

    def test_true_head_hidden_matches_manual_composition(self):
        draft = build(True)
        captured = {}
        draft.norm.register_forward_pre_hook(
            lambda module, inputs: captured.__setitem__("x", inputs[0])
        )
        out, calls = self._forward(draft)
        self.assertEqual(calls, [("norm", (1, 4, 32))])
        with torch.no_grad():
            manual = draft.output_proj(draft.norm(captured["x"]))
            self.assertTrue(torch.equal(draft.head_hidden(out), manual))

        lm = nn.Linear(64, 64, bias=False)
        out2, _ = self._forward(draft)
        lm(draft.head_hidden(out2)).sum().backward()
        self.assertTrue(torch.isfinite(draft.output_proj.weight.grad).all())
        self.assertTrue(torch.isfinite(draft.norm.weight.grad).all())

    def test_correction_head_runs_on_draft_width(self):
        draft = build(True)
        hidden = torch.randn(2, 3, 4, 32)
        embeds = torch.randn(2, 3, 4, 64)
        correction = draft.compute_correction_logits(
            prev_token_embeddings=embeds,
            hidden_states=hidden,
        )
        self.assertEqual(tuple(correction.shape), (2, 3, 3, 64))

    def test_online_objective_both_modes(self):
        for after_norm in (False, True):
            draft = build(after_norm)
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
            self.assertTrue(torch.isfinite(loss), after_norm)
            self.assertTrue(torch.isfinite(accuracy), after_norm)

    def test_spec_generate_both_modes(self):
        for after_norm in (False, True):
            draft = build(after_norm)
            target = FakeTarget(64, 64)
            output_ids, stats = draft.spec_generate(
                target=target,
                input_ids=torch.tensor([[1, 2, 3, 4, 5, 6]]),
                max_new_tokens=12,
                stop_token_ids=[0],
                temperature=0.0,
                return_acceptance_stats=True,
            )
            self.assertGreater(output_ids.shape[1], 6, after_norm)
            self.assertIn("mean_acceptance_length", stats)


if __name__ == "__main__":
    unittest.main()
