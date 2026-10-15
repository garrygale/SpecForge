"""VHA premix/postmix training path and GQA fold parity."""

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from specforge.algorithms.common.dflash_family_model import create_dflash_sdpa_mask
from specforge.modeling.auto import AutoDraftModel, AutoDraftModelConfig
from specforge.modeling.draft.vha import fold_vha_state_dict

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "fold_vha.py"


def tiny_config(vha=None, **extra):
    dflash_config = {
        "mask_token_id": 63,
        "target_layer_ids": [0, 1],
        "sliding_window": [-1, 64],
    }
    if vha is not None:
        dflash_config["vha"] = vha
    dflash_config.update(extra)
    return {
        "architectures": ["DFlashDraftModel"],
        "model_type": "qwen3",
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "num_hidden_layers": 2,
        "layer_types": ["full_attention", "sliding_attention"],
        "num_target_layers": 2,
        "target_hidden_size": 32,
        "block_size": 4,
        "vocab_size": 64,
        "max_position_embeddings": 256,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000,
        "use_sliding_window": True,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "tie_word_embeddings": False,
        "use_cache": False,
        "dflash_config": dflash_config,
    }


def build(vha=None, seed=0, **extra):
    torch.manual_seed(seed)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "config.json")
        with open(path, "w") as f:
            json.dump(tiny_config(vha, **extra), f)
        cfg = AutoDraftModelConfig.from_file(path)
        model = AutoDraftModel.from_config(cfg)
    model.config._attn_implementation = "eager"
    model.eval()
    return model


def forward_inputs(model, S=48):
    bs = model.block_size
    anchors = torch.tensor([[4, 12, 28]])
    keep = torch.ones_like(anchors, dtype=torch.bool)
    noise = torch.randn(1, anchors.numel() * bs, model.config.hidden_size)
    target_hidden = torch.randn(1, S, 64)
    draft_pos = (anchors.unsqueeze(-1) + torch.arange(bs).view(1, 1, -1)).reshape(1, -1)
    pos = torch.cat([torch.arange(S).unsqueeze(0), draft_pos], dim=1)
    mask = create_dflash_sdpa_mask(anchors, keep, S, bs, torch.device("cpu"))
    return noise, target_hidden, pos, mask


def perturb_vha(model, seed=7):
    torch.manual_seed(seed)
    with torch.no_grad():
        for layer in model.layers:
            attn = layer.self_attn
            attn.vha_premix.add_(torch.randn_like(attn.vha_premix) * 0.3)
            attn.vha_postmix_u.add_(torch.randn_like(attn.vha_postmix_u) * 0.1)
            attn.vha_postmix_v.add_(torch.randn_like(attn.vha_postmix_v) * 0.1)


class VhaOffByDefault(unittest.TestCase):
    def test_no_vha_params(self):
        model = build()
        self.assertIsNone(model.layers[0].self_attn.vha)
        self.assertFalse(any("vha" in k for k in model.state_dict()))
        self.assertEqual(
            model.layers[0].self_attn.q_proj.weight.shape,
            (4 * 8, 32),
        )


class VhaParamsTest(unittest.TestCase):
    def test_shapes_and_init(self):
        model = build(vha={"physical_q_heads": 2})
        attn = model.layers[0].self_attn
        self.assertEqual(attn.vha["physical_q_heads"], 2)
        self.assertEqual(attn.vha["postmix_rank"], 1)
        self.assertEqual(attn.vha_premix.shape, (2, 8, 8))
        self.assertEqual(attn.vha_postmix_u.shape, (4, 1))
        self.assertEqual(attn.vha_postmix_v.shape, (4, 1))
        self.assertEqual(attn.q_proj.weight.shape, (2 * 8, 32))
        self.assertEqual(attn.o_proj.weight.shape, (32, 4 * 8))
        for group in range(2):
            gap = (attn.vha_premix[group] - torch.eye(8)).abs().max()
            self.assertLess(gap, 0.2)
            self.assertGreater(gap, 0.0)
        self.assertTrue(torch.all(attn.vha_postmix_v == 0))

    def test_rejects_mismatched_shape(self):
        with self.assertRaises(ValueError):
            build(vha={"physical_q_heads": 3})

    def test_rejects_unknown_keys(self):
        with self.assertRaises(ValueError):
            build(vha={"physical_q_heads": 2, "premix_sigma": 1.0})

    def test_rejects_non_gqa_mode(self):
        with self.assertRaises(ValueError):
            build(vha={"physical_q_heads": 2}, attention_mode="mla")


class VhaFoldParity(unittest.TestCase):
    def test_folded_gqa_matches_vha_forward(self):
        model_vha = build(vha={"physical_q_heads": 2})
        perturb_vha(model_vha)
        inputs = forward_inputs(model_vha)
        with torch.no_grad():
            expected = model_vha(
                position_ids=inputs[2],
                noise_embedding=inputs[0],
                target_hidden=inputs[1],
                attention_mask=[inputs[3], inputs[3]],
            )[0]

        folded_sd = fold_vha_state_dict(model_vha.state_dict())
        self.assertFalse(any("vha" in k for k in folded_sd))

        plain = build()
        plain.load_state_dict(folded_sd, strict=True)
        with torch.no_grad():
            actual = plain(
                position_ids=inputs[2],
                noise_embedding=inputs[0],
                target_hidden=inputs[1],
                attention_mask=[inputs[3], inputs[3]],
            )[0]
        self.assertTrue(torch.allclose(expected, actual, atol=1e-5, rtol=1e-5))

    def test_gradients_reach_vha_params(self):
        model = build(vha={"physical_q_heads": 2})
        perturb_vha(model)
        inputs = forward_inputs(model)
        hidden = model(
            position_ids=inputs[2],
            noise_embedding=inputs[0],
            target_hidden=inputs[1],
            attention_mask=[inputs[3], inputs[3]],
        )[0]
        hidden.sum().backward()
        for layer in model.layers:
            self.assertIsNotNone(layer.self_attn.vha_premix.grad)
            self.assertIsNotNone(layer.self_attn.vha_postmix_u.grad)
            self.assertIsNotNone(layer.self_attn.vha_postmix_v.grad)
            self.assertGreater(layer.self_attn.vha_premix.grad.abs().sum(), 0)
            self.assertGreater(layer.self_attn.vha_postmix_v.grad.abs().sum(), 0)


class FoldScriptTest(unittest.TestCase):
    def test_roundtrip(self):
        model = build(vha={"physical_q_heads": 2})
        perturb_vha(model)
        inputs = forward_inputs(model)
        with torch.no_grad():
            expected = model(
                position_ids=inputs[2],
                noise_embedding=inputs[0],
                target_hidden=inputs[1],
                attention_mask=[inputs[3], inputs[3]],
            )[0]

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "vha"
            dst = Path(tmp) / "folded"
            model.save_pretrained(src)
            with open(src / "config.json") as f:
                self.assertIn("vha", json.load(f)["dflash_config"])

            spec = importlib.util.spec_from_file_location("fold_vha", SCRIPT)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            old_argv = sys.argv
            sys.argv = ["fold_vha.py", "--input", str(src), "--output", str(dst)]
            try:
                module.main()
            finally:
                sys.argv = old_argv

            with open(dst / "config.json") as f:
                folded_config = json.load(f)
            self.assertNotIn("vha", folded_config["dflash_config"])
            reloaded = AutoDraftModel.from_pretrained(dst)
            reloaded.config._attn_implementation = "eager"
            reloaded.eval()
            with torch.no_grad():
                actual = reloaded(
                    position_ids=inputs[2],
                    noise_embedding=inputs[0],
                    target_hidden=inputs[1],
                    attention_mask=[inputs[3], inputs[3]],
                )[0]
            self.assertTrue(torch.allclose(expected, actual, atol=1e-5, rtol=1e-5))

    def test_refuses_plain_checkpoint(self):
        model = build()
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "plain"
            model.save_pretrained(src)
            spec = importlib.util.spec_from_file_location("fold_vha", SCRIPT)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            sys.argv = ["fold_vha.py", "--input", str(src), "--output", str(tmp)]
            with self.assertRaises(ValueError):
                module.main()


if __name__ == "__main__":
    unittest.main()
