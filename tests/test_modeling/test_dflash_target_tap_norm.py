# coding=utf-8
"""Per-tap RMSNorm (``dflash_config.target_tap_norm``) on the DFlash fusion.

The flag normalizes every captured target tap before the fusion in all three
branches (fc / flare / per-layer). These tests pin the default-off invariance,
the parameter layout shared with the served Laguna drafts, the tap index space
used in per-layer mode, and the export round-trip.
"""

import gc
import os
import shutil
import tempfile
import time
import unittest

import torch
from torch import nn
from transformers import Qwen3Config

from specforge.algorithms.common.dflash_family_model import OnlineDFlashModel
from specforge.modeling.auto import AutoDraftModel

HIDDEN = 32
TARGET_HIDDEN = 32
TAPS = [1, 3, 5]
NUM_TARGET_LAYERS = 8
EPS = 1e-6


def _payload(*, tap_norm=None, fusion=None, per_layer_lists=None):
    dflash_config = {"mask_token_id": 63}
    if per_layer_lists is None:
        dflash_config["target_layer_ids"] = list(TAPS)
    else:
        dflash_config["target_layer_ids"] = [
            list(sublist) for sublist in per_layer_lists
        ]
    if fusion == "flare":
        dflash_config["fusion_mode"] = "flare"
        dflash_config["heterogeneous_kv"] = True
    if tap_norm is not None:
        dflash_config["target_tap_norm"] = tap_norm

    layers = len(per_layer_lists) if per_layer_lists is not None else 1
    return {
        "architectures": ["DFlashDraftModel"],
        "model_type": "qwen3",
        "hidden_size": HIDDEN,
        "intermediate_size": 64,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "num_hidden_layers": layers,
        "layer_types": ["full_attention"] * layers,
        "num_target_layers": NUM_TARGET_LAYERS,
        "target_hidden_size": TARGET_HIDDEN,
        "block_size": 4,
        "vocab_size": 64,
        "max_position_embeddings": 32,
        "rms_norm_eps": EPS,
        "rope_theta": 10000,
        "sliding_window": None,
        "use_sliding_window": False,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "tie_word_embeddings": False,
        "use_cache": True,
        "dflash_config": dflash_config,
    }


def build(**kwargs):
    # Same seed for every build so two models differ only by the tap norm.
    torch.manual_seed(0)
    config = Qwen3Config.from_dict(_payload(**kwargs))
    model = AutoDraftModel.from_config(config, torch_dtype=torch.float32)
    model.eval()
    return model


def _manual_rms_norm(x, weight, eps=EPS):
    """transformers' Qwen3RMSNorm math, written out by hand."""
    variance = x.pow(2).mean(-1, keepdim=True)
    return weight * (x * torch.rsqrt(variance + eps))


def _remove_tree(path: str) -> None:
    """Best-effort cleanup: Windows may still hold the safetensors mmap."""
    for _ in range(3):
        shutil.rmtree(path, ignore_errors=True)
        if not os.path.exists(path):
            return
        gc.collect()
        time.sleep(0.1)


def _module_input(model, target_hidden, module):
    """Run one draft forward and return the tensor ``module`` received."""
    captured = {}
    handle = module.register_forward_pre_hook(
        lambda _module, inputs: captured.setdefault("x", inputs[0])
    )
    seq_len = target_hidden.shape[1]
    try:
        with torch.no_grad():
            model(
                noise_embedding=torch.randn(1, seq_len, HIDDEN),
                target_hidden=target_hidden,
                # Context + noise, as spec_generate lays the positions out.
                position_ids=torch.arange(2 * seq_len).unsqueeze(0),
            )
    finally:
        handle.remove()
    return captured["x"]


def _set_distinct_norm_weights(model):
    """Give each tap a different gain so a wrong tap index cannot pass."""
    with torch.no_grad():
        for index, norm in enumerate(model.aux_hidden_norms):
            norm.weight.fill_(0.5 + index)


class TapNormBuildTest(unittest.TestCase):
    def test_default_off_fuses_the_raw_concat(self):
        model = build()
        self.assertFalse(model.target_tap_norm)
        self.assertIsNone(model.aux_hidden_norms)
        self.assertEqual(
            [k for k in model.state_dict() if "aux_hidden_norms" in k], []
        )
        target = torch.randn(1, 6, len(TAPS) * TARGET_HIDDEN)
        torch.testing.assert_close(
            _module_input(model, target, model.fc), target, rtol=0, atol=0
        )

    def test_flag_builds_one_norm_per_capture_layer(self):
        model = build(tap_norm=True)
        self.assertTrue(model.target_tap_norm)
        self.assertEqual(model.capture_layer_ids, TAPS)
        self.assertEqual(len(model.aux_hidden_norms), len(TAPS))
        self.assertEqual(
            {k for k in model.state_dict() if "aux_hidden_norms" in k},
            {f"aux_hidden_norms.{i}.weight" for i in range(len(TAPS))},
        )
        for norm in model.aux_hidden_norms:
            self.assertEqual(tuple(norm.weight.shape), (TARGET_HIDDEN,))
            self.assertTrue(torch.equal(norm.weight, torch.ones(TARGET_HIDDEN)))
        # The fusion widths are untouched: the norm is not a re-projection.
        self.assertEqual(model.fc.in_features, len(TAPS) * TARGET_HIDDEN)
        self.assertEqual(model.hidden_norm.weight.numel(), HIDDEN)

    def test_norm_weights_change_the_fused_feature(self):
        model = build(tap_norm=True)
        target = torch.randn(1, 6, len(TAPS) * TARGET_HIDDEN)
        before = _module_input(model, target, model.hidden_norm).clone()
        _set_distinct_norm_weights(model)
        after = _module_input(model, target, model.hidden_norm)
        self.assertFalse(torch.allclose(before, after))


class TapNormFusionTest(unittest.TestCase):
    def test_fc_mode_normalizes_each_tap_before_the_projection(self):
        model = build(tap_norm=True)
        _set_distinct_norm_weights(model)
        target = torch.randn(1, 6, len(TAPS) * TARGET_HIDDEN)
        taps = target.view(1, 6, len(TAPS), TARGET_HIDDEN)
        expected = torch.cat(
            [
                _manual_rms_norm(taps[:, :, i, :], model.aux_hidden_norms[i].weight)
                for i in range(len(TAPS))
            ],
            dim=-1,
        )
        torch.testing.assert_close(
            _module_input(model, target, model.fc),
            expected,
            rtol=1e-5,
            atol=1e-6,
        )
        with torch.no_grad():
            fused = model.fc(expected)
        torch.testing.assert_close(
            _module_input(model, target, model.hidden_norm),
            fused,
            rtol=1e-5,
            atol=1e-6,
        )

    def test_flare_mode_normalizes_each_tap_before_the_weighted_sum(self):
        model = build(tap_norm=True, fusion="flare")
        self.assertTrue(model._flare)
        _set_distinct_norm_weights(model)
        target = torch.randn(1, 6, len(TAPS) * TARGET_HIDDEN)
        taps = target.view(1, 6, len(TAPS), TARGET_HIDDEN)
        normed = torch.stack(
            [
                _manual_rms_norm(taps[:, :, i, :], model.aux_hidden_norms[i].weight)
                for i in range(len(TAPS))
            ],
            dim=2,
        )
        weights = torch.softmax(model.layer_fusion_weights, dim=1)[0].view(
            1, 1, -1, 1
        )
        fused = (normed * weights).sum(dim=2)
        torch.testing.assert_close(
            _module_input(model, target, model.hidden_norm),
            fused,
            rtol=1e-5,
            atol=1e-6,
        )

    def test_per_layer_mode_uses_the_capture_layer_order(self):
        # First-appearance order (5, 1, 3) differs from the sorted capture
        # order so a norm indexed by target_layer_ids position would fail.
        model = build(tap_norm=True, per_layer_lists=[[5, 1], [3, 5]])
        self.assertTrue(model._per_layer)
        self.assertEqual(model.capture_layer_ids, [1, 3, 5])
        self.assertEqual(model._per_layer_gather, [[0, 2], [1, 2]])
        _set_distinct_norm_weights(model)
        target = torch.randn(1, 6, 3 * TARGET_HIDDEN)
        taps = target.view(1, 6, 3, TARGET_HIDDEN)
        normed = [
            _manual_rms_norm(taps[:, :, i, :], model.aux_hidden_norms[i].weight)
            for i in range(3)
        ]
        for layer_index, gather in enumerate(model._per_layer_gather):
            expected = torch.cat([normed[g] for g in gather], dim=-1)
            torch.testing.assert_close(
                _module_input(model, target, model.fcs[layer_index]),
                expected,
                rtol=1e-5,
                atol=1e-6,
            )

    def test_default_off_leaves_every_branch_unchanged(self):
        cases = (
            {},
            {"fusion": "flare"},
            {"per_layer_lists": [[1, 3], [3, 5]]},
        )
        for kwargs in cases:
            with self.subTest(**kwargs):
                model = build(**kwargs)
                self.assertIsNone(model.aux_hidden_norms)
                target = torch.randn(1, 6, 3 * TARGET_HIDDEN)
                module = (
                    model.hidden_norms[0] if model._per_layer else model.hidden_norm
                )
                fused = _module_input(model, target, module)
                self.assertTrue(torch.isfinite(fused).all())


class TapNormTrainingTest(unittest.TestCase):
    def test_tap_norm_parameters_receive_gradient(self):
        model = build(tap_norm=True)
        target = torch.randn(
            1, 6, len(TAPS) * TARGET_HIDDEN, requires_grad=True
        )
        out = model(
            noise_embedding=torch.randn(1, 6, HIDDEN),
            target_hidden=target,
            position_ids=torch.arange(12).unsqueeze(0),
        )
        out.sum().backward()
        for norm in model.aux_hidden_norms:
            self.assertIsNotNone(norm.weight.grad)
            self.assertGreater(float(norm.weight.grad.abs().sum()), 0.0)
        self.assertTrue(torch.isfinite(target.grad).all())

    def test_online_objective_runs_with_and_without_tap_norm(self):
        for tap_norm in (False, True):
            with self.subTest(tap_norm=tap_norm):
                model = build(tap_norm=tap_norm)
                wrapper = OnlineDFlashModel(
                    draft_model=model,
                    target_lm_head=nn.Linear(HIDDEN, 64, bias=False),
                    target_embed_tokens=nn.Embedding(64, HIDDEN),
                    mask_token_id=63,
                    block_size=4,
                    attention_backend="sdpa",
                    num_anchors=2,
                )
                ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
                hidden = torch.randn(1, 8, len(TAPS) * TARGET_HIDDEN)
                out = wrapper(ids, hidden, torch.ones(1, 8))
                self.assertTrue(torch.isfinite(out[0]))

    def test_export_roundtrip_keeps_tap_norm_weights(self):
        model = build(tap_norm=True)
        _set_distinct_norm_weights(model)
        tmp = tempfile.mkdtemp()
        try:
            model.save_pretrained(tmp)
            loaded = AutoDraftModel.from_pretrained(tmp)
            self.assertTrue(loaded.target_tap_norm)
            self.assertEqual(len(loaded.aux_hidden_norms), len(TAPS))
            for before, after in zip(
                model.aux_hidden_norms, loaded.aux_hidden_norms
            ):
                self.assertTrue(torch.allclose(before.weight, after.weight))
            target = torch.randn(1, 6, len(TAPS) * TARGET_HIDDEN)
            torch.testing.assert_close(
                _module_input(model, target, model.hidden_norm),
                _module_input(loaded, target, loaded.hidden_norm),
                rtol=1e-5,
                atol=1e-6,
            )
            # Release the safetensors mmap before the directory is removed;
            # Windows otherwise refuses the delete and leaves test litter.
            del loaded
            gc.collect()
        finally:
            _remove_tree(tmp)


if __name__ == "__main__":
    unittest.main()
