"""HCA (compressed_attention) pooling, mask seam rules, and forward parity."""

import json
import os
import tempfile
import unittest

import torch

from specforge.algorithms.common.dflash_family_model import create_dflash_sdpa_mask
from specforge.algorithms.model_providers import describe_draft_features
from specforge.modeling.auto import AutoDraftModel, AutoDraftModelConfig
from specforge.modeling.draft.dflash import (
    apply_rotary_pos_emb,
    hat_window_kernel,
    validate_dflash_attention_backend,
)


HCA_LAYER_TYPES = ["heavily_compressed_attention", "sliding_attention"]
HCA_WINDOWS = [64, 64]


def tiny_config(hca=None, layer_types=None, sliding_window=None, **extra):
    dflash_config = {
        "mask_token_id": 63,
        "target_layer_ids": [0, 1],
    }
    if hca is not None:
        dflash_config["hca"] = hca
    if sliding_window is not None:
        dflash_config["sliding_window"] = sliding_window
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
        "layer_types": layer_types or ["full_attention", "sliding_attention"],
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


def build(hca=None, layer_types=None, sliding_window=None, seed=0, **extra):
    torch.manual_seed(seed)
    config = tiny_config(hca, layer_types, sliding_window, **extra)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "config.json")
        with open(path, "w") as f:
            json.dump(config, f)
        cfg = AutoDraftModelConfig.from_file(path)
        model = AutoDraftModel.from_config(cfg)
    model.config._attn_implementation = "eager"
    model.eval()
    return model


def build_hca(hca, seed=0, **extra):
    return build(
        hca=hca,
        layer_types=HCA_LAYER_TYPES,
        sliding_window=HCA_WINDOWS,
        seed=seed,
        **extra,
    )


def forward_inputs(model, S=128):
    bs = model.block_size
    anchors = torch.tensor([[64, 96, 120]])
    keep = torch.ones_like(anchors, dtype=torch.bool)
    noise = torch.randn(1, anchors.numel() * bs, model.config.hidden_size)
    target_hidden = torch.randn(1, S, 64)
    draft_pos = (anchors.unsqueeze(-1) + torch.arange(bs).view(1, 1, -1)).reshape(1, -1)
    pos = torch.cat([torch.arange(S).unsqueeze(0), draft_pos], dim=1)
    return anchors, keep, noise, target_hidden, pos


def layer_mask(model, anchors, keep, S, spec, causal=False):
    return create_dflash_sdpa_mask(
        anchors,
        keep,
        S,
        model.block_size,
        torch.device("cpu"),
        sliding_window=spec["window"],
        sliding_draft_causal=causal,
        hca=spec,
    )


def per_layer_masks(model, anchors, keep, S):
    """One mask per layer: hca spec, plain sliding, or full, matching the model."""

    masks = []
    specs = model.layer_hca_params or [None] * model.config.num_hidden_layers
    for window, spec in zip(model.layer_sliding_windows, specs):
        if spec is not None:
            masks.append(layer_mask(model, anchors, keep, S, spec))
        else:
            masks.append(
                create_dflash_sdpa_mask(
                    anchors,
                    keep,
                    S,
                    model.block_size,
                    torch.device("cpu"),
                    sliding_window=window,
                )
            )
    return masks


class HcaOffByDefault(unittest.TestCase):
    def test_no_hca_params(self):
        model = build(sliding_window=[-1, 64])
        self.assertIsNone(model.layer_hca_params)
        self.assertFalse(any("hca" in k for k in model.state_dict()))
        self.assertIsNone(model.layers[0].self_attn.hca)


class HcaConfigTest(unittest.TestCase):
    def test_shapes_and_init_vanilla(self):
        model = build_hca({"variants": "vanilla", "hops": 16})
        attn = model.layers[0].self_attn
        sliding = model.layers[1].self_attn
        self.assertIsNotNone(attn.hca)
        self.assertIsNone(sliding.hca)
        self.assertFalse(hasattr(sliding, "hca_compressor"))
        spec = model.layer_hca_params[0]
        self.assertEqual((spec["hop"], spec["win"], spec["coff"]), (16, 16, 1))
        self.assertEqual(spec["max_entries"], 2048 - 64 - 4 - 64)
        self.assertEqual(attn.hca_compressor.kv_proj.weight.shape, (8, 32))
        self.assertEqual(attn.hca_compressor.gate_proj.weight.shape, (8, 32))
        self.assertEqual(attn.hca_compressor.position_bias.shape, (16, 8))
        self.assertTrue(torch.all(attn.hca_compressor.position_bias == 0))
        self.assertTrue(attn.hca_compressor.position_bias.requires_grad)

    def test_shapes_and_init_hat(self):
        model = build_hca({"variants": "hat", "hops": 8, "bias_mode": "frozen"})
        spec = model.layer_hca_params[0]
        self.assertEqual((spec["hop"], spec["win"], spec["coff"]), (8, 16, 2))
        bias = model.layers[0].self_attn.hca_compressor.position_bias
        expected = torch.log(hat_window_kernel(16) + 1e-8).unsqueeze(-1).expand(16, 8)
        self.assertTrue(torch.allclose(bias, expected.to(bias.dtype), atol=1e-6))
        self.assertFalse(bias.requires_grad)
        # The tent kernel peaks at the window center.
        self.assertEqual(int(bias[:, 0].argmax()), 7)

    def test_rejects_compressed_without_hca(self):
        with self.assertRaises(ValueError):
            build(layer_types=HCA_LAYER_TYPES, sliding_window=HCA_WINDOWS)

    def test_rejects_hca_without_compressed(self):
        with self.assertRaises(ValueError):
            build(hca={"variants": "vanilla"})

    def test_rejects_unknown_keys(self):
        with self.assertRaises(ValueError):
            build_hca({"variants": "vanilla", "hops": 16, "stride": 4})

    def test_rejects_bad_variant(self):
        with self.assertRaises(ValueError):
            build_hca({"variants": "rc5"})

    def test_rejects_wrong_hops_length(self):
        with self.assertRaises(ValueError):
            build_hca({"variants": "vanilla", "hops": [16, 16]})

    def test_rejects_short_window(self):
        # win(16) + block_size(4) > 16: the seam would drop tokens.
        with self.assertRaises(ValueError):
            build(
                hca={"variants": "vanilla", "hops": 16},
                layer_types=HCA_LAYER_TYPES,
                sliding_window=[16, 64],
            )

    def test_rejects_window_over_budget(self):
        with self.assertRaises(ValueError):
            build(
                hca={"variants": "vanilla", "hops": 16},
                layer_types=HCA_LAYER_TYPES,
                sliding_window=[3000, 64],
            )

    def test_rejects_mla_mode(self):
        with self.assertRaises(ValueError):
            build_hca(
                {"variants": "vanilla", "hops": 16},
                attention_mode="mla",
                kv_lora_rank=8,
                qk_nope_head_dim=4,
                qk_rope_head_dim=4,
                v_head_dim=8,
            )

    def test_rejects_flex_backend(self):
        from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

        config = Qwen3Config.from_dict(
            tiny_config({"variants": "vanilla", "hops": 16}, HCA_LAYER_TYPES, HCA_WINDOWS)
        )
        with self.assertRaises(ValueError):
            validate_dflash_attention_backend(config, "flex_attention")


def hand_visibility(spec, anchors, S, block_size, max_entries=None):
    """Per-query (t, band_lower, visible entries, emitted count) from the R2 rule."""

    hop, win, W = spec["hop"], spec["win"], spec["window"]
    E = max_entries if max_entries is not None else spec["max_entries"]
    capacity = (S - win) // hop + 1 if S >= win else 0
    rows = []
    for anchor in anchors:
        emitted = max(0, (anchor - win) // hop + 1)
        for off in range(block_size):
            t = anchor + off
            band_lower = t + 1 - W
            visible = [
                j
                for j in range(capacity)
                if j * hop < band_lower and j >= emitted - E
            ]
            rows.append((anchor, t, band_lower, visible, emitted))
    return rows


class HcaSeamMaskTest(unittest.TestCase):
    def _check(self, variant, hop, win, S=48, anchors=(4, 12, 28), max_entries=None):
        model = build_hca({"variants": variant, "hops": hop}, seed=1)
        spec = dict(model.layer_hca_params[0])
        if max_entries is not None:
            spec["max_entries"] = max_entries
        block_size = model.block_size
        anchors_t = torch.tensor([list(anchors)])
        keep = torch.ones_like(anchors_t, dtype=torch.bool)
        mask = layer_mask(model, anchors_t, keep, S, spec)
        capacity = (S - win) // hop + 1 if S >= win else 0
        self.assertEqual(mask.shape[-1], capacity + S + anchors_t.numel() * block_size)

        rows = hand_visibility(spec, list(anchors), S, block_size, max_entries)
        for q, (anchor, t, band_lower, visible, emitted) in enumerate(rows):
            row = mask[0, 0, q]
            self.assertEqual(
                [j for j in range(capacity) if bool(row[j])], visible, f"t={t}"
            )
            # Raw context band: [max(band_lower, 0), anchor).
            band_lo = max(band_lower, 0)
            self.assertEqual(
                [p for p in range(S) if bool(row[capacity + p])],
                list(range(band_lo, min(t + 1, anchor))),
                f"t={t}",
            )
            # No gap: every token below the band lies in a visible entry.
            for p in range(0, min(band_lower, anchor)):
                covered = any(j * hop <= p < j * hop + win for j in visible)
                self.assertTrue(covered, f"t={t} p={p} uncovered")
            # Bounded seam double count.
            overlap = [
                p
                for p in range(band_lower, anchor)
                if any(j * hop <= p < j * hop + win for j in visible)
            ]
            self.assertLessEqual(len(overlap), win - 1, f"t={t}")
            if variant == "vanilla":
                # Interior tokens sit in exactly one entry.
                for p in range(0, min(band_lower - win, anchor)):
                    count = sum(1 for j in visible if j * hop <= p < j * hop + win)
                    self.assertEqual(count, 1, f"t={t} p={p}")
        return mask

    def test_vanilla_seam(self):
        self._check("vanilla", hop=16, win=16)

    def test_hat_seam(self):
        self._check("hat", hop=8, win=16)

    def test_max_entries_window_triggers(self):
        mask = self._check("vanilla", hop=16, win=16, max_entries=1)
        capacity = (48 - 16) // 16 + 1
        for q in range(mask.shape[2]):
            visible = mask[0, 0, q].nonzero().flatten().tolist()
            self.assertLessEqual(len([c for c in visible if c < capacity]), 1)

    def test_short_context_degenerates_to_swa(self):
        model = build_hca({"variants": "vanilla", "hops": 16}, seed=1)
        spec = model.layer_hca_params[0]
        anchors = torch.tensor([[12]])
        keep = torch.ones_like(anchors, dtype=torch.bool)
        mask = layer_mask(model, anchors, keep, 16, spec)
        capacity = (16 - 16) // 16 + 1
        self.assertTrue(torch.all(~mask[0, 0, :, :capacity]))
        # The whole context stays raw-visible for the newest query.
        self.assertTrue(torch.all(mask[0, 0, -1, capacity : capacity + 12]))


class HcaForwardParity(unittest.TestCase):
    """Compressed-layer forward vs a loop-based reference implementation."""

    def _reference_layer0(self, model, noise, target_hidden, pos, mask, spec):
        attn = model.layers[0].self_attn
        noise = model.layers[0].input_layernorm(noise)
        hop, win = spec["hop"], spec["win"]
        S = target_hidden.shape[1]
        L = noise.shape[1]
        nE = (S - win) // hop + 1 if S >= win else 0

        # Pooling by explicit per-window loops (independent of unfold/broadcast).
        kv = attn.hca_compressor.kv_proj(target_hidden)[0]
        gate = attn.hca_compressor.gate_proj(target_hidden)[0]
        bias = attn.hca_compressor.position_bias
        entries = []
        for j in range(nE):
            logits = gate[j * hop : j * hop + win] + bias
            weights = torch.softmax(logits.float(), dim=0).to(kv.dtype)
            pooled = (kv[j * hop : j * hop + win] * weights).sum(0)
            entries.append(attn.hca_compressor.norm(pooled.unsqueeze(0))[0])
        entries = (
            torch.stack(entries)
            if entries
            else kv.new_zeros(0, kv.shape[-1])
        )

        cos, sin = model.rotary_emb(noise, pos)
        anchor_off = 0 if spec["coff"] == 1 else win // 2
        anchor_pos = torch.arange(nE) * hop + anchor_off
        # RoPE the K view at the anchors; the V view stays unrotated.
        cos_e = cos[:, anchor_pos, :]
        sin_e = sin[:, anchor_pos, :]

        def rotate_half(x):
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat((-x2, x1), dim=-1)

        # [1, 1, nE, hd]: single shared head, ready to expand over KV heads.
        k_entry = entries * cos_e.unsqueeze(0) + rotate_half(entries) * sin_e.unsqueeze(
            0
        )

        q = attn.q_proj(noise).view(1, L, -1, attn.head_dim)
        q = attn.q_norm(q).transpose(1, 2)
        k = torch.cat(
            [attn.k_proj(target_hidden), attn.k_proj(noise)], dim=1
        ).view(1, S + L, -1, attn.head_dim)
        v = torch.cat(
            [attn.v_proj(target_hidden), attn.v_proj(noise)], dim=1
        ).view(1, S + L, -1, attn.head_dim)
        k = attn.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        k = torch.cat([k_entry.expand(-1, k.shape[1], -1, -1), k], dim=2)
        v = torch.cat(
            [entries.unsqueeze(0).unsqueeze(0).expand(-1, v.shape[1], -1, -1), v],
            dim=2,
        )

        k = torch.repeat_interleave(k, attn.num_key_value_groups, dim=1)
        v = torch.repeat_interleave(v, attn.num_key_value_groups, dim=1)
        scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * attn.scaling
        scores = scores.masked_fill(
            ~mask[0, 0].unsqueeze(0), torch.finfo(scores.dtype).min
        )
        out = torch.matmul(torch.softmax(scores, dim=-1), v.float()).to(v.dtype)
        out = out.transpose(1, 2).reshape(1, L, -1)
        return attn.o_proj(out)

    def _parity(self, variant, hop):
        model = build_hca({"variants": variant, "hops": hop}, seed=3)
        spec = model.layer_hca_params[0]
        anchors, keep, noise, target_hidden, pos = forward_inputs(model)
        mask = layer_mask(model, anchors, keep, 128, spec)
        sliding_mask = create_dflash_sdpa_mask(
            anchors,
            keep,
            128,
            model.block_size,
            torch.device("cpu"),
            sliding_window=64,
        )
        with torch.no_grad():
            actual = model(
                position_ids=pos,
                noise_embedding=noise,
                target_hidden=target_hidden,
                attention_mask=[mask, sliding_mask],
            )[0]
            # Mirror the model forward's target transform (base fusion path).
            layer_target = model.hidden_norm(model.fc(target_hidden))
            expected_layer0 = self._reference_layer0(
                model, noise, layer_target, pos, mask, spec
            )
            layer0_attn = model.layers[0].self_attn(
                hidden_states=model.layers[0].input_layernorm(noise),
                target_hidden=layer_target,
                attention_mask=mask,
                position_ids=pos,
                position_embeddings=model.rotary_emb(noise, pos),
            )[0]
        self.assertTrue(
            torch.allclose(layer0_attn, expected_layer0, atol=1e-4, rtol=1e-4),
            msg=f"layer0 attention mismatch ({variant})",
        )
        self.assertTrue(torch.isfinite(actual).all())

    def test_parity_vanilla(self):
        self._parity("vanilla", hop=16)

    def test_parity_hat(self):
        self._parity("hat", hop=8)


class HcaFailClosedAndBanner(unittest.TestCase):
    def test_maskless_forward_raises(self):
        model = build_hca({"variants": "vanilla", "hops": 16})
        _, _, noise, target_hidden, pos = forward_inputs(model)
        with self.assertRaises(ValueError):
            model(
                position_ids=pos,
                noise_embedding=noise,
                target_hidden=target_hidden,
                attention_mask=None,
            )

    def test_banner_lists_hca(self):
        model = build_hca({"variants": "vanilla", "hops": 16})
        banner = describe_draft_features(model)
        self.assertIn("hca(", banner)
        self.assertIn("hops=[16]", banner)

    def test_gradients_reach_compressor(self):
        model = build_hca({"variants": "vanilla", "hops": 16})
        anchors, keep, noise, target_hidden, pos = forward_inputs(model)
        hidden = model(
            position_ids=pos,
            noise_embedding=noise,
            target_hidden=target_hidden,
            attention_mask=per_layer_masks(model, anchors, keep, 128),
        )[0]
        hidden.sum().backward()
        attn = model.layers[0].self_attn
        params = dict(attn.named_parameters())
        for name in ("hca_compressor.kv_proj.weight", "hca_compressor.gate_proj.weight"):
            self.assertGreater(params[name].grad.abs().sum(), 0)
        self.assertIsNotNone(attn.hca_compressor.position_bias.grad)

    def test_frozen_bias_gets_no_grad(self):
        model = build_hca({"variants": "hat", "hops": 8, "bias_mode": "frozen"})
        anchors, keep, noise, target_hidden, pos = forward_inputs(model)
        hidden = model(
            position_ids=pos,
            noise_embedding=noise,
            target_hidden=target_hidden,
            attention_mask=per_layer_masks(model, anchors, keep, 128),
        )[0]
        hidden.sum().backward()
        bias = model.layers[0].self_attn.hca_compressor.position_bias
        self.assertFalse(bias.requires_grad)
        self.assertIsNone(bias.grad)


if __name__ == "__main__":
    unittest.main()
