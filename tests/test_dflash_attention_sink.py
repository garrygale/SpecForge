"""Learned attention-sink coverage for the DFlash family."""

import json
import os
import tempfile

import torch

from specforge.algorithms.common.dflash_family_model import create_dflash_sdpa_mask
from specforge.modeling.auto import AutoDraftModel, AutoDraftModelConfig


def tiny_config(attn_sink=0, **extra):
    dflash_config = {
        "mask_token_id": 63,
        "target_layer_ids": [0, 1],
        "sliding_window": [-1, 64],
    }
    if attn_sink:
        dflash_config["attn_sink"] = attn_sink
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


def build(attn_sink=0, seed=0):
    torch.manual_seed(seed)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "config.json")
        with open(path, "w") as f:
            json.dump(tiny_config(attn_sink), f)
        cfg = AutoDraftModelConfig.from_file(path)
        model = AutoDraftModel.from_config(cfg)
    model.config._attn_implementation = "eager"
    return model


def forward_inputs(model, S=48, device="cpu"):
    bs = model.block_size
    anchors = torch.tensor([[4, 12, 28]])
    keep = torch.ones_like(anchors, dtype=torch.bool)
    noise = torch.randn(1, anchors.numel() * bs, model.config.hidden_size)
    target_hidden = torch.randn(1, S, 64)
    draft_pos = (
        anchors.unsqueeze(-1) + torch.arange(bs).view(1, 1, -1)
    ).reshape(1, -1)
    pos = torch.cat([torch.arange(S).unsqueeze(0), draft_pos], dim=1)
    mask = create_dflash_sdpa_mask(
        anchors,
        keep,
        S,
        bs,
        torch.device(device),
        num_sinks=getattr(model, "num_sinks", 0),
    )
    return noise, target_hidden, pos, mask, anchors, keep


def test_off_by_default_has_no_sink_params():
    model = build(0)
    assert model.num_sinks == 0
    assert not any("sink" in k for k in model.state_dict())


def test_sink_params_exist_and_zero_init():
    model = build(1)
    assert model.num_sinks == 1
    for layer in model.layers:
        attn = layer.self_attn
        assert attn.sink_k.shape == (1, 2, 8)
        assert attn.sink_v.shape == (1, 2, 8)
        assert torch.all(attn.sink_k == 0)
        assert torch.all(attn.sink_v == 0)


def test_sink_off_matches_reference_mask():
    anchors = torch.tensor([[4, 12]])
    keep = torch.ones_like(anchors, dtype=torch.bool)
    for kwargs in (
        {},
        {"sliding_window": 64, "sliding_draft_causal": True},
        {"sliding_window": 64, "sliding_draft_causal": False},
    ):
        base = create_dflash_sdpa_mask(anchors, keep, 48, 4, torch.device("cpu"), **kwargs)
        with_sink_arg = create_dflash_sdpa_mask(
            anchors, keep, 48, 4, torch.device("cpu"), num_sinks=0, **kwargs
        )
        assert torch.equal(base, with_sink_arg)


def test_mask_sink_columns_visible_only_for_valid_blocks():
    anchors = torch.tensor([[4, 12]])
    keep = torch.tensor([[True, False]])
    mask = create_dflash_sdpa_mask(
        anchors, keep, 48, 4, torch.device("cpu"), sliding_window=64, num_sinks=1
    )
    assert mask.shape == (1, 1, 8, 48 + 8 + 1)
    sink_col = mask[0, 0, :, -1]
    assert torch.all(sink_col[:4])  # valid block rows see the sink
    assert torch.all(~sink_col[4:])  # invalid block rows stay fully masked
    # non-sink visibility is unchanged by num_sinks
    no_sink = create_dflash_sdpa_mask(
        anchors, keep, 48, 4, torch.device("cpu"), sliding_window=64, num_sinks=0
    )
    assert torch.equal(mask[..., :-1], no_sink)


def _capture_attn(model):
    captured = {}

    def hook(module, _args, output):
        captured["weights"] = output[1]
        captured["module"] = module

    model.layers[0].self_attn.register_forward_hook(hook)
    return captured


def test_forward_attends_sink_and_gradients_flow():
    model = build(1)
    noise, target_hidden, pos, mask, anchors, keep = forward_inputs(model)
    captured = _capture_attn(model)
    hidden = model(
        position_ids=pos,
        noise_embedding=noise,
        target_hidden=target_hidden,
        attention_mask=[mask, mask],
    )
    assert hidden.shape == noise.shape
    weights = captured["weights"][0].float()  # [H, Q, KV]
    assert torch.allclose(
        weights.sum(-1), torch.ones_like(weights.sum(-1)), atol=2e-2
    )
    p_sink = weights[..., -1].mean()
    assert p_sink > 0.01  # zero-logit sink grabs ~1/(n_visible+1) of the mass

    hidden.sum().backward()
    for layer in model.layers:
        assert layer.self_attn.sink_k.grad is not None
        assert layer.self_attn.sink_v.grad is not None
        assert layer.self_attn.sink_k.grad.abs().sum() > 0
        assert layer.self_attn.sink_v.grad.abs().sum() > 0


def test_dynamic_cache_never_holds_sink_rows():
    import gc

    from transformers import DynamicCache

    model = build(1)
    layer = model.layers[0]
    noise = torch.randn(1, 4, model.config.hidden_size)
    raw_target = torch.randn(1, 8, 64)
    target_hidden = model.hidden_norm(model.fc(raw_target))
    pos = torch.arange(12, dtype=torch.long).unsqueeze(0)
    pos_embeddings = model.rotary_emb(noise, pos)
    cache = DynamicCache()
    for expected in (12, 24):
        # Each generate-style call appends the full recomputed span
        # (ctx 8 + q 4); a leaked sink row would make it 13 / 25.
        layer(
            hidden_states=noise,
            target_hidden=target_hidden,
            position_embeddings=pos_embeddings,
            attention_mask=None,
            past_key_value=cache,
            use_cache=True,
        )
        assert cache.get_seq_length() == expected
    assert layer.self_attn.num_sinks == 1
    gc.collect()


def test_export_roundtrip_keeps_sink_params():
    import gc

    model = build(1)
    with torch.no_grad():
        model.layers[0].self_attn.sink_k.normal_(0, 0.02)
        model.layers[1].self_attn.sink_v.normal_(0, 0.02)
    tmp = tempfile.mkdtemp()
    try:
        model.save_pretrained(tmp)
        loaded = AutoDraftModel.from_pretrained(tmp)
        assert loaded.num_sinks == 1
        assert torch.allclose(
            loaded.layers[0].self_attn.sink_k, model.layers[0].self_attn.sink_k
        )
        assert torch.allclose(
            loaded.layers[1].self_attn.sink_v, model.layers[1].self_attn.sink_v
        )
        del loaded
        gc.collect()  # release the safetensors mmap before cleanup (Windows)
    finally:
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)


def test_training_wrapper_forward_with_sink():
    from specforge.algorithms.common.dflash_family_model import OnlineDFlashModel

    model = build(1)
    S = 48
    embed = torch.nn.Embedding(64, model.config.hidden_size)
    wrapper = OnlineDFlashModel(
        draft_model=model,
        target_lm_head=torch.nn.Identity(),
        target_embed_tokens=embed,
        mask_token_id=63,
        block_size=model.block_size,
        attention_backend="sdpa",
        num_anchors=4,
    )
    anchors, keep, output = wrapper._forward_draft_blocks(
        input_ids=torch.randint(0, 64, (1, S)),
        hidden_states=torch.randn(1, S, 64),
        loss_mask=torch.ones(1, S),
    )
    assert output.shape[1] == anchors.numel() * model.block_size
    assert keep.any()


def test_mla_rejects_sink():
    config = tiny_config(1, attention_mode="mla")
    config.update(
        {
            "kv_lora_rank": 16,
            "qk_nope_head_dim": 8,
            "qk_rope_head_dim": 8,
            "v_head_dim": 8,
        }
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "config.json")
        with open(path, "w") as f:
            json.dump(config, f)
        cfg = AutoDraftModelConfig.from_file(path)
        try:
            AutoDraftModel.from_config(cfg)
        except ValueError as exc:
            assert "attn_sink" in str(exc)
        else:
            raise AssertionError("MLA + attn_sink must raise ValueError")


if __name__ == "__main__":
    import sys

    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:
                failures += 1
                print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    sys.exit(1 if failures else 0)
