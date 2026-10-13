"""Domino + DFlash2 local causal conv coverage."""

import json
import os
import tempfile

import torch

from specforge.algorithms.common.dflash_family_model import OnlineDominoModel
from specforge.modeling.auto import AutoDraftModel, AutoDraftModelConfig


def domino_config(conv: dict | None = None):
    dflash_config = {
        "projector_type": "domino",
        "mask_token_id": 63,
        "target_layer_ids": [0, 1],
        "sliding_window": [-1, 64],
        "emb_dim": 8,
        "gru_hidden_dim": 16,
        "shift_label": True,
        "use_hidden_proj": False,
        "pure_draft_prefix_len": 1,
    }
    if conv:
        dflash_config.update(conv)
    return {
        "architectures": ["DominoDraftModel"],
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


def build(conv):
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "config.json")
        with open(path, "w") as f:
            json.dump(domino_config(conv), f)
        cfg = AutoDraftModelConfig.from_file(path)
        return AutoDraftModel.from_config(cfg)


def test_domino_without_conv_keys_has_no_conv_modules():
    model = build(None)
    assert not any("conv" in k for k in model.state_dict())


def test_domino_with_conv_builds_and_runs():
    model = build({"conv_kernel_size": 2, "conv_group_size": 8})
    keys = model.state_dict()
    for layer in range(2):
        for part in ("attention_conv", "mlp_conv"):
            assert f"layers.{layer}.{part}.base_kernel" in keys
            assert f"layers.{layer}.{part}.kernel_projection.weight" in keys

    S = 48
    embed = torch.nn.Embedding(64, model.config.hidden_size)
    lm_head = torch.nn.Linear(
        model.target_hidden_size, model.config.vocab_size, bias=False
    )
    wrapper = OnlineDominoModel(
        draft_model=model,
        target_lm_head=lm_head,
        target_embed_tokens=embed,
        mask_token_id=63,
        block_size=model.block_size,
        attention_backend="sdpa",
        num_anchors=4,
        shift_label=True,
    )
    loss, accuracy, metrics = wrapper(
        input_ids=torch.randint(0, 64, (1, S)),
        hidden_states=torch.randn(1, S, 64),
        loss_mask=torch.ones(1, S),
    )
    assert torch.isfinite(loss)


def test_fresh_conv_is_exact_identity():
    """base identity + zero projection => prepare/finish chain is a no-op."""

    from specforge.modeling.draft.dflash2 import DFlashGroupedConv

    conv = DFlashGroupedConv(hidden_size=32, block_size=4, taps=3, group_size=8)
    x = torch.randn(2, 12, 32)  # [B, 3 blocks, H]
    prepared, coeff = conv.prepare(x)
    assert torch.equal(prepared, x)
    assert torch.equal(conv.finish(x, coeff), x)


def test_conv_export_roundtrip():
    import gc
    import shutil

    model = build({"conv_kernel_size": 2, "conv_group_size": 8})
    with torch.no_grad():
        model.layers[0].attention_conv.base_kernel.normal_(0, 0.05)
    tmp = tempfile.mkdtemp()
    try:
        model.save_pretrained(tmp)
        loaded = AutoDraftModel.from_pretrained(tmp)
        assert torch.allclose(
            loaded.layers[0].attention_conv.base_kernel,
            model.layers[0].attention_conv.base_kernel,
        )
        del loaded
        gc.collect()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


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
