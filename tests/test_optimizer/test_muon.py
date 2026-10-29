"""Muon optimizer mode: param split, step behaviour, lr scheduling, checkpoint
round-trips and factory guards."""

import json
import os
import tempfile
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from specforge.optimizer import BF16Optimizer, split_params_for_muon

TORCH_HAS_MUON = hasattr(torch.optim, "Muon")


class _SyntheticDraft(nn.Module):
    """Carries one param of every kind the draft inventory has: hidden 2D
    matrices, 1D norms, embeddings/vocab projections (hint and shape-guard
    cases), a degenerate [1, N] head, table-like 2D params and 3D tensors."""

    def __init__(self, vocab=96, hidden=32):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.fc = nn.Linear(2 * hidden, hidden, bias=False)
        self.prefix_gru = nn.GRU(hidden, 16, batch_first=True, bias=False)
        self.embed_proj = nn.Sequential(
            nn.Linear(hidden + 16, 8, bias=False),
            nn.SiLU(),
            nn.Linear(8, vocab, bias=False),
        )
        self.markov_head = nn.ModuleDict(
            {
                "markov_w1": nn.Embedding(vocab, 8),
                "markov_w2": nn.Linear(8, vocab, bias=False),
            }
        )
        self.confidence_head = nn.ModuleDict(
            {"proj": nn.Linear(hidden, 1, bias=False)}
        )
        self.norm_weight = nn.Parameter(torch.ones(hidden))
        self.layer_fusion_weights = nn.Parameter(torch.ones(2, 3))
        self.position_bias = nn.Parameter(torch.randn(8, 8))
        self.codebook_weight = nn.Parameter(torch.randn(vocab, 8))
        self.sink_k = nn.Parameter(torch.randn(2, 4, 8))


class TestSplitParamsForMuon(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = _SyntheticDraft()
        self.muon, self.adamw = split_params_for_muon(
            self.model.named_parameters(), vocab_dims=(96,)
        )
        self.muon_names = {name for name, _ in self.muon}
        self.adamw_names = {name for name, _ in self.adamw}

    def test_expected_partition(self):
        self.assertEqual(
            self.muon_names,
            {
                "q_proj.weight",
                "fc.weight",
                "prefix_gru.weight_ih_l0",
                "prefix_gru.weight_hh_l0",
                "embed_proj.0.weight",
            },
        )
        self.assertEqual(
            self.adamw_names,
            {
                "embed_tokens.weight",
                "lm_head.weight",
                "embed_proj.2.weight",
                "markov_head.markov_w1.weight",
                "markov_head.markov_w2.weight",
                "confidence_head.proj.weight",
                "norm_weight",
                "layer_fusion_weights",
                "position_bias",
                "codebook_weight",
                "sink_k",
            },
        )

    def test_vocab_guard_is_load_bearing(self):
        # Without vocab_dims the domino vocab projection would silently ride
        # the Muon group — the exact speculators markov-head failure.
        muon, _ = split_params_for_muon(self.model.named_parameters())
        self.assertIn("embed_proj.2.weight", {name for name, _ in muon})

    def test_extra_hints_move_params_to_adamw(self):
        muon, adamw = split_params_for_muon(
            self.model.named_parameters(), vocab_dims=(96,), name_hints=("fc",)
        )
        self.assertNotIn("fc.weight", {name for name, _ in muon})
        self.assertIn("fc.weight", {name for name, _ in adamw})


def _tiny_payload(arch, vocab, dflash_overrides):
    dflash_config = {
        "mask_token_id": vocab - 1,
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
    }
    dflash_config.update(dflash_overrides)
    return {
        "architectures": [arch],
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
        "vocab_size": vocab,
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


def _build_tiny(arch, vocab, **dflash_overrides):
    from specforge.modeling.auto import AutoDraftModel, AutoDraftModelConfig

    payload = _tiny_payload(arch, vocab, dflash_overrides)
    path = os.path.join(tempfile.mkdtemp(), "tiny_draft.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    config = AutoDraftModelConfig.from_file(path)
    return AutoDraftModel.from_config(config, torch_dtype=torch.float32)


class TestMuonGoldenClassification(unittest.TestCase):
    """Exact partitions for the real tiny draft models. A new trainable param
    fails here until its group assignment is consciously decided — the
    regression test the speculators Muon port lacked."""

    def _partition(self, model, vocab):
        muon, adamw = split_params_for_muon(
            [(n, p) for n, p in model.named_parameters() if p.requires_grad],
            vocab_dims=(vocab,),
        )
        return {name for name, _ in muon}, {name for name, _ in adamw}

    def test_domino_default_head(self):
        model = _build_tiny("DominoDraftModel", 96)
        muon, adamw = self._partition(model, 96)
        self.assertEqual(
            muon,
            {
                "layers.0.self_attn.q_proj.weight",
                "layers.0.self_attn.k_proj.weight",
                "layers.0.self_attn.v_proj.weight",
                "layers.0.self_attn.o_proj.weight",
                "layers.0.self_attn.k_proj_target.weight",
                "layers.0.self_attn.v_proj_target.weight",
                "layers.0.mlp.gate_proj.weight",
                "layers.0.mlp.up_proj.weight",
                "layers.0.mlp.down_proj.weight",
                "input_proj.weight",
                "output_proj.weight",
                "prefix_gru.weight_ih_l0",
                "prefix_gru.weight_hh_l0",
                "embed_proj.0.weight",
            },
        )
        self.assertEqual(
            adamw,
            {
                "layer_fusion_weights",
                "layers.0.self_attn.q_norm.weight",
                "layers.0.self_attn.k_norm.weight",
                "layers.0.input_layernorm.weight",
                "layers.0.post_attention_layernorm.weight",
                "norm.weight",
                "hidden_norm.weight",
                "embed_proj.2.weight",
            },
        )

    def test_domino_shared_emb_readout(self):
        model = _build_tiny("DominoDraftModel", 96, shared_emb_readout=True)
        muon, adamw = self._partition(model, 96)
        self.assertEqual(
            muon,
            {
                "layers.0.self_attn.q_proj.weight",
                "layers.0.self_attn.k_proj.weight",
                "layers.0.self_attn.v_proj.weight",
                "layers.0.self_attn.o_proj.weight",
                "layers.0.self_attn.k_proj_target.weight",
                "layers.0.self_attn.v_proj_target.weight",
                "layers.0.mlp.gate_proj.weight",
                "layers.0.mlp.up_proj.weight",
                "layers.0.mlp.down_proj.weight",
                "input_proj.weight",
                "prefix_gru.weight_ih_l0",
                "prefix_gru.weight_hh_l0",
                "head_down_proj.weight",
                "head_up_proj.weight",
                "embed_proj.weight",
            },
        )
        self.assertEqual(
            adamw,
            {
                "layer_fusion_weights",
                "layers.0.self_attn.q_norm.weight",
                "layers.0.self_attn.k_norm.weight",
                "layers.0.input_layernorm.weight",
                "layers.0.post_attention_layernorm.weight",
                "norm.weight",
                "hidden_norm.weight",
            },
        )

    def test_dspark_markov_head_stays_on_adamw(self):
        model = _build_tiny(
            "DSparkDraftModel",
            96,
            projector_type="dspark",
            markov_head_type="vanilla",
            markov_rank=8,
        )
        muon, adamw = self._partition(model, 96)
        self.assertEqual(
            muon,
            {
                "layers.0.self_attn.q_proj.weight",
                "layers.0.self_attn.k_proj.weight",
                "layers.0.self_attn.v_proj.weight",
                "layers.0.self_attn.o_proj.weight",
                "layers.0.self_attn.k_proj_target.weight",
                "layers.0.self_attn.v_proj_target.weight",
                "layers.0.mlp.gate_proj.weight",
                "layers.0.mlp.up_proj.weight",
                "layers.0.mlp.down_proj.weight",
                "input_proj.weight",
                "output_proj.weight",
            },
        )
        self.assertEqual(
            adamw,
            {
                "layer_fusion_weights",
                "layers.0.self_attn.q_norm.weight",
                "layers.0.self_attn.k_norm.weight",
                "layers.0.input_layernorm.weight",
                "layers.0.post_attention_layernorm.weight",
                "norm.weight",
                "hidden_norm.weight",
                "markov_head.markov_w1.weight",
                "markov_head.markov_w2.weight",
            },
        )


class _TinyTrainable(nn.Module):
    def __init__(self, vocab=96, hidden=32):
        super().__init__()
        self.fc1 = nn.Linear(hidden, 64, bias=False)
        self.norm_w = nn.Parameter(torch.ones(64))
        self.head = nn.Linear(64, vocab, bias=False)
        self.config = SimpleNamespace(vocab_size=vocab)


def _set_grads(model, seed):
    generator = torch.Generator().manual_seed(seed)
    for p in model.parameters():
        p.grad = torch.randn(p.shape, generator=generator)


@unittest.skipUnless(TORCH_HAS_MUON, "torch>=2.9 required for torch.optim.Muon")
class TestMuonOptimizerStep(unittest.TestCase):
    def _optimizer(self, model=None, **kwargs):
        model = model or _TinyTrainable()
        defaults = dict(
            lr=1e-2,
            total_steps=100,
            warmup_ratio=0.0,
            lr_scheduler="constant",
            optimizer_type="muon",
        )
        defaults.update(kwargs)
        return BF16Optimizer(model, **defaults), model

    def test_muon_steps_and_reduces_loss(self):
        torch.manual_seed(0)
        model = nn.Sequential(
            nn.Linear(32, 64, bias=False), nn.GELU(), nn.Linear(64, 32, bias=False)
        )
        model.norm_w = nn.Parameter(torch.ones(32))
        optimizer = BF16Optimizer(
            model, lr=5e-2, total_steps=200, lr_scheduler="constant",
            optimizer_type="muon",
        )
        torch.manual_seed(1)
        inputs = torch.randn(128, 32)
        targets = inputs @ torch.randn(32, 32)
        criterion = nn.MSELoss()
        first = last = None
        for step in range(80):
            loss = criterion(model(inputs), targets)
            optimizer.model.zero_grad()
            loss.backward()
            last = float(loss)
            if first is None:
                first = last
            optimizer.step()
        self.assertLess(last, first * 0.5)

    def test_lr_ratio_holds_through_schedule(self):
        optimizer, _ = self._optimizer(
            lr=1e-3, muon_lr=0.03, total_steps=100, warmup_ratio=0.2,
            lr_scheduler="cosine",
        )
        ratio = 0.03 / 1e-3

        def check():
            adamw_lr = optimizer.optimizer.param_groups[0]["lr"]
            muon_lr = optimizer.muon_optimizer.param_groups[0]["lr"]
            self.assertGreater(float(adamw_lr), 0.0)
            self.assertAlmostEqual(
                float(muon_lr), float(adamw_lr) * ratio, places=9
            )

        check()  # post-init sync already matches warmup step 1
        model = optimizer.model
        for step in range(10):
            _set_grads(model, seed=step)
            optimizer.step()
        check()  # mid warmup
        for step in range(10, 60):
            _set_grads(model, seed=step)
            optimizer.step()
        check()  # mid cosine

    def test_nonfinite_grad_skips_step(self):
        optimizer, model = self._optimizer()
        before = [p.detach().clone() for p in model.parameters()]
        model.fc1.weight.grad = torch.full_like(
            model.fc1.weight.grad
            if model.fc1.weight.grad is not None
            else model.fc1.weight,
            float("inf"),
        )
        with self.assertLogs("specforge.optimizer", level="WARNING"):
            optimizer.step()
        for p, snapshot in zip(model.parameters(), before):
            self.assertTrue(torch.equal(p.detach(), snapshot))


@unittest.skipUnless(TORCH_HAS_MUON, "torch>=2.9 required for torch.optim.Muon")
class TestMuonCheckpointing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # load_state_dict logs through print_on_rank0, which needs a
        # process group even on a single rank.
        import torch.distributed as dist

        if not dist.is_initialized():
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", "29517")
            dist.init_process_group(backend="gloo", rank=0, world_size=1)
            cls._owns_group = True
        else:
            cls._owns_group = False

    @classmethod
    def tearDownClass(cls):
        import torch.distributed as dist

        if cls._owns_group and dist.is_initialized():
            dist.destroy_process_group()

    def _make(self):
        torch.manual_seed(0)
        model = _TinyTrainable()
        optimizer = BF16Optimizer(
            model,
            lr=1e-2,
            total_steps=100,
            warmup_ratio=0.0,
            lr_scheduler="constant",
            optimizer_type="muon",
        )
        return optimizer, model

    def test_roundtrip_keeps_next_step_identical(self):
        optimizer_a, model_a = self._make()
        optimizer_b, model_b = self._make()
        for step in range(3):
            _set_grads(model_a, seed=step)
            optimizer_a.step()
        optimizer_b.load_state_dict(optimizer_a.state_dict())
        _set_grads(model_a, seed=99)
        _set_grads(model_b, seed=99)
        optimizer_a.step()
        optimizer_b.step()
        for p_a, p_b in zip(model_a.parameters(), model_b.parameters()):
            self.assertTrue(torch.equal(p_a, p_b))

    def test_state_dict_carries_muon_state(self):
        optimizer, _ = self._make()
        _set_grads(optimizer.model, seed=0)
        optimizer.step()
        state = optimizer.state_dict()
        self.assertEqual(state["optimizer_type"], "muon")
        self.assertIn("muon_state_dict", state)
        muon_state = next(iter(state["muon_state_dict"]["state"].values()))
        self.assertIn("momentum_buffer", muon_state)

    def test_optimizer_type_mismatch_raises(self):
        muon_optimizer, _ = self._make()
        torch.manual_seed(0)
        adamw_model = _TinyTrainable()
        adamw_optimizer = BF16Optimizer(
            adamw_model, lr=1e-2, total_steps=100, lr_scheduler="constant"
        )
        muon_state = muon_optimizer.state_dict()
        with self.assertRaisesRegex(ValueError, "optimizer"):
            adamw_optimizer.load_state_dict(muon_state)
        with self.assertRaisesRegex(ValueError, "optimizer"):
            muon_optimizer.load_state_dict(adamw_optimizer.state_dict())

    def test_legacy_payload_without_type_key_loads_as_adamw(self):
        torch.manual_seed(0)
        model = _TinyTrainable()
        stepped = BF16Optimizer(
            model, lr=1e-2, total_steps=100, lr_scheduler="constant"
        )
        for step in range(2):
            _set_grads(model, seed=step)
            stepped.step()
        legacy = stepped.state_dict()
        legacy.pop("optimizer_type")
        torch.manual_seed(0)
        fresh = BF16Optimizer(
            _TinyTrainable(), lr=1e-2, total_steps=100, lr_scheduler="constant"
        )
        fresh.load_state_dict(legacy)  # must not raise without the discriminator

    def test_muon_payload_missing_muon_state_raises(self):
        optimizer, _ = self._make()
        _set_grads(optimizer.model, seed=0)
        optimizer.step()
        state = optimizer.state_dict()
        del state["muon_state_dict"]
        with self.assertRaisesRegex(ValueError, "muon_state_dict"):
            optimizer.load_state_dict(state)


def _training_cfg(**overrides):
    base = dict(
        learning_rate=1e-3,
        max_grad_norm=0.5,
        warmup_ratio=0.0,
        lr_scheduler="cosine",
        optimizer_cpu_offload=False,
        total_steps=100,
        max_steps=None,
        fsdp_sharding="SHARD_GRAD_OP",
        optimizer="adamw",
        muon_lr=None,
        muon_momentum=0.95,
        muon_nesterov=True,
        muon_weight_decay=0.0,
        muon_ns_steps=5,
        muon_adjust_lr="match_rms_adamw",
        muon_adamw_name_hints=[],
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class TestOptimizerFactoryGuards(unittest.TestCase):
    def _factory(self, **training_overrides):
        from specforge.training.assembly import _ConfiguredOptimizerFactory

        cfg = SimpleNamespace(training=_training_cfg(**training_overrides))
        return _ConfiguredOptimizerFactory(cfg)

    def test_muon_rejects_full_shard(self):
        factory = self._factory(fsdp_sharding="FULL_SHARD", optimizer="muon")
        with self.assertRaisesRegex(ValueError, "FULL_SHARD"):
            factory(SimpleNamespace())

    @unittest.skipUnless(TORCH_HAS_MUON, "torch>=2.9 required for torch.optim.Muon")
    def test_muon_constructs_under_shard_grad_op(self):
        factory = self._factory(optimizer="muon", muon_lr=0.02)
        optimizer = factory(_TinyTrainable())
        self.assertEqual(optimizer.optimizer_type, "muon")
        self.assertIsNotNone(optimizer.muon_optimizer)
        # the scheduler init already moved the AdamW lr off base; the Muon
        # group must mirror it at the configured ratio
        adamw_lr = optimizer.optimizer.param_groups[0]["lr"]
        self.assertAlmostEqual(
            float(optimizer.muon_optimizer.param_groups[0]["lr"]),
            float(adamw_lr) * 0.02 / 1e-3,
            places=9,
        )

    def test_adamw_ignores_full_shard_guard(self):
        factory = self._factory(fsdp_sharding="FULL_SHARD", optimizer="adamw")
        optimizer = factory(_TinyTrainable())
        self.assertIsNone(optimizer.muon_optimizer)


if __name__ == "__main__":
    unittest.main()
