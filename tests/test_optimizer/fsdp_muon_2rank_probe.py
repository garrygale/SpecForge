"""2-rank FSDP regression probe for the muon param split.

Launched by test_muon.py via torchrun with 2 gloo ranks. Reproduces the
Trainer wiring (wrap first, then the optimizer factory on model.draft_model):

* SHARD_GRAD_OP: the draft's params reach the factory as 1D flat-shard views
  (use_orig_params=True), which is why the factory must fail closed on every
  sharding strategy that shards parameters.
* NO_SHARD (DDP): params stay 2D and the real factory builds a non-empty
  Muon group whose masters are all 2D.
"""

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
from torch import nn


class _Block(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.norm = nn.Parameter(torch.ones(hidden))

    def forward(self, x):
        return self.q_proj(x) + self.norm


class _Draft(nn.Module):
    _no_split_modules = ["_Block"]

    def __init__(self, hidden, vocab):
        super().__init__()
        self.layers = nn.ModuleList([_Block(hidden) for _ in range(2)])
        self.head = nn.Linear(hidden, vocab, bias=False)
        # real drafts are PreTrainedModels; the vocab-size guard reads this
        self.config = SimpleNamespace(vocab_size=vocab)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


class _Composite(nn.Module):
    _no_split_modules = ["_Block"]

    def __init__(self, hidden, vocab, with_head):
        super().__init__()
        draft = _Draft(hidden, vocab)
        if not with_head:
            draft.head = nn.Linear(hidden, hidden, bias=False)
        self.draft_model = draft

    def forward(self, x):
        return self.draft_model(x)


def _training_cfg(**overrides):
    base = dict(
        learning_rate=1e-3,
        max_grad_norm=0.5,
        warmup_ratio=0.0,
        lr_scheduler="cosine",
        optimizer_cpu_offload=False,
        total_steps=10,
        max_steps=None,
        fsdp_sharding="NO_SHARD",
        optimizer="muon",
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


def _shard_check(rank):
    """SHARD_GRAD_OP with no factory: every trainable param on the factory
    target must be a 1D flat-shard view — the reason the factory guard
    rejects every param-sharding strategy."""
    from specforge.training.backend import FSDPTrainingBackend, ParallelConfig

    backend = FSDPTrainingBackend(
        ParallelConfig(sharding_strategy="SHARD_GRAD_OP"), optimizer_factory=None
    )
    composite = _Composite(8, 96, with_head=False)
    backend.prepare_model(composite, optimizer_target=composite.draft_model)
    shapes = [
        tuple(p.shape)
        for _, p in composite.draft_model.named_parameters()
        if p.requires_grad
    ]
    all_1d = all(len(shape) == 1 for shape in shapes)
    if rank == 0:
        print(f"PROBE shard_grad_op_all_params_1d={all_1d}", flush=True)
        print(f"PROBE shard_grad_op_shapes={shapes}", flush=True)
    if not all_1d:
        raise AssertionError(
            "SHARD_GRAD_OP params are no longer 1D shard views; the muon "
            "factory guard premise needs re-review"
        )


def _no_shard_check(rank):
    """NO_SHARD with the real factory: Muon group non-empty, masters 2D."""
    from specforge.training.assembly import _ConfiguredOptimizerFactory
    from specforge.training.backend import FSDPTrainingBackend, ParallelConfig

    cfg = SimpleNamespace(training=_training_cfg())
    backend = FSDPTrainingBackend(
        ParallelConfig(sharding_strategy="NO_SHARD"),
        optimizer_factory=_ConfiguredOptimizerFactory(cfg),
    )
    composite = _Composite(8, 96, with_head=True)
    backend.prepare_model(composite, optimizer_target=composite.draft_model)
    optimizer = backend.optimizer
    muon_group = (
        optimizer.muon_optimizer.param_groups[0]["params"]
        if optimizer.muon_optimizer is not None
        else []
    )
    masters_2d = all(p.ndim == 2 for p in muon_group)
    if rank == 0:
        print(f"PROBE no_shard_muon_params={len(muon_group)}", flush=True)
        print(f"PROBE no_shard_masters_2d={masters_2d}", flush=True)
    assert optimizer.muon_optimizer is not None, "NO_SHARD built an empty Muon group"
    # the two q_proj matrices; the vocab head stays on AdamW via the guard
    assert len(muon_group) == 2, f"expected 2 muon masters, got {len(muon_group)}"
    assert masters_2d


def main():
    import torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29755")

    dist.init_process_group("gloo")
    rank = dist.get_rank()
    try:
        _shard_check(rank)
        dist.barrier()
    finally:
        dist.destroy_process_group()

    dist.init_process_group("gloo")
    try:
        _no_shard_check(rank)
        dist.barrier()
    finally:
        dist.destroy_process_group()
    if rank == 0:
        print("PROBE OK", flush=True)


if __name__ == "__main__":
    main()
