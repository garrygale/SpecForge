#!/usr/bin/env python3
"""Same-weights A/B: training mask vs served (band) mask on a sliding layer.

Trains a tiny Domino draft on a synthetic long-lag copy task, then evaluates
the *same* weights under

  * the training mask (shipped served band by default; the pre-fix causal
    draft block with ``--legacy-mask``), and
  * the mask the vLLM / vLLM-Ascend draft attention actually applies for a
    non-causal sliding layer (pre_tokens=W, next_tokens=W -> symmetric band).

Only the attention mask changes between the two evaluations, so any accuracy
gap is caused by the train/serve mask mismatch rather than by the data.
The ``evaluator_mask`` column is the attention pattern SpecForge's in-process
``spec_generate`` acceptance path runs (no window at all).

Run from the SpecForge root:

    set PYTHONPATH=C:/Users/g/Desktop/codeAgents/SpecForge
    C:/Users/g/Desktop/codeAgents/phi-GNNv2/.venv/Scripts/python.exe \
        probes/domino_migration/probe_domino_window_ab.py
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile

import torch
import torch.nn.functional as F
from torch import nn

from specforge.algorithms.common.dflash_family_model import (
    OnlineDominoModel,
    create_dflash_sdpa_mask,
)
from specforge.modeling.auto import AutoDraftModel, AutoDraftModelConfig


def make_config(
    *,
    window: int,
    block_size: int,
    layers: int = 2,
    hidden: int = 128,
    vocab: int = 64,
    causal: bool = False,
) -> dict:
    return {
        "architectures": ["DominoDraftModel"],
        "model_type": "qwen3",
        "hidden_size": hidden,
        "intermediate_size": 2 * hidden,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": hidden // 4,
        "num_hidden_layers": layers,
        "layer_types": ["sliding_attention"] * layers,
        "num_target_layers": 3,
        "target_hidden_size": hidden,
        "block_size": block_size,
        "vocab_size": vocab,
        "max_position_embeddings": 4096,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1_000_000,
        "sliding_window": [window] * layers,
        "use_sliding_window": True,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "tie_word_embeddings": False,
        "use_cache": True,
        "dflash_config": {
            "mask_token_id": vocab - 1,
            "target_layer_ids": [1],
            "projector_type": "domino",
            "fusion_mode": "flare",
            "heterogeneous_kv": True,
            "pure_draft_prefix_len": 1,
            # Same field the service reads (`_domino_layer_attention`).
            "causal": causal,
            "emb_dim": 32,
            "gru_hidden_dim": 64,
            "shift_label": True,
            "use_hidden_proj": False,
            "target_hidden_size": hidden,
        },
    }


def build_draft(payload: dict) -> nn.Module:
    path = os.path.join(tempfile.mkdtemp(), "tiny_domino_window_ab.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    config = AutoDraftModelConfig.from_file(path)
    return AutoDraftModel.from_config(config, torch_dtype=torch.float32)


class LagTask:
    """x_t = perm[x_{t-lag}]: the label depends on a context token only."""

    def __init__(self, *, vocab: int, lag: int, seq_len: int, seed: int = 0):
        gen = torch.Generator().manual_seed(seed)
        self.vocab = vocab
        self.lag = lag
        self.seq_len = seq_len
        self.perm = torch.randperm(vocab, generator=gen)
        self.head = torch.randint(1, vocab - 1, (lag,), generator=gen)
        self.gen = gen

    def batch(self, batch_size: int, hidden_fn) -> tuple[torch.Tensor, torch.Tensor]:
        ids = torch.empty(batch_size, self.seq_len, dtype=torch.long)
        ids[:, : self.lag] = self.head.unsqueeze(0)
        for t in range(self.lag, self.seq_len):
            ids[:, t] = self.perm[ids[:, t - self.lag]]
        hidden = hidden_fn(ids)
        return ids, hidden


def build_hidden_fn(vocab: int, hidden: int, seed: int = 7):
    gen = torch.Generator().manual_seed(seed)
    embed = torch.randn(vocab, hidden, generator=gen) * 0.5
    proj = torch.randn(hidden, hidden, generator=gen) / hidden**0.5

    def hidden_fn(ids: torch.Tensor) -> torch.Tensor:
        return torch.tanh(embed[ids] @ proj)

    return hidden_fn


def band_mask(
    *,
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    seq_len: int,
    block_size: int,
    window: int,
) -> torch.Tensor:
    """Served (FIA/CUDA) mask: context lower bound + whole-block visibility."""
    bsz, num_blocks = anchor_positions.shape
    q_len = num_blocks * block_size
    kv_len = seq_len + num_blocks * block_size
    q_idx = torch.arange(q_len).view(1, 1, -1, 1)
    kv_idx = torch.arange(kv_len).view(1, 1, 1, -1)
    q_block = q_idx // block_size
    q_off = q_idx % block_size
    anchor = anchor_positions.view(bsz, 1, num_blocks, 1).repeat_interleave(
        block_size, dim=2
    )
    q_pos = anchor + q_off

    is_context = kv_idx < seq_len
    mask_context = is_context & (kv_idx < anchor) & (kv_idx >= q_pos - (window - 1))
    is_draft = kv_idx >= seq_len
    kv_block = (kv_idx - seq_len) // block_size
    kv_offset = (kv_idx - seq_len) % block_size
    kv_pos = anchor + kv_offset
    # Band on both sides inside the own block (matching pre=W, next=W).
    mask_draft = (
        is_draft
        & (q_block == kv_block)
        & (kv_pos >= q_pos - (window - 1))
        & (kv_pos <= q_pos + window)
    )
    valid = block_keep_mask.view(bsz, 1, num_blocks, 1).repeat_interleave(
        block_size, dim=2
    )
    return (mask_context | mask_draft) & valid


def evaluator_mask(
    *,
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    seq_len: int,
    block_size: int,
) -> torch.Tensor:
    """Mask implicit in SpecForge's in-process ``spec_generate`` evaluator.

    The evaluator passes ``attention_mask=None`` and HF's sdpa/eager attention
    ignores ``sliding_window``; the draft's own cache only holds positions
    ``< anchor`` (``past_key_values_draft.crop(start)``), so queries see the
    *entire* context and the *entire* own block, but no future target taps.
    """
    bsz, num_blocks = anchor_positions.shape
    q_len = num_blocks * block_size
    kv_len = seq_len + num_blocks * block_size
    q_idx = torch.arange(q_len).view(1, 1, -1, 1)
    kv_idx = torch.arange(kv_len).view(1, 1, 1, -1)
    q_block = q_idx // block_size
    anchor = anchor_positions.view(bsz, 1, num_blocks, 1).repeat_interleave(
        block_size, dim=2
    )

    mask_context = (kv_idx < seq_len) & (kv_idx < anchor)
    is_draft = kv_idx >= seq_len
    kv_block = (kv_idx - seq_len) // block_size
    mask_draft = is_draft & (q_block == kv_block)
    valid = block_keep_mask.view(bsz, 1, num_blocks, 1).repeat_interleave(
        block_size, dim=2
    )
    return (mask_context | mask_draft) & valid


@torch.no_grad()
def evaluate(
    *,
    trainer: OnlineDominoModel,
    draft: nn.Module,
    lm_head: nn.Module,
    embed: nn.Module,
    task: LagTask,
    hidden_fn,
    block_size: int,
    window: int,
    num_anchors: int,
    batches: int = 4,
    batch_size: int = 8,
    seed: int = 1234,
) -> dict[str, float]:
    torch.manual_seed(seed)
    draft.eval()
    names = ("train_mask", "band_mask", "no_mask", "evaluator_mask")
    acc = {name: 0.0 for name in names}
    base_acc = {name: 0.0 for name in names}
    per_offset_hit = {name: torch.zeros(block_size) for name in acc}
    per_offset_total = torch.zeros(block_size)
    predictions: dict[str, list[torch.Tensor]] = {name: [] for name in acc}
    weights: list[torch.Tensor] = []
    total = 0
    for _ in range(batches):
        ids, hidden = task.batch(batch_size, hidden_fn)
        seq_len = ids.shape[1]
        device = ids.device
        anchors, keep = trainer._sample_anchor_positions(
            seq_len, torch.ones_like(ids, dtype=torch.float), device
        )
        noise = trainer._create_noise_embed(ids, anchors, keep)
        position_ids = torch.cat(
            [
                torch.arange(seq_len, device=device).unsqueeze(0).expand(
                    batch_size, -1
                ),
                trainer._create_position_ids(anchors),
            ],
            dim=1,
        )
        train_mask = create_dflash_sdpa_mask(
            anchor_positions=anchors,
            block_keep_mask=keep,
            S=seq_len,
            block_size=block_size,
            device=device,
            sliding_window=window,
        )
        served = band_mask(
            anchor_positions=anchors,
            block_keep_mask=keep,
            seq_len=seq_len,
            block_size=block_size,
            window=window,
        )
        masks = {
            "train_mask": train_mask,
            "band_mask": served,
            "no_mask": None,
            "evaluator_mask": evaluator_mask(
                anchor_positions=anchors,
                block_keep_mask=keep,
                seq_len=seq_len,
                block_size=block_size,
            ),
        }

        # Labels (Domino, shift_label=True).
        offsets = torch.arange(1, 1 + block_size, device=device).view(1, 1, -1)
        label_idx = anchors.unsqueeze(-1) + offsets
        valid_label = label_idx < seq_len
        safe_idx = label_idx.clamp(max=seq_len - 1)
        gather_ids = ids.unsqueeze(1).expand(-1, anchors.shape[1], -1)
        target_ids = torch.gather(gather_ids, 2, safe_idx)
        prev_ids = torch.gather(
            gather_ids,
            2,
            (anchors.unsqueeze(-1) + offsets - 1).clamp(max=seq_len - 1),
        )
        weight = keep.unsqueeze(-1).float() * valid_label.float()
        weight[:, :, 0] = 0.0  # offset 0 is the pure-draft-prefix position

        for name, mask in masks.items():
            out = draft(
                position_ids=position_ids,
                noise_embedding=noise,
                target_hidden=hidden,
                attention_mask=mask,
            )
            hidden4d = out.reshape(
                batch_size, anchors.shape[1], block_size, out.shape[-1]
            )
            base_logits = lm_head(
                hidden4d.reshape(batch_size, -1, hidden4d.shape[-1])
            ).reshape(batch_size, anchors.shape[1], block_size, -1)
            correction = draft.compute_correction_logits(
                hidden_states=hidden4d,
                prev_token_embeddings=embed(prev_ids),
            )
            suffix = draft.suffix_start
            final_logits = torch.cat(
                [
                    base_logits[:, :, :suffix, :],
                    base_logits[:, :, suffix:, :] + correction,
                ],
                dim=2,
            )
            for key, logits in (("", final_logits), ("base_", base_logits)):
                pred = logits.argmax(dim=-1)
                hit = ((pred == target_ids) & (weight > 0.5)).sum().item()
                denom = (weight > 0.5).sum().item()
                if key == "":
                    acc[name] += hit
                    predictions[name].append(pred)
                    per_offset_hit[name] += (
                        (pred == target_ids) & (weight > 0.5)
                    ).sum(dim=(0, 1)).float()
                else:
                    base_acc[name] += hit
        weights.append(weight > 0.5)
        per_offset_total += weights[-1].sum(dim=(0, 1)).float()
        total += (weight > 0.5).sum().item()
    draft.train()
    agreement = {}
    for name in ("band_mask", "no_mask", "evaluator_mask"):
        matches = 0
        denom = 0
        for base_pred, other_pred, mask in zip(
            predictions["train_mask"], predictions[name], weights
        ):
            matches += int(((base_pred == other_pred) & mask).sum().item())
            denom += int(mask.sum().item())
        agreement[name] = matches / max(1, denom)
    return {
        **{f"final/{k}": v / max(1, total) for k, v in acc.items()},
        **{f"base/{k}": v / max(1, total) for k, v in base_acc.items()},
        **{f"agree_with_train_mask/{k}": v for k, v in agreement.items()},
        "per_offset/acc_train_mask": per_offset_hit["train_mask"]
        / per_offset_total.clamp_min(1),
        "per_offset/acc_band_mask": per_offset_hit["band_mask"]
        / per_offset_total.clamp_min(1),
    }


def run(
    *,
    window: int,
    block_size: int,
    steps: int,
    batch_size: int,
    seq_len: int,
    lag: int,
    num_anchors: int,
    seed: int,
    causal: bool = False,
) -> dict[str, float]:
    torch.manual_seed(seed)
    payload = make_config(
        window=window,
        block_size=block_size,
        causal=causal,
    )
    vocab = payload["vocab_size"]
    hidden_fn = build_hidden_fn(vocab, payload["target_hidden_size"])
    task = LagTask(vocab=vocab, lag=lag, seq_len=seq_len, seed=seed)
    draft = build_draft(payload)
    lm_head = nn.Linear(payload["target_hidden_size"], vocab, bias=False)
    embed = nn.Embedding(vocab, payload["target_hidden_size"])
    trainer = OnlineDominoModel(
        draft_model=draft,
        target_lm_head=lm_head,
        target_embed_tokens=embed,
        mask_token_id=payload["dflash_config"]["mask_token_id"],
        block_size=block_size,
        attention_backend="sdpa",
        num_anchors=num_anchors,
        shift_label=True,
    )
    params = list(draft.parameters()) + list(lm_head.parameters()) + list(
        embed.parameters()
    )
    opt = torch.optim.Adam(params, lr=3e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=3e-3, total_steps=steps, pct_start=0.1
    )
    torch.manual_seed(seed + 1)
    for step in range(steps):
        ids, hidden = task.batch(batch_size, hidden_fn)
        loss, acc, _ = trainer(ids, hidden, torch.ones_like(ids, dtype=torch.float))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if (step + 1) % max(1, steps // 4) == 0:
            print(
                f"    step {step + 1:>4}: loss={loss.item():.4f} "
                f"acc={acc.item():.4f}"
            )
    metrics = evaluate(
        trainer=trainer,
        draft=draft,
        lm_head=lm_head,
        embed=embed,
        task=task,
        hidden_fn=hidden_fn,
        block_size=block_size,
        window=window,
        num_anchors=num_anchors,
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=192)
    parser.add_argument("--lag", type=int, default=12)
    parser.add_argument("--num-anchors", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--block-sweep", action="store_true")
    parser.add_argument(
        "--legacy-mask",
        action="store_true",
        help="train with the pre-fix causal draft block instead of the shipped band",
    )
    args = parser.parse_args()

    runs = [(8, 8), (16, 8), (32, 8), (64, 8)] if args.sweep else []
    if args.block_sweep:
        runs += [(16, 16), (64, 16)]
    if not runs:
        runs = [(args.window, args.block_size)]
    for window, block_size in runs:
        if window < block_size:
            continue
        print(
            f"\n=== window={window}, block_size={block_size}, "
            f"lag={args.lag} (label depends on a context token {args.lag} back), "
            f"training mask="
            f"{'legacy causal block' if args.legacy_mask else 'served band'} ==="
        )
        metrics = run(
            window=window,
            block_size=block_size,
            steps=args.steps,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            lag=args.lag,
            num_anchors=args.num_anchors,
            seed=args.seed,
            causal=args.legacy_mask,
        )
        for key in ("final/train_mask", "final/band_mask", "final/no_mask",
                    "final/evaluator_mask",
                    "base/train_mask", "base/band_mask",
                    "base/evaluator_mask",
                    "agree_with_train_mask/band_mask",
                    "agree_with_train_mask/no_mask",
                    "agree_with_train_mask/evaluator_mask"):
            print(f"    {key:>34}: {metrics[key]:.4f}")
        print(
            "    per-offset acc (train mask): "
            + " ".join(f"{float(v):.2f}" for v in metrics["per_offset/acc_train_mask"])
        )
        print(
            "    per-offset acc (band mask) : "
            + " ".join(f"{float(v):.2f}" for v in metrics["per_offset/acc_band_mask"])
        )


if __name__ == "__main__":
    main()
