#!/usr/bin/env python3
"""Stage-0 probe: decompose DFlash-family draft attention mass by KV origin.

Replays the training forward (sdpa mask builder + eager attention, weights
captured per layer via forward hooks) over target hidden states:

  H1  anchor column absorbs most within-block mass (emergent sink)
  H2  mask-column mass is positional smoothing, not content addressing
  H3  left edge of the visible context band absorbs excess mass
  H4  diffuse context attention (high entropy) predicts argmax errors,
      concentrated at block suffix offsets
  H5  rows whose band reaches position 0 (anchor < W) lean less on
      anchor/edge than rows with anchor >> W

Two data sources:

* HF capture (no sglang needed): load the target with transformers, run the
  prompts jsonl through it, and take hidden_states[target_layer_ids + 1].
  The +1 is the same convention the sglang capture patch uses ("capture
  before layer i" -> HF-style output of layer k captured at k+1), so the
  layers are identical to what online disaggregated training consumed.

    python probes/attention_sink/probe_attn_mass.py \
        --draft-checkpoint /path/to/draft_ckpt \
        --target-model-path /path/to/target_hf_dir \
        --prompts /path/to/train.jsonl --chat-template qwen3.5 \
        --prompt-max-length 8192 --out attn_mass_out

* Offline shards: a directory of precomputed .ckpt feature files, as produced
  by scripts/prepare_hidden_states.py and consumed by offline training.

    python probes/attention_sink/probe_attn_mass.py \
        --draft-checkpoint /path/to/draft_ckpt \
        --target-checkpoint /path/to/target_hf_dir \
        --data /path/to/capture_shards --out attn_mass_out

Local smoke test (no data needed, exercises both modes):

  python probes/attention_sink/probe_attn_mass.py --selftest
"""

from __future__ import annotations

import argparse
import glob
import gzip
import io
import json
import os
import sys
import tempfile
from collections import defaultdict

import numpy as np
import torch
from torch import nn

from specforge.algorithms.common.dflash_family_model import (
    OnlineDFlashModel,
    create_dflash_sdpa_mask,
)
from specforge.algorithms.common.hidden_states_data import normalize_offline_sample
from specforge.modeling.auto import AutoDraftModel


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def pick_device(name: str) -> torch.device:
    order = [name] if name != "auto" else ["npu", "cuda", "cpu"]
    for cand in order:
        if cand == "npu" and hasattr(torch, "npu") and torch.npu.is_available():
            return torch.device("npu")
        if cand == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if cand == "cpu":
            return torch.device("cpu")
    raise RuntimeError(f"no usable device for {name!r}")


def load_target_matrices(target_dir: str, vocab: int):
    """Partial-load only embed_tokens and lm_head from a target HF checkpoint."""
    keys = {}
    index = os.path.join(target_dir, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index) as f:
            shards = sorted(set(json.load(f)["weight_map"].values()))
        shards = [os.path.join(target_dir, s) for s in shards]
    else:
        shards = sorted(glob.glob(os.path.join(target_dir, "*.safetensors")))
    if shards:
        from safetensors.torch import safe_open

        for shard in shards:
            with safe_open(shard, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.endswith(
                        ("embed_tokens.weight", "lm_head.weight")
                    ):
                        keys[key] = f.get_tensor(key)
    if not keys:
        for bin_path in sorted(
            glob.glob(os.path.join(target_dir, "pytorch_model*.bin"))
        ):
            state = torch.load(bin_path, map_location="cpu", weights_only=False)
            for key, tensor in state.items():
                if key.endswith(("embed_tokens.weight", "lm_head.weight")):
                    keys[key] = tensor
            del state
    embed = [t for k, t in keys.items() if "embed_tokens" in k]
    head = [t for k, t in keys.items() if "lm_head" in k]
    if not embed:
        raise RuntimeError(f"no embed_tokens weight found under {target_dir}")
    embed_w, head_w = embed[0], (head[0] if head else embed[0])
    if embed_w.shape[0] != vocab or head_w.shape[0] != vocab:
        raise RuntimeError(
            f"target vocab mismatch: embed {tuple(embed_w.shape)}, "
            f"lm_head {tuple(head_w.shape)}, draft config vocab {vocab}"
        )
    return embed_w, head_w


def find_shards(data_path: str):
    suffixes = (".ckpt", ".pt", ".ckpt.gz", ".pt.gz")
    if os.path.isfile(data_path):
        return [data_path]
    files = []
    for root, _dirs, names in os.walk(data_path):
        for name in names:
            if name.endswith(suffixes):
                files.append(os.path.join(root, name))
    return sorted(files)


def load_shard(path: str):
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as f:
            return torch.load(io.BytesIO(f.read()), weights_only=False)
    return torch.load(path, weights_only=False, mmap=True)


# ---------------------------------------------------------------------------
# HF target capture (bypasses sglang entirely)
# ---------------------------------------------------------------------------


def capture_samples_hf(args, draft):
    """Run the prompts jsonl through the HF target and concat capture layers.

    The draft config's target_layer_ids are HF-style 0-based decoder-layer
    output ids; hidden_states[k + 1] matches the sglang capture convention.
    Returns materialized samples on CPU so the target can be freed before the
    draft replay starts.
    """
    from specforge.data.prompt_builder import prepare_prompt_tasks

    layer_ids = draft.target_layer_ids
    if layer_ids and isinstance(layer_ids[0], (list, tuple)):
        raise RuntimeError(
            "per-layer (nested) target_layer_ids are not supported by the HF "
            "capture path; flatten them or precompute .ckpt shards"
        )

    tokenizer = None
    if args.chat_template is not None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.target_model_path, trust_remote_code=True
        )
    prompts = prepare_prompt_tasks(
        args.prompts,
        tokenizer,
        chat_template=args.chat_template,
        max_length=args.prompt_max_length,
        is_preformatted=False,
        train_only_last_turn=False,
        cache_dir=None,
        cache_key=None,
        num_proc=None,
        min_loss_tokens=2,
        max_prompts=args.max_samples,
    )
    prompts = list(prompts)[: args.max_samples]
    if not prompts:
        raise RuntimeError(f"no usable prompts in {args.prompts}")

    from transformers import AutoModelForCausalLM

    print(
        f"[probe] HF capture: {len(prompts)} prompts, target="
        f"{args.target_model_path}, layers={[k + 1 for k in layer_ids]}"
    )
    try:
        target = AutoModelForCausalLM.from_pretrained(
            args.target_model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )
    except Exception as exc:  # no accelerate / single-device envs
        print(f"[probe] device_map=auto failed ({exc}); falling back to .to(device)")
        target = AutoModelForCausalLM.from_pretrained(
            args.target_model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(pick_device(args.device))
    target.eval()
    num_target_layers = target.config.num_hidden_layers
    if max(layer_ids) >= num_target_layers:
        raise RuntimeError(
            f"target_layer_ids {layer_ids} exceed target depth "
            f"{num_target_layers}"
        )

    samples = []
    with torch.inference_mode():
        for idx, prompt in enumerate(prompts):
            input_ids = torch.tensor(
                prompt["payload"]["input_ids"], dtype=torch.long
            ).unsqueeze(0)
            loss_mask = torch.tensor(
                prompt["payload"]["loss_mask"], dtype=torch.float
            ).unsqueeze(0)
            length = input_ids.shape[1]
            out = target(
                input_ids.to(target.device), output_hidden_states=True, use_cache=False
            )
            selected = [out.hidden_states[k + 1] for k in layer_ids]
            hidden = torch.cat(selected, dim=-1)[
                :, : args.max_length
            ].to("cpu", torch.bfloat16)
            input_ids = input_ids[:, : args.max_length]
            loss_mask = loss_mask[:, : args.max_length]
            del out, selected
            if hidden.shape[1] != input_ids.shape[1]:
                raise RuntimeError("prompt truncation mismatch")
            print(
                f"[probe] captured prompt {idx}: S={input_ids.shape[1]}, "
                f"loss_tokens={int(loss_mask.sum())}, width={hidden.shape[-1]}"
            )
            samples.append(
                (
                    f"prompt_{idx:04d}",
                    {"input_ids": input_ids, "loss_mask": loss_mask,
                     "hidden_states": hidden},
                )
            )
    del target
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.empty_cache()
    return samples


# ---------------------------------------------------------------------------
# forward replay (mirrors OnlineDFlashModel._forward_draft_blocks with fixed
# anchors and eager attention; weights are captured by hooks because the
# decoder layer drops them)
# ---------------------------------------------------------------------------


class AttnCapture:
    def __init__(self, draft):
        self.per_layer = [[] for _ in draft.layers]
        self.handles = [
            layer.self_attn.register_forward_hook(self._make(idx))
            for idx, layer in enumerate(draft.layers)
        ]

    def _make(self, idx):
        def hook(_module, _args, output):
            weights = output[1]
            if weights is not None:
                self.per_layer[idx].append(weights.detach())

        return hook

    def clear(self):
        for buf in self.per_layer:
            buf.clear()

    def remove(self):
        for handle in self.handles:
            handle.remove()


def _chunk_inputs(wrapper, input_ids, hidden_states, anchors, seq_len, device):
    keep = torch.ones_like(anchors, dtype=torch.bool)
    noise_embedding = wrapper._create_noise_embed(input_ids, anchors, keep)
    ctx_ids = (
        torch.arange(seq_len, device=device).unsqueeze(0).expand(input_ids.shape[0], -1)
    )
    position_ids = torch.cat([ctx_ids, wrapper._create_position_ids(anchors)], dim=1)
    masks = build_layer_masks(wrapper, anchors, keep, seq_len, device)
    return noise_embedding, position_ids, masks


def build_layer_masks(wrapper, anchors, keep, seq_len, device):
    block_size = wrapper.block_size
    args = {
        "anchor_positions": anchors,
        "block_keep_mask": keep,
        "S": seq_len,
        "block_size": block_size,
        "device": device,
        "num_sinks": getattr(wrapper.draft_model, "num_sinks", 0),
    }
    full = create_dflash_sdpa_mask(**args)
    windows = getattr(wrapper.draft_model, "layer_sliding_windows", None)
    if windows is not None:
        return [
            (
                full
                if w is None
                else create_dflash_sdpa_mask(
                    **args,
                    sliding_window=w,
                    sliding_draft_causal=wrapper.sliding_draft_causal,
                )
            )
            for w in windows
        ]
    window = wrapper.draft_model.sliding_window
    if window is not None:
        return create_dflash_sdpa_mask(
            **args,
            sliding_window=window,
            sliding_draft_causal=wrapper.sliding_draft_causal,
        )
    return full


def _forward_with_impl(wrapper, impl, input_ids, hidden_states, anchors, seq_len,
                       device):
    noise_embedding, position_ids, masks = _chunk_inputs(
        wrapper, input_ids, hidden_states, anchors, seq_len, device
    )
    draft = wrapper.draft_model
    prev = draft.config._attn_implementation
    draft.config._attn_implementation = impl
    try:
        hidden = draft(
            position_ids=position_ids,
            noise_embedding=noise_embedding,
            target_hidden=hidden_states,
            attention_mask=masks,
        )
    finally:
        draft.config._attn_implementation = prev
    masks_list = masks if isinstance(masks, list) else [masks] * len(draft.layers)
    return hidden, masks_list


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------


class LayerStats:
    KEYS = (
        "p_block",
        "p_anchor",
        "p_edge",
        "edge_ratio",
        "h_ctx",
        "support",
        "p_sink",
    )

    def __init__(self):
        self.rows = []  # dict of [H, Q] tensors per chunk
        self.err = []
        self.offsets = []
        self.reach_pos0 = []
        self.decay = defaultdict(float)
        self.decay_n = defaultdict(int)
        self.top1 = defaultdict(int)

    def add(self, metrics, err, offsets, reach, decay_pairs, top1_class):
        self.rows.append(metrics)
        self.err.append(err)
        self.offsets.append(offsets)
        self.reach_pos0.append(reach)
        for d, mass in decay_pairs:
            self.decay[d] += mass
            self.decay_n[d] += 1
        for cls, count in top1_class.items():
            self.top1[cls] += count

    def cat(self, key):
        return torch.cat([r[key].flatten() for r in self.rows])


def chunk_metrics(attn, mask, anchors, seq_len, block_size, window, edge_k):
    """attn [H, Q, KV] float32, mask [Q, KV] bool -> metrics dict of [H, Q]."""
    heads, q_len, kv_len = attn.shape
    device = attn.device
    kv = torch.arange(kv_len, device=device)
    q_idx = torch.arange(q_len, device=device)
    block_id = q_idx // block_size
    q_off = q_idx % block_size
    anchor_rows = anchors[0, block_id]
    vis = mask.unsqueeze(0).float()  # [1, Q, KV]
    w = attn * vis

    blk_start = seq_len + block_id * block_size
    own_block = (kv.unsqueeze(0) >= blk_start.unsqueeze(1)) & (
        kv.unsqueeze(0) < (blk_start + block_size).unsqueeze(1)
    )  # [Q, KV]
    p_block = (w * own_block.unsqueeze(0)).sum(-1)
    p_anchor = w[:, q_idx, blk_start]

    is_ctx = kv < seq_len
    ctx_mask = vis * is_ctx.unsqueeze(0).float()  # [1, Q, KV]
    ctx_count = ctx_mask.sum(-1)  # [1, Q]

    lo = (
        (anchor_rows + q_off - (window - 1)).clamp(min=0)
        if window is not None
        else torch.zeros_like(anchor_rows)
    )
    edge = (
        (kv.unsqueeze(0) >= lo.unsqueeze(1))
        & (kv.unsqueeze(0) < (lo + edge_k).unsqueeze(1))
        & is_ctx.unsqueeze(0)
    )  # [Q, KV]
    p_edge = (w * edge.unsqueeze(0)).sum(-1)
    edge_ratio = p_edge * ctx_count / float(edge_k)

    p_ctx = w * ctx_mask
    p_norm = p_ctx / p_ctx.sum(-1).clamp_min(1e-9).unsqueeze(-1)
    h_ctx = -(p_norm * torch.nan_to_num(p_norm.log())).sum(-1)

    p_sink = w[..., seq_len + anchors.shape[1] * block_size :].sum(-1)

    return {
        "p_block": p_block,
        "p_anchor": p_anchor,
        "p_edge": p_edge,
        "edge_ratio": edge_ratio,
        "h_ctx": h_ctx,
        "support": h_ctx.exp(),
        "p_sink": p_sink,
    }


def top1_classes(attn, mask, anchors, seq_len, block_size, window, edge_k):
    heads, q_len, kv_len = attn.shape
    device = attn.device
    kv = torch.arange(kv_len, device=device)
    q_idx = torch.arange(q_len, device=device)
    block_id = q_idx // block_size
    q_off = q_idx % block_size
    anchor_rows = anchors[0, block_id]
    masked = attn * mask.unsqueeze(0).float()
    top = masked.argmax(-1)  # [H, Q]

    blk_start = seq_len + block_id * block_size
    is_anchor = top == blk_start.unsqueeze(0)
    in_block = (top >= blk_start.unsqueeze(0)) & (
        top < (blk_start + block_size).unsqueeze(0)
    )
    lo = (
        (anchor_rows + q_off - (window - 1)).clamp(min=0)
        if window is not None
        else torch.zeros_like(anchor_rows)
    )
    is_edge = (top < seq_len) & (top >= lo.unsqueeze(0)) & (
        top < (lo + edge_k).unsqueeze(0)
    )
    classes = {}
    for name, sel in (
        ("anchor", is_anchor),
        ("block", in_block & ~is_anchor),
        ("edge", is_edge),
        ("mid", (top < seq_len) & ~is_edge),
    ):
        classes[name] = int(sel.sum())
    return classes


def block_decay(attn, seq_len, block_size):
    """Mean within-block normalized mass on mask columns by |o - o'|."""
    heads, q_len, kv_len = attn.shape
    device = attn.device
    q_idx = torch.arange(q_len, device=device)
    kv = torch.arange(kv_len, device=device)
    block_id = q_idx // block_size
    q_off = q_idx % block_size
    blk_start = seq_len + block_id * block_size
    own = (kv.unsqueeze(0) >= blk_start.unsqueeze(1)) & (
        kv.unsqueeze(0) < (blk_start + block_size).unsqueeze(1)
    )
    p_block = (attn * own.unsqueeze(0)).sum(-1).clamp_min(1e-9)  # [H, Q]
    sums = defaultdict(float)
    counts = defaultdict(int)
    for o_prime in range(1, block_size):
        col = blk_start + o_prime
        mass = attn[:, q_idx, col] / p_block
        dist = (q_off - o_prime).abs()
        for d in range(block_size):
            sel = dist == d
            if sel.any():
                sums[d] += float(mass[:, sel].sum())
                counts[d] += int(mass[:, sel].numel())
    return [(d, sums[d] / max(counts[d], 1)) for d in sorted(sums)]


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def capture_width(draft) -> int:
    if getattr(draft, "_flare", False):
        return draft.num_target_layers * draft.target_hidden_size
    if getattr(draft, "_per_layer", False):
        unique = {lid for sub in draft.target_layer_ids for lid in sub}
        return len(unique) * draft.target_hidden_size
    return draft.fc.in_features


def select_anchors(loss_mask, seq_len, block_size, max_anchors):
    limit = seq_len - block_size
    if limit <= 0:
        return torch.empty(0, dtype=torch.long)
    ok = (loss_mask[0, :limit] > 0.5) & (loss_mask[0, 1 : limit + 1] > 0.5)
    cand = ok.nonzero().flatten()
    if cand.numel() > max_anchors:
        stride = cand.numel() / max_anchors
        idx = (torch.arange(max_anchors, dtype=torch.float) * stride).long()
        cand = cand[idx.clamp(max=cand.numel() - 1).unique()]
    return cand


def run_probe(args):
    device = pick_device(args.device)
    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    torch.manual_seed(args.seed)

    print(f"[probe] device={device} dtype={dtype}")
    draft = AutoDraftModel.from_pretrained(args.draft_checkpoint, torch_dtype=dtype)
    draft = draft.to(device).eval()
    config = draft.config
    block_size = draft.block_size
    num_layers = config.num_hidden_layers
    windows = getattr(draft, "layer_sliding_windows", None)
    if windows is None:
        window = getattr(draft, "sliding_window", None)
        windows = [window if lt == "sliding_attention" else None
                   for lt in config.layer_types]
    shift_label = bool(getattr(draft, "shift_label", False))
    print(
        f"[probe] draft layers={num_layers} block={block_size} "
        f"windows={list(windows)} shift_label={shift_label}"
    )

    embed_w, head_w = load_target_matrices(args.target_checkpoint, config.vocab_size)
    embed = nn.Embedding.from_pretrained(embed_w, freeze=True).to(device, dtype)
    lm_head = nn.Linear(head_w.shape[1], head_w.shape[0], bias=False)
    with torch.no_grad():
        lm_head.weight.copy_(head_w)
    lm_head = lm_head.to(device, dtype).eval()

    wrapper = OnlineDFlashModel(
        draft_model=draft,
        target_lm_head=lm_head,
        target_embed_tokens=embed,
        mask_token_id=config.dflash_config["mask_token_id"],
        block_size=block_size,
        attention_backend="sdpa",
        num_anchors=args.max_anchors,
    )

    if getattr(args, "hf_mode", False):
        samples = capture_samples_hf(args, draft)
    else:
        shards = find_shards(args.data)
        if not shards:
            raise RuntimeError(f"no capture shards under {args.data}")
        print(f"[probe] {len(shards)} shards, using up to {args.max_samples}")
        samples = [
            (os.path.basename(s), load_shard(s)) for s in shards[: args.max_samples]
        ]

    capture = AttnCapture(draft)
    stats = [LayerStats() for _ in range(num_layers)]
    parity_done = False
    n_samples = 0

    with torch.inference_mode():
        for name, raw in samples:
            if getattr(args, "hf_mode", False):
                sample = raw
            else:
                sample = normalize_offline_sample(raw, args.max_length)
            input_ids = sample["input_ids"].to(device)
            hidden_states = sample["hidden_states"].to(device, dtype)
            seq_len = input_ids.shape[1]
            expected = capture_width(draft)
            if hidden_states.shape[-1] != expected:
                raise RuntimeError(
                    f"capture width {hidden_states.shape[-1]} != expected "
                    f"{expected} from the draft config"
                )

            anchors_all = select_anchors(
                sample["loss_mask"], seq_len, block_size, args.max_anchors
            ).to(device)
            if anchors_all.numel() == 0:
                print(f"[probe] {name}: no valid anchors, skipped")
                continue
            n_samples += 1
            print(
                f"[probe] {name}: S={seq_len}, anchors={anchors_all.numel()}"
            )

            for start in range(0, anchors_all.numel(), args.chunk_blocks):
                anchors = anchors_all[start : start + args.chunk_blocks].unsqueeze(0)
                hidden, masks = _forward_with_impl(
                    wrapper, "eager", input_ids, hidden_states, anchors,
                    seq_len, device,
                )
                captured = [buf[0] if buf else None for buf in capture.per_layer]
                capture.clear()
                if not parity_done:
                    ref, _ = _forward_with_impl(
                        wrapper, "sdpa", input_ids, hidden_states, anchors,
                        seq_len, device,
                    )
                    diff = (hidden.float() - ref.float()).abs().max().item()
                    rel = diff / ref.float().abs().max().clamp_min(1e-9).item()
                    print(
                        f"[probe] eager-vs-sdpa parity (first chunk): "
                        f"max|dH|={diff:.3e} rel={rel:.3e}"
                    )
                    parity_done = True

                q_len = anchors.numel() * block_size
                q_idx = torch.arange(q_len, device=device)
                block_id = q_idx // block_size
                q_off = q_idx % block_size

                label_off = torch.arange(block_size, device=device) + (
                    1 if shift_label else 0
                )
                label_idx = anchors[0, block_id] + label_off[q_off]
                label_ok = label_idx < seq_len
                targets = input_ids[0][label_idx.clamp(max=seq_len - 1)]
                preds = lm_head(draft.head_hidden(hidden)).argmax(-1)[0]
                err = (preds != targets) | ~label_ok  # [Q]

                for layer in range(num_layers):
                    weights = captured[layer]
                    if weights is None:
                        raise RuntimeError(
                            f"layer {layer} returned no attention weights; "
                            "eager backend not active?"
                        )
                    attn = weights[0].float()
                    mask_l = masks[layer][0, 0]
                    row_sum = (attn * mask_l.unsqueeze(0).float()).sum(-1)
                    assert torch.allclose(row_sum, torch.ones_like(row_sum), atol=5e-2), (
                        f"layer {layer}: visible-row softmax sum off: "
                        f"{(row_sum - 1).abs().max().item():.3e}"
                    )
                    metrics = chunk_metrics(
                        attn, mask_l, anchors, seq_len, block_size,
                        windows[layer], args.edge_k,
                    )
                    window_l = windows[layer]
                    reach = (
                        anchors[0, block_id] < window_l
                        if window_l is not None
                        else torch.ones(q_len, dtype=torch.bool, device=device)
                    )
                    stats[layer].add(
                        metrics,
                        err,
                        q_off,
                        reach,
                        block_decay(attn, seq_len, block_size),
                        top1_classes(
                            attn, mask_l, anchors, seq_len, block_size,
                            window_l, args.edge_k,
                        ),
                    )
                del captured, hidden, masks

    capture.remove()
    if n_samples == 0:
        raise RuntimeError("no usable samples")

    report = summarize(stats, windows, block_size, num_layers, args)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "attn_mass_report.json"), "w") as f:
        json.dump(report["json"], f, indent=2, default=float)
    save_npz(stats, os.path.join(args.out, "attn_mass_arrays.npz"), num_layers)
    print(report["console"])
    print(
        f"[probe] wrote {args.out}/attn_mass_report.json, "
        f"{args.out}/attn_mass_arrays.npz"
    )


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def _repeat_rows(vec, heads):
    return vec.unsqueeze(0).expand(heads, -1)


def summarize(stats, windows, block_size, num_layers, args):
    lines = []
    result = {"block_size": block_size, "edge_k": args.edge_k, "layers": [],
              "verdicts": {}}

    def emit(text=""):
        lines.append(text)

    emit("=" * 76)
    emit("ATTENTION MASS PROBE — per-layer summary (means over heads x rows)")
    emit("=" * 76)
    header = (
        f"{'L':>2} {'W':>5} {'P_blk':>7} {'P_anch':>8} {'anch/blk':>8} "
        f"{'edgeR':>7} {'H_ctx':>7} {'P_sink':>7}  {'top1 anch/blk/edge/mid %':>26}"
    )
    emit(header)
    emit("-" * len(header))

    suffix_half = block_size // 2
    for layer in range(num_layers):
        st = stats[layer]
        p_block = st.cat("p_block")
        p_anchor = st.cat("p_anchor")
        h_ctx = st.cat("h_ctx")
        edge_ratio = st.cat("edge_ratio")
        heads = st.rows[0]["p_block"].shape[0]
        err = torch.cat(st.err)
        off = torch.cat(st.offsets)
        err_h = _repeat_rows(err, heads).flatten().float()
        off_h = _repeat_rows(off, heads).flatten()

        block_rows = p_block > 0.05
        share = (
            (p_anchor / p_block.clamp_min(1e-9))[block_rows].mean().item()
            if block_rows.any()
            else float("nan")
        )
        top_total = sum(st.top1.values())
        top_str = "/".join(
            f"{100 * st.top1.get(k, 0) / max(top_total, 1):.0f}"
            for k in ("anchor", "block", "edge", "mid")
        )
        row = {
            "layer": layer,
            "window": windows[layer],
            "p_block": p_block.mean().item(),
            "p_anchor": p_anchor.mean().item(),
            "anchor_share_of_block": share,
            "edge_ratio": edge_ratio.mean().item(),
            "h_ctx": h_ctx.mean().item(),
            "p_sink": st.cat("p_sink").mean().item(),
            "err_rate": err_h.mean().item(),
            "top1_pct": {
                k: st.top1.get(k, 0) / max(top_total, 1) for k in st.top1
            },
        }
        # stash per-layer aligned vectors for cross sections below
        row["_vecs"] = (p_block, p_anchor, h_ctx, edge_ratio, err_h, off_h)
        result["layers"].append(row)
        emit(
            f"{layer:>2} {str(windows[layer]):>5} {row['p_block']:>7.3f} "
            f"{row['p_anchor']:>8.3f} {share:>8.3f} {row['edge_ratio']:>7.2f} "
            f"{row['h_ctx']:>7.2f} {row['p_sink']:>7.4f}  {top_str:>26}"
        )

    emit()
    emit("Suffix view (layer 0): err / P_anchor / H_ctx by block offset")
    emit("-" * 60)
    p_anchor0, h_ctx0 = result["layers"][0]["_vecs"][1], result["layers"][0]["_vecs"][2]
    err0, off0 = result["layers"][0]["_vecs"][4], result["layers"][0]["_vecs"][5]
    emit(f"{'o':>3} {'err':>6} {'P_anch':>8} {'H_ctx':>7}")
    for o in range(block_size):
        sel = off0 == o
        if sel.any():
            emit(
                f"{o:>3} {err0[sel].mean():>6.3f} {p_anchor0[sel].mean():>8.3f} "
                f"{h_ctx0[sel].mean():>7.2f}"
            )

    emit()
    emit("H5 contrast: band reaches position 0 (anchor < W) vs not")
    emit("-" * 60)
    h5_gaps = []
    for layer in range(num_layers):
        if windows[layer] is None:
            emit(f"L{layer}: full attention — every row reaches position 0")
            continue
        st = stats[layer]
        heads = st.rows[0]["p_block"].shape[0]
        reach = torch.cat(st.reach_pos0)
        if reach.all() or (~reach).all():
            emit(
                f"L{layer}: single group only "
                f"(reach={int(reach.sum())}/{reach.numel()}) — sequence too "
                "short for the H5 contrast"
            )
            continue
        reach_h = _repeat_rows(reach, heads).flatten()
        p_anchor, h_ctx, edge_ratio = (
            result["layers"][layer]["_vecs"][i] for i in (1, 2, 3)
        )
        emit(
            f"L{layer} W={windows[layer]}: P_anchor "
            f"{p_anchor[reach_h].mean():.3f} vs {p_anchor[~reach_h].mean():.3f} | "
            f"edgeR {edge_ratio[reach_h].mean():.2f} vs {edge_ratio[~reach_h].mean():.2f} | "
            f"H_ctx {h_ctx[reach_h].mean():.2f} vs {h_ctx[~reach_h].mean():.2f} "
            f"(n={int(reach.sum())}/{int((~reach).sum())})"
        )
        h5_gaps.append(
            p_anchor[~reach_h].mean().item() - p_anchor[reach_h].mean().item()
        )

    emit()
    emit(f"H4: error rate by H_ctx quartile (suffix = o >= {suffix_half})")
    emit("-" * 60)
    quart = {}
    h4_gaps = []
    for layer in range(num_layers):
        h_ctx, err_h, off_h = (
            result["layers"][layer]["_vecs"][i] for i in (2, 4, 5)
        )
        edges = torch.quantile(
            h_ctx, torch.tensor([0.25, 0.5, 0.75], device=h_ctx.device)
        )
        buckets, sfx = [], []
        bounds = [(None, edges[0]), (edges[0], edges[1]),
                  (edges[1], edges[2]), (edges[2], None)]
        for lo, hi in bounds:
            sel = torch.ones_like(h_ctx, dtype=torch.bool)
            if lo is not None:
                sel &= h_ctx >= lo
            if hi is not None:
                sel &= h_ctx < hi
            buckets.append(err_h[sel].mean().item() if sel.any() else float("nan"))
            sel_s = sel & (off_h >= suffix_half)
            sfx.append(err_h[sel_s].mean().item() if sel_s.any() else float("nan"))
        quart[layer] = {"all": buckets, "suffix": sfx}
        gap = sfx[-1] - sfx[0]
        if gap == gap:
            h4_gaps.append(gap)
        emit(
            f"L{layer}: Q1..Q4 err {buckets[0]:.3f}/{buckets[1]:.3f}/"
            f"{buckets[2]:.3f}/{buckets[3]:.3f} | suffix {sfx[0]:.3f}/"
            f"{sfx[1]:.3f}/{sfx[2]:.3f}/{sfx[3]:.3f}"
        )

    emit()
    emit("H2: within-block normalized mass on mask columns by |o - o'|")
    emit("-" * 60)
    decay_fits = []
    for layer in range(num_layers):
        st = stats[layer]
        ds = sorted(st.decay.keys())
        if len(ds) < 3:
            continue
        ys = np.array([st.decay[d] for d in ds])
        xs = np.array(ds, dtype=np.float64)
        ok = ys > 0
        slope, intercept = np.polyfit(xs[ok], np.log(ys[ok]), 1)
        resid = np.log(ys[ok]) - (slope * xs[ok] + intercept)
        ss_tot = ((np.log(ys[ok]) - np.log(ys[ok]).mean()) ** 2).sum()
        r2 = 1 - (resid**2).sum() / ss_tot if ss_tot > 0 else float("nan")
        decay_fits.append({"layer": layer, "slope": slope, "r2": r2})
        emit(
            f"L{layer}: slope={slope:.3f} R2={r2:.2f}  "
            + " ".join(f"{st.decay[d]:.3f}" for d in ds)
        )

    for row in result["layers"]:
        del row["_vecs"]

    emit()
    emit("=" * 76)
    emit("VERDICTS")
    emit("=" * 76)
    verdicts = {}

    shares = [
        r["anchor_share_of_block"]
        for r in result["layers"]
        if r["anchor_share_of_block"] == r["anchor_share_of_block"]
    ]
    med_share = float(np.median(shares)) if shares else float("nan")
    h1 = bool(shares) and med_share > 0.5
    verdicts["H1_anchor_quasi_sink"] = {"pass": h1, "median_share": med_share}
    emit(
        f"H1 anchor quasi-sink: {'PASS' if h1 else 'FAIL'} "
        f"(median anch/blk={med_share:.3f} over rows P_blk>5%, threshold 0.5)"
    )

    h2 = bool(decay_fits) and all(f["r2"] > 0.8 and f["slope"] < 0 for f in decay_fits)
    verdicts["H2_positional_smoothing"] = {"pass": h2, "fits": decay_fits}
    emit(
        f"H2 mask columns = positional smoothing: {'PASS' if h2 else 'FAIL'} "
        f"(monotone log-linear decay, R2>0.8, every layer)"
    )

    edge_ok = [r["edge_ratio"] for r in result["layers"] if r["window"] is not None]
    med_edge = float(np.median(edge_ok)) if edge_ok else float("nan")
    h3 = bool(edge_ok) and med_edge > 3.0
    verdicts["H3_edge_pileup"] = {"pass": h3, "median_edge_ratio": med_edge}
    emit(
        f"H3 band-edge pileup: {'PASS' if h3 else 'FAIL'} "
        f"(median edgeR={med_edge:.2f}, threshold 3x uniform)"
    )

    med_h4 = float(np.median(h4_gaps)) if h4_gaps else float("nan")
    h4 = bool(h4_gaps) and med_h4 >= 0.03
    verdicts["H4_diffuse_predicts_error"] = {
        "pass": h4,
        "median_suffix_q4_q1_gap": med_h4,
        "quartiles": quart,
    }
    emit(
        f"H4 diffuse attention predicts suffix errors: {'PASS' if h4 else 'FAIL'} "
        f"(median suffix Q4-Q1 gap={med_h4:.3f}, threshold 0.03)"
    )

    med_h5 = float(np.median(h5_gaps)) if h5_gaps else float("nan")
    h5 = bool(h5_gaps) and med_h5 > 0.02
    verdicts["H5_no_stable_sink_station"] = {"pass": h5, "median_gap": med_h5}
    emit(
        f"H5 far-anchor rows lean harder on anchor: {'PASS' if h5 else 'FAIL'} "
        f"(median P_anchor gap={med_h5:.3f}, threshold 0.02)"
    )

    emit()
    if h1 and h3 and h4:
        emit("CONCLUSION: sink story supported — proceed to Stage-1 ablation.")
    elif h1 or h3 or h5:
        emit(
            "CONCLUSION: dumping observed but the error link is missing — a sink "
            "would mostly formalize existing behavior; weigh against the conv arm."
        )
    else:
        emit("CONCLUSION: no dumping signal — drop the sink idea, keep the negative.")

    result["verdicts"] = verdicts
    return {"console": "\n".join(lines), "json": result}


def save_npz(stats, path, num_layers):
    arrays = {}
    for layer in range(num_layers):
        st = stats[layer]
        for key in LayerStats.KEYS:
            arrays[f"L{layer}_{key}"] = st.cat(key).cpu().numpy()
        arrays[f"L{layer}_err"] = torch.cat(st.err).cpu().numpy()
        arrays[f"L{layer}_offset"] = torch.cat(st.offsets).cpu().numpy()
        arrays[f"L{layer}_reach_pos0"] = torch.cat(st.reach_pos0).cpu().numpy()
    np.savez_compressed(path, **arrays)


# ---------------------------------------------------------------------------
# selftest
# ---------------------------------------------------------------------------


def make_selftest_artifacts():
    tiny = {
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
        "dflash_config": {
            "mask_token_id": 63,
            "target_layer_ids": [0, 1],
            "sliding_window": [-1, 64],
        },
    }
    from specforge.modeling.auto import AutoDraftModelConfig

    tmp = tempfile.mkdtemp(prefix="attn_mass_selftest_")
    draft_dir = os.path.join(tmp, "draft")
    os.makedirs(draft_dir)
    with open(os.path.join(draft_dir, "config.json"), "w") as f:
        json.dump(tiny, f)
    cfg = AutoDraftModelConfig.from_file(os.path.join(draft_dir, "config.json"))
    model = AutoDraftModel.from_config(cfg)
    model.save_pretrained(draft_dir)

    target_dir = os.path.join(tmp, "target")
    os.makedirs(target_dir)
    torch.save(
        {
            "model.embed_tokens.weight": torch.randn(64, 32),
            "lm_head.weight": torch.randn(64, 32),
        },
        os.path.join(target_dir, "pytorch_model.bin"),
    )

    data_dir = os.path.join(tmp, "shards")
    os.makedirs(data_dir)
    seq = 96
    torch.save(
        {
            "input_ids": torch.randint(0, 64, (seq,)),
            "loss_mask": torch.ones(seq),
            "hidden_states": torch.randn(seq, 64),
        },
        os.path.join(data_dir, "sample_000.ckpt"),
    )
    return draft_dir, target_dir, data_dir


def make_hf_selftest_artifacts():
    """Tiny HF target + pre-tokenized prompts for the HF capture selftest."""
    draft_dir, _target, _data = make_selftest_artifacts()
    tmp = os.path.dirname(draft_dir)

    from transformers import Qwen3Config, Qwen3ForCausalLM

    hf_dir = os.path.join(tmp, "hf_target")
    cfg = Qwen3Config(
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_hidden_layers=4,
        vocab_size=64,
        max_position_embeddings=256,
        tie_word_embeddings=False,
    )
    Qwen3ForCausalLM(cfg).save_pretrained(hf_dir)

    seq = 96
    prompts_path = os.path.join(tmp, "prompts.jsonl")
    with open(prompts_path, "w") as f:
        f.write(
            json.dumps(
                {
                    "input_ids": torch.randint(0, 64, (seq,)).tolist(),
                    "loss_mask": [1] * seq,
                }
            )
            + "\n"
        )
    return draft_dir, hf_dir, prompts_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    has_selftest = "--selftest" in sys.argv
    parser.add_argument("--draft-checkpoint", required=not has_selftest)
    parser.add_argument(
        "--target-checkpoint",
        help="HF dir for embed_tokens/lm_head; defaults to --target-model-path "
        "in HF capture mode",
    )
    parser.add_argument("--data", help="directory of offline .ckpt feature shards")
    parser.add_argument(
        "--target-model-path",
        help="HF target dir; with --prompts enables HF capture mode (no sglang)",
    )
    parser.add_argument("--prompts", help="prompts jsonl for HF capture mode")
    parser.add_argument("--chat-template", default=None)
    parser.add_argument("--prompt-max-length", type=int, default=8192)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-samples", type=int, default=6)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--max-anchors", type=int, default=256)
    parser.add_argument("--chunk-blocks", type=int, default=16)
    parser.add_argument("--edge-k", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="attn_mass_out")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    if args.selftest:
        draft_dir, target_dir, data_dir = make_selftest_artifacts()
        base = dict(
            max_samples=1,
            max_anchors=24,
            chunk_blocks=8,
            edge_k=16,
            device=args.device,
            seed=args.seed,
        )
        ckpt_args = argparse.Namespace(
            draft_checkpoint=draft_dir,
            target_checkpoint=target_dir,
            data=data_dir,
            hf_mode=False,
            out=args.out + "_ckpt",
            max_length=512,
            **base,
        )
        hf_draft, hf_target, hf_prompts = make_hf_selftest_artifacts()
        hf_args = argparse.Namespace(
            draft_checkpoint=hf_draft,
            target_checkpoint=hf_target,
            target_model_path=hf_target,
            prompts=hf_prompts,
            data=None,
            hf_mode=True,
            chat_template=None,
            prompt_max_length=512,
            out=args.out + "_hf",
            max_length=512,
            **base,
        )
        print("=== selftest mode 1/2: offline ckpt shards ===")
        run_probe(ckpt_args)
        print("\n=== selftest mode 2/2: HF target capture ===")
        run_probe(hf_args)
        return

    if args.target_model_path or args.prompts:
        missing = [
            name
            for name in ("target_model_path", "prompts", "draft_checkpoint")
            if not getattr(args, name)
        ]
        if missing:
            raise SystemExit(f"HF capture mode requires --{', --'.join(missing)}")
        args.hf_mode = True
        if args.target_checkpoint is None:
            args.target_checkpoint = args.target_model_path
    else:
        args.hf_mode = False
        if not args.data:
            raise SystemExit(
                "pass either --data (offline shards) or --target-model-path + "
                "--prompts (HF capture)"
            )
    run_probe(args)


if __name__ == "__main__":
    main()
