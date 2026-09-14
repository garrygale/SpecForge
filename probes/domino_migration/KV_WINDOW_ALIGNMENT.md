# Domino draft KV cache: training vs serving alignment

Scope: SpecForge Domino training (`configs/qwen3.6-35b-a3b-domino-dflare-verifiedBase.json`
and small-window variants of it) compared against the vLLM / vLLM-Ascend Domino
draft attention, plus SpecForge's in-process acceptance evaluator.

**Status (2026-09-14): fixed on the training side.** Sliding draft layers now
train against the served KV visibility by default (section 6). The tables in
sections 2-4 document the pre-fix behaviour; reproduce them with
`probe_kv_window_alignment.py` (legacy column) and
`probe_domino_window_ab.py --legacy-mask`.

Reproduce with:

```bash
set PYTHONPATH=C:/Users/g/Desktop/codeAgents/SpecForge
C:/Users/g/Desktop/codeAgents/phi-GNNv2/.venv/Scripts/python.exe \
    probes/domino_migration/probe_kv_window_alignment.py [--verbose]
C:/Users/g/Desktop/codeAgents/phi-GNNv2/.venv/Scripts/python.exe \
    probes/domino_migration/probe_domino_window_ab.py --sweep --block-sweep
```

## 1. Layout is aligned; only the mask is not

Both sides put the same absolute-position K/V into the draft's attention input:

| | training (SpecForge) | serving (vLLM / vLLM-Ascend) |
| --- | --- | --- |
| context K/V | target hidden states, flare-fused per draft layer, projected with `k_proj_target` / `v_proj_target` | same fusion (`combine_hidden_states`) and same target projections, pre-written every step by `precompute_and_store_context_kv` |
| context positions | `0 .. S-1` (whole sequence) | `0 .. L` where `L = last_valid_pos` (target batch positions, re-written each step) |
| block K/V | draft hidden states of the block inputs (`mask_token_id`, offset 0 = anchor token) | same (query K/V written by the draft attention into the same cache) |
| block positions | `anchor + j`, `j = 0..block_size-1` | `L + 1 + j`, `j = 0..N-1` (`N = num_speculative_tokens`) |
| block element 0 | anchor token (the token the target has already produced) | bonus token (`last_sampled`): the corrected/re-sampled token that starts the next block |
| RoPE positions | absolute (`full_position_ids`) | absolute (`context_positions` / query positions) |
| labels | `label_start = 1` (shift): element `j` predicts token `anchor+j+1` | element `j` predicts the token after its own position (`sample_pos = query_pos + 1`) |

So if you align the training anchor `A` with the serving `L + 1` (the bonus
token position), every K/V slot, position and label lines up. The draft KV
cache contents and positions are *not* the source of the problem.

## 2. What is misaligned: which cached entries each query may read

Training builds the sliding mask in
`specforge/algorithms/common/dflash_family_model.py` (`create_dflash_sdpa_mask`,
`create_dflash_block_mask`):

```python
mask_context = (kv_idx < S) & (kv_idx < anchor)
if sliding_window is not None:
    context_lower_bound = anchor + q_offset - (sliding_window - 1)
    mask_context &= kv_idx >= context_lower_bound
mask_draft = is_draft & (q_block == kv_block)
if sliding_window is not None:
    mask_draft &= kv_block_offset <= q_block_offset      # <-- causal in block
```

i.e. for every sliding layer the visible set of query `j` (position `q = A + j`)
is the *contiguous* window `[q - (W-1), q]` — W positions ending at the query.

Serving does something different for sliding layers, because Domino declares
every draft layer **non-causal**:

* `vllm/model_executor/models/qwen3_domino.py:91` —
  `causal = bool(dflash_config.get("causal", False))` (default `False` for all layers).
* `vllm/v1/attention/backends/flash_attn.py:308` `_maybe_symmetrize_window`
  turns a causal `(W-1, 0)` window into `(W-1, W-1)` when attention is non-causal.
* vLLM-Ascend `vllm_ascend/attention/attention_v1.py:1456` runs
  `sparse_mode=4, pre_tokens=W, next_tokens=W` for non-causal sliding layers.

So the served visible set of query `j` is `[q - (W-1), q + W]` — a *symmetric*
band. The context half is identical to training; the difference is the block
half: at serving each query also reads the K/V of its own block's **future**
positions (the mask tokens it will fill in later).

Measured delta (probe 1, block_size 16, long context):

| W | trained K/V per query (j=0) | served (j=0) | extra (never seen in training, j=0) | extra / trained | hidden by served mask (W < block) |
| --- | --- | --- | --- | --- | --- |
| 2 | 2 | 4 | +2 | +100% | 77% of trained K/V |
| 4 | 4 | 8 | +4 | +100% | 55% |
| 8 | 8 | 16 | +8 | +100% | 22% |
| 16 | 16 | 31 | +15 | +94% | 0 |
| 32 | 32 | 47 | +15 | +47% | 0 |
| 64 | 64 | 79 | +15 | +23% | 0 |
| 128 | 128 | 143 | +15 | +11.7% | 0 |
| 512 | 512 | 527 | +15 | +2.9% | 0 |
| 2048 | 2048 | 2063 | +15 | +0.7% | 0 |

The forward reach of the band is always `min(W, block_size - 1 - j)` positions,
so with `W >= block_size` the served model reads the *whole* block, and with
`W < block_size` it additionally drops block positions older than `q - (W-1)`
that training exposed.

Two control rows from the same probe:

* a *causal* sliding layer (what DFlash does: `pre_tokens=W, next_tokens=0`)
  matches training exactly — 0 extra, 0 missing for every W;
* full-attention layers match exactly (0 extra, 0 missing).

So the mismatch is specific to the combination "sliding layer + non-causal
serving", which is exactly the Domino path.

## 3. Why the error grows as the window shrinks

The extra entries are constant in count (`<= block_size - 1`, i.e. 15 for the
checked-in configs) but their share of the attention input scales as `1/W`:
0.7% at W=2048, 2.9% at W=512, 23% at W=64, 47% at W=32, 94% at W=16. Their
(query, key) score pairs were masked out during training, so nothing constrains
them at serving, and they compete for softmax mass in all 7 sliding layers.
Below `W = block_size` the served band also *withholds* K/V the model was trained
to use (22%/55%/77% of the trained window at W=8/4/2), so both directions break.

That is exactly the observed pattern: large windows look fine, small windows
degrade.

## 4. A second, independent mismatch: the in-process evaluator

`inference/check_acceptance.py` measures acceptance in-process through
`DFlashDraftModel.spec_generate` -> `_domino_generate_step`
(`specforge/modeling/draft/dflash.py`). That path passes **no attention mask**
and `is_causal=False`; HuggingFace's `sdpa_attention_forward` /
`eager_attention_forward` take `sliding_window` in `**kwargs` and ignore it,
while `past_key_values_draft.crop(start)` leaves the *entire* context in the
draft cache. The evaluated draft therefore sees all context positions and the
whole block, bidirectionally — a pattern that matches neither training nor the
vLLM/vLLM-Ascend service. (With `flash_attention_2` instead of sdpa/eager the
window *is* applied, as a symmetric band, because transformers maps a
non-causal `sliding_window` to `(W-1, W-1)`.)

Same-weights A/B (probe 2, tiny draft trained to 100% with the training mask;
only the mask changes at evaluation):

| window | training mask | served band mask | evaluator (unmasked) mask |
| --- | --- | --- | --- |
| block 8, W=8 | 1.0000 | 0.9949 | **0.1487** |
| block 8, W=16 | 1.0000 | 0.9972 | **0.3128** |
| block 8, W=32 | 1.0000 | 1.0000 | **0.5459** |
| block 8, W=64 | 1.0000 | 1.0000 | **0.7698** |
| block 16, W=16 | 1.0000 | 0.9954 | **0.2320** |
| block 16, W=64 | 1.0000 | 0.9962 | **0.7482** |

The unmasked evaluator degrades monotonically as the window shrinks, which is
the same shape as the reported symptom; the band mismatch is real but small in
this toy. If the "inference accuracy" you measured came from
`inference/check_acceptance.py`, these numbers are the dominant explanation; if
it came from the service, section 2 is.

## 5. Corroboration in-tree

The same failure mode was already identified and fixed for the Draft-OPD replay
path on branch `upstream/codex/draft-opd-replay`:

* commit `3b582366` "Fix noncausal sliding attention in DFlash replay" adds
  `sliding_draft_causal` to both mask builders and switches it off when the
  draft config declares `is_causal=False`;
* `docs/sections/concepts/DraftOPD.md` ("Serving and replay must describe the
  same policy"): *"With `is_causal=False`, draft positions can attend to future
  positions within their own block, while the sliding lower bound still excludes
  old positions. They cannot attend to another draft block or to target taps at
  or after the anchor."*;
* `tests/test_modeling/test_dflash_opd_swa.py` locks the contract in.

Upstream reference (`Domino-main/code/dflash.py`) also sets `is_causal = False`
on every draft layer and transformers maps a non-causal `sliding_window` to
`(W-1, W-1)` (`modeling_flash_attention_utils.py:655`), i.e. the symmetric band
*is* the intended serving semantics. That makes the **training mask** the side
to fix.

## 6. What was changed (implemented)

The sliding mask now describes what serving does. In both
`create_dflash_sdpa_mask` and `create_dflash_block_mask` the draft half of a
sliding layer is

```python
draft_forward_reach = q_block_offset if sliding_draft_causal else q_block_offset + sliding_window
mask_draft &= (kv_block_offset >= q_block_offset - (sliding_window - 1))
mask_draft &= (kv_block_offset <= draft_forward_reach)
```

so the default (`sliding_draft_causal=False`) is the served band
`[q-(W-1), q+W]`, and `True` is the causal band `[q-(W-1), q]`. Both modes keep
the sliding lower bound — for `W >= block_size` the causal mode is therefore
bit-identical to the historical mask — and both keep `kv < anchor`, so no
target tap at or after the anchor ever leaks into the draft block.

Causality is resolved by `resolve_sliding_draft_causal` in
`specforge/modeling/draft/dflash.py`, which is exposed as
`DFlashDraftModel.sliding_draft_causal` and consumed by
`OnlineDFlashModel` (plumbed through **both** branches of
`_forward_draft_blocks`, including the per-layer `layer_sliding_windows`
branch that the Domino configs use) and recorded in the DFlash, DSpark and
Domino resume contracts (`dflash_/dspark_/domino_sliding_draft_causal`).
There is **no training-only config key**: the resolver mirrors the served
decision verbatim from `dflash_config.causal` (the field
`_domino_layer_attention` and `_dflash_layer_causal` read) plus each family's
served default:

1. `dflash_config.causal` when present (Domino, DFlash and DSpark);
2. otherwise Domino -> `False` (non-causal band), DFlash/DSpark -> `True`
   (sliding layers causal, unchanged behaviour).

The checked-in Domino draft configs
(`configs/qwen3*-domino-dflare-verifiedBase.json`) therefore need no change:
with no `causal` key they already match the service's non-causal default.

`_build_dflash_family_model` prints the resolved choice once at model build
time (`describe_draft_mask_choice`), e.g.
`[draft-mask] DRAFT BLOCK NON-CAUSAL (band=q-(W-1)..q+W): sliding_layers=7 ...; dflash_config.causal not set -> Domino default false`,
so a training log shows both the mask and where the decision came from.

Tests:

* `tests/test_utils/test_dflash_serving_parity.py` asserts the shipped default
  equals the served band (dense and, when available, flex builders) for
  `W` from 1 to 2048, that the legacy flag equals the causal band, that the
  two modes share the context half and differ only in the block's own future
  entries, that full-attention layers are unaffected, the resolution order, and
  that the checked-in Domino configs resolve to the band.
* `tests/test_utils/test_dflash_mask.py` (now CPU-runnable) keeps the
  element-level reference for both modes.
* `tests/test_modeling/test_dflash_sliding.py` and
  `tests/test_utils/test_dflash_losses.py` cover the model attribute, the
  config overrides and the mask plumbing.

## 7. Remaining follow-ups

### A. Retrain (or opt the service into the causal mask)

Existing Domino checkpoints were trained with the legacy causal draft block;
they now need either a retrain (recommended, so the draft learns the future
mask-K/V it will see) or the serving-side opt-in below.

### B. Serving-side alternative (no retraining)

Make sliding layers causal at serving so an existing checkpoint matches its
training mask exactly: use
`causal = (layer_types[i] == "sliding_attention")` in
`_domino_layer_attention` (mirroring DFlash's `_dflash_layer_causal`), or for
all-sliding configs set `dflash_config.causal = true` in the exported config.
That yields `pre_tokens=W, next_tokens=0`, which the probe shows is identical
to the legacy training mask for every W. Caveat: the global override would
also make full-attention layers causal, and it changes serving semantics for
every existing checkpoint, so re-validate acceptance.

### C. Measurement and guardrails

The in-process evaluator described in section 4 still ignores the window; fix
it (build the same mask in `_domino_generate_step`) before trusting its numbers
for windowed configs. Keep `block_size == num_speculative_tokens` and
`W >= block_size`, and keep non-causal band windows `<= 2048` on Ascend
(`vllm-ascend/docs/domino_acceptance_batch_issue_2026-09-04.md`).

### D. What the fix does not address

The Domino correction head is still teacher-forced during training
(ground-truth `prev_ids`) while it consumes its own sampled tokens at serving.
That exposure bias is not KV-cache related, but it grows in relative
importance when the window leaves little usable context.
contains the reference implementation and prints PASS).

Expect the training-time `acc` to dip slightly (the model now also sees future
mask K/V) while served acceptance rises; retrain or fine-tune the affected
checkpoints.

### B. Serving-side (no retraining)

Make sliding layers causal at serving so they match training exactly: use
`causal = (layer_types[i] == "sliding_attention")` in
`_domino_layer_attention` (mirroring DFlash's `_dflash_layer_causal`), or for
all-sliding configs set `dflash_config.causal = true` in the exported config.
That yields `pre_tokens=W, next_tokens=0`, which the probe shows is identical
to training for every W. Caveat: the global override would also make
full-attention layers causal, and it changes serving semantics for every
existing checkpoint, so re-validate acceptance.

### C. Measurement and guardrails

1. Fix the in-process evaluator (or stop using it for windowed configs): build
   the same mask (window + block causality) in `_domino_generate_step`, or at
   minimum force `attn_implementation="sdpa"` and assert the window is applied.
2. Keep `block_size == num_speculative_tokens` (`docs/benchmarks/domino_vllm_ascend.md`)
   and keep `W >= block_size`; below that the band both adds and removes K/V.
3. Ascend: non-causal band windows above 2048 are unsupported by the fixed
   `2048x2048` band mask (`vllm-ascend/docs/domino_acceptance_batch_issue_2026-09-04.md`).
4. The training mask reads `dflash_config.causal`, the same field vLLM /
   vLLM-Ascend read, so there is nothing to keep in sync manually. Export the
   draft config after any change; the service reads the exported
   `config.json`, not `configs/*.json`. (Draft-OPD's separate replay module on
   `upstream/codex/draft-opd-replay` still keys off a top-level `is_causal`.)
5. Anything left after A/B is a *different* gap: the Domino correction head is
   teacher-forced in training (ground-truth `prev_ids`) and consumes its own
   sampled tokens at serving. That exposure bias is not KV-cache related but
   grows in relative importance when the window leaves little usable context.
