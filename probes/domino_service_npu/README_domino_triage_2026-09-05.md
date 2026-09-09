# Domino acceptance triage scripts

These probes were used to isolate the acceptance-decay issue (dp=1/32 or
dp=2/16) from the previously fixed DP hang. The acceptance decay is resolved;
the graph-only target-state corruption has a further 2026-09-09 follow-up
recorded below. The ad-hoc patch/note artifacts from the investigation have
been removed and the final fixes live in the vllm/vllm-ascend branches.

## Results so far (2026-09-05)

| Experiment | Result |
| --- | --- |
| dp=1/16 | Healthy, no decay. |
| dp=1/16, KV usage forced to ~80% | Healthy, no decay. |
| dp=1/32, full eager | Decays; near-end per-position rates fall to ~0. |
| dp=1/32 replacement delay 0 / 100 / 500 ms | All three still decay. This rules out finished-request cleanup speed; it is a steady-state batch/state bug. |
| dp=2/16, graph mode | In between dp=1/16 and dp=1/32: decays, recovers partially, decays again. |
| Worker-count boundary | Around 28 in the tested setup; may drift. |
| Same humaneval/159 prompt, healthy vs degraded | Healthy output readable; degraded output random throughout (no repetitive pattern). |
| Draft sliding-window layers replaced by full attention | Draft accuracy drops as expected, but worker-count instability largely disappears; 64 workers show less decay than 32 workers with sliding attention. |
| Full-attention draft degraded sample | Prompt prefix is correct, but generated continuation can be random digit-like text (e.g. `2   2   19  2  2 2     2`). |
| Cap only the two 3072 windows to 2048 | No acceptance decay at 32 workers; stable in a larger 64-worker run as well. |
| dp=2/48, graph vs eager | Eager stays healthy; the FULL-graph draft path still shows acceptance decay, so a second, graph-replay-specific issue exists. |
| dp=2/48 graph with `8050f9801` | Decay still exists after keeping captured block tables and enabling max workspace for mixed Domino windows; those two suspects are ruled out. |
| Fine-grained graph capture (user-forced every 7 from 7 to 224, every 8 from 8 to 256) | Decay still occurs: coarse capture-bucket padding is not required for the bug, because DP-wide dispatch can pad a rank even when fine sizes are available. |
| Decay onset vs request churn | Decay starts after a few dozen prompts have finished; consistent with stale padded-row metadata being replayed and then interacting with freed KV/state-block reuse. |
| Fallback fix `827ca25d7` | Domino draft kept off FULL graph (draft eager, target graph preserved) did **not** fix the decay: the corruption is in the target graph, not the draft graph. Reverted in `d39421346`. |
| Config/mode matrix (user-confirmed) | The dp=2/48 graph runs used windows capped to 2048 or below. The 3072→2048 cap is stable at 32/48/64 workers only in eager; graph mode is not stable. `--max-num-seqs` was never set. Full-attention graph comparison was inconclusive because acceptance was too low. |
| Degraded output over time (dp=2/48 graph) | Early in the decay a probe still returns readable text; near the end of the run it returns trash — first repeating “disabled”, then random patterns, then alternating single-word repetition and random text. |
| `[DOMINO_DEBUG]` hook | Hook-active marker and per-step dumps were added during triage and removed in `28ea68dd0` / `13cc5214e` once the root cause was found. |

## Resolution (2026-09-07)

There were two independent triggers:

1. **Draft windows above the FIA band ceiling (eager and graph).** The
   trained `[3072, 2048, 512, 512, 1024, 1024, 3072]` recipe corrupts the
   non-causal FIA band path (`sparse_mode=4`, fixed `2048x2048` band mask).
   Capping the two 3072-window layers to 2048 removes that corruption and is
   stable through 64 workers in eager mode.

2. **Target FULL-graph replay with stale padded GDN/Mamba rows (graph only).**
   After the windows were capped, dp=2/48 graph still decayed. The remaining
   bug was not the draft: forcing the Domino draft eager while keeping the
   target graph intact did not help, attention-side graph fixes (captured
   block tables, max workspace) did not help, and neither did fine-grained
   capture sizes or `--no-async-scheduling`.

Root cause of #2: `_pad_query_start_loc_for_fia` collapsed uniform FULL
decode padding back to the live request count (`num_reqs_padded = num_reqs`).
GDN graphs capture metadata at request granularity, so replaying a padded
graph over fewer metadata rows left the padded slots' persistent
conv/recurrent state rows stale; once requests finished and blocks were
reused, the next replay advanced state through freed/reallocated blocks.

Fix (vllm-ascend `aa66a707e`, porting vllm-ascend PR #15529 semantics):

- preserve the captured request shape for uniform decode graphs in
  `NPUModelRunner._pad_query_start_loc_for_fia`;
- mark padded Mamba/GDN rows as speculative dummies
  (`num_decode_draft_tokens = num_spec`) so replay stays on the same pure-spec
  GDN metadata path as capture and refreshes/nullifies every padded row;
- expose `embed_input_ids` through the ACL graph wrapper.

After `aa66a707e`, the previously failing graph configurations run without
the acceptance decay.

### Follow-up (2026-09-09): correct acceptance, repeated garbage tokens

The service later showed correct acceptance counters but repeated garbage
text after roughly 60 requests, often beginning midway through a response.
The remaining bug was still in `_pad_query_start_loc_for_fia`: it classified
a FULL graph as uniform decode from total token count alone. A mixed batch
with descriptor 4, `decode_query_len=8`, and real query lengths
`[4, 12, 16]` sums to 32, so the old code produced
`query_start_loc=[0, 4, 16, 32, 40]` even though the graph has only 32
tokens. The new query-length guard (vllm-ascend `3bc298c3e`, port of PR
#15707) produces the correct mixed layout `[0, 4, 16, 32, 32]`.

The same upstream PR also expands 1D text positions to the three T/H/W
planes expected by Qwen3.5/3.6's fused MRoPE kernel. Without that, the H/W
plane offsets read the wrong cos/sin cache rows as positions grow, which can
corrupt attention late in long responses.

### Follow-up (2026-09-09): no-spec FULL replay and the 16-per-rank boundary

After the fixed-row GDN state-indexing change, graph mode could still produce
correct acceptance counters with garbage text after request churn. The new
observation is that the failure does not occur below 16 concurrent requests
per DP rank.

That boundary points to mixed prefill/decode batches. A FULL graph captured
with speculative decoding contains the speculative conv1d/recurrent tasks.
At replay those tasks consume persistent spec inputs instead of rebuilding
the Python branch. The builder reset those inputs only for a pure non-spec
decode replay. Mixed batches with no runtime draft rows therefore replayed
the captured spec tasks with stale `spec_state_indices_tensor`,
`spec_query_start_loc`, `num_accepted_tokens`, and
`spec_actual_seq_lengths`, advancing persistent GDN state for the wrong
request. Below 16 per-rank concurrency the scheduler rarely forms such a
mixed batch, which explains the threshold.

The fix (vllm-ascend `5f1e3ee03`) resets the captured spec inputs whenever
`num_spec_decodes == 0`, including mixed prefill/decode replays, and clears
the spec masks when a dynamic-SD batch has no runtime draft tokens. It also
keeps Domino on the base `1 + 2 * num_spec` reorder threshold; the local
`num_spec` override was smaller than the target's `1 + num_spec`
verification width.

Regression tests cover mixed-prefill no-spec FULL replay, zero-draft dynamic
SD rows, and the Domino threshold. NPU end-to-end confirmation is pending.

## Script notes after resolution

- `probe_non_causal_band.py` supports `--compare` for W=2048 vs W=3072 if the
  FIA band behavior ever needs to be re-audited.
- The old per-request `[DOMINO_DEBUG]` procedure no longer applies: the hook
  was removed from the vllm-ascend worktree.

## Test 1: replacement-rate at fixed concurrency

Run:

```bash
python probes/domino_service_npu/check_acceptance_replacement_rate.py \
  --server-port 4144 \
  --served-model-name qwen3.6-35b \
  --dataset humaneval \
  --dataset-path /path/to/human-eval-v2-20210705.jsonl \
  --num-workers 32 \
  --max-tokens 256 \
  --replacement-delay-ms 0 \
  --monitor-interval 10
```

Then repeat with `--replacement-delay-ms 100` and `--replacement-delay-ms 500`.
Use the same prompt set and `--num-prompts` for each run.

Interpretation:

- Delay 0 still decays, but 100/500 ms delays recover or weaken the decay:
  the bug is tied to finish/reuse rate.
- All delays still decay: the bug is a steady-state batch-size/state bug, not
  the cleanup speed.

Confirmed: all three delays still decay.

The script writes a JSON summary under
`results/domino_acceptance/replacement_delay_*.json`.

## Test 2 (historical): per-step draft dump for a degraded request

The `[DOMINO_DEBUG]` hook used to isolate the degraded window printed
`req_ids/num_sampled/num_rejected` plus `prev_drafts/new_drafts` rows from
`AscendDominoSpeculator.propose`. It confirmed that degraded requests were
rejecting every draft, but the lines were interleaved with the 32-worker
background and did not identify the root cause. The hook was removed from
vllm-ascend in `28ea68dd0` / `13cc5214e`; this procedure is kept only as a
record and no longer applies to the current worktree.

## Experiment status (final)

- Cap the two 3072 windows to 2048: DONE, and required for the eager path.
- Graph-only target GDN/Mamba padding replay bug: FIXED in vllm-ascend
  `aa66a707e`; the failing dp=2/48 graph run no longer decays.
