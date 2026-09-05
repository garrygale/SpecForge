# Domino acceptance triage scripts

These probes isolate the remaining acceptance-decay issue (dp=1/32 or
dp=2/16) from the previously fixed DP hang and graph-mode problems.

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
| `[DOMINO_DEBUG]` hook | Hook-active marker prints with `VLLM_DOMINO_DEBUG=1`; per-request lines are still interleaved under the 32-worker background load and have not yet been isolated. |

Conclusion from these results: the remaining problem is a steady-state,
batch-count-dependent state corruption. The non-causal sliding-window draft
attention path is strongly implicated because replacing sliding attention
with full attention removes most of the worker-count dependence.

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

## Test 2: per-step draft dump for a degraded request

### Server-side instrumentation

The current vllm-ascend worktree contains an env-gated debug hook in
`vllm_ascend/worker/v2/spec_decode/domino/speculator.py`. It activates only
with `VLLM_DOMINO_DEBUG=1`, and it prints when at least one request has
`num_sampled <= 1`:

```text
[DOMINO_DEBUG] req_ids=... num_sampled=... num_rejected=...
[DOMINO_DEBUG] req=<id> prev_drafts=[...] new_drafts=[...]
```

`prev_drafts` are the drafts that were just verified (usually rejected in the
collapse); `new_drafts` are the proposals for the next round.

If the worktree is not synced to the NPU host, apply the equivalent patch
manually: add an `import os`, call a `_debug_log_proposal(...)` helper in
`AscendDominoSpeculator.propose`, and log rows where `num_sampled <= 1`.

### Client procedure

1. Restart the server with:

   ```bash
   VLLM_DOMINO_DEBUG=1 <your normal vllm serve command>
   ```

2. Run the 32-worker humaneval workload until the monitor or server running
   stats show near-zero per-position acceptance.

3. While the service is still degraded, start a single long request:

   ```bash
   python probes/domino_service_npu/probe_acceptance_over_generation.py \
     --server-port 4144 \
     --served-model-name qwen3.6-35b \
     --prompt "Solve this step by step: ..." \
     --max-tokens 512
   ```

4. Capture the server log tail for `[DOMINO_DEBUG]`, and keep the probe JSON
   from `results/domino_acceptance/acceptance_trace_*.json`.

Send back:

- the launch command / startup log (max_num_seqs, mamba_cache_mode,
  async_scheduling, graph vs eager);
- the replacement-delay summaries;
- the `[DOMINO_DEBUG]` lines for a degraded request;
- the probe trace JSON showing whether that single request stayed degraded.

## Next experiments (pending)

1. Keep sliding attention but cap each draft layer window to `<=2048`
   (replace the two `3072` windows with `2048`). If corruption disappears,
   focus on the FIA non-causal band/mask boundary.
2. Keep the 512/1024 sliding layers but make the two large-window layers full
   attention, preserving more draft quality while isolating which layer(s)
   trigger corruption.
3. Make the debug hook request-scoped (wall-clock timestamped per-request JSON
   files) so the degraded single request can be separated from the 32-worker
   background.
