# Domino service NPU probes

These probes are for the Qwen3.6-35B-A3B / Qwen3.8-27B Domino
`vllm` + `vllm-ascend` service path.  They intentionally do **not** depend on
`sglang` and do not download model weights.  Run them on the actual Ascend
machine, not on this CPU-only workstation.

## Files

* `probe_quant_paths.py`
  W4A8 / W4A4 / W8A8 and mixed Domino QKV projections, comparing the fused
  single-pack path against the per-layer path (the service invariant), plus
  eager and ACL graph replay.
* `probe_grouped_fused_kv.py`
  One grouped `npu_grouped_matmul` W4A8 context-K/V projection versus the
  seven per-layer baseline, with ACL graph replay.
* `probe_gdn_accepted_boundary.py`
  CPU/device simulation of the accepted-token / `spec_query_start_loc`
  boundary that is reported as issue #9956.  It shows whether the failure is
  limited to all-accepted tokens or also appears for partial final rounds, and
  prints the per-row clamp that `vllm-ascend` now applies in the service.
* `probe_domino_config_and_7steps.py`
  Validates the migrated draft configs and exercises a 7-step GRU correction
  loop against the SpecForge `eagerGRU` path.
* `probe_non_causal_band.py`
  Non-causal sliding-window FIA band-mode smoke test (`sparse_mode=4`,
  `pre_tokens == next_tokens == window`), using the prefill-no-cache service
  shape (`block_table=None`, contiguous TND K/V).
* `probe_acceptance_over_generation.py`
  Starts one long chat request and polls the server's spec-decode counters
  while it runs, printing per-interval acceptance.  Use this to distinguish
  acceptance decay caused by generation length/context from decay caused by
  concurrent request churn.

## Example

Run every probe in order with one command:

```bash
bash probes/domino_service_npu/run_all_probes.sh
```

To use a specific Python environment:

```bash
PYTHON=/path/to/target/python bash probes/domino_service_npu/run_all_probes.sh
```

The individual equivalent commands are:

```bat
cd /d C:\Users\g\Desktop\codeAgents\SpecForge
set PYTHONPATH=C:\Users\g\Desktop\codeAgents\SpecForge
C:\Users\g\Desktop\codeAgents\phi-GNNv2\.venv\Scripts\python.exe probes\domino_service_npu\probe_quant_paths.py --real-dims
C:\Users\g\Desktop\codeAgents\phi-GNNv2\.venv\Scripts\python.exe probes\domino_service_npu\probe_grouped_fused_kv.py --real-dims
C:\Users\g\Desktop\codeAgents\phi-GNNv2\.venv\Scripts\python.exe probes\domino_service_npu\probe_gdn_accepted_boundary.py
C:\Users\g\Desktop\codeAgents\phi-GNNv2\.venv\Scripts\python.exe probes\domino_service_npu\probe_domino_config_and_7steps.py
C:\Users\g\Desktop\codeAgents\phi-GNNv2\.venv\Scripts\python.exe probes\domino_service_npu\probe_non_causal_band.py --window 512
```

The probes return a non-zero exit code on assertion failure.  A probe that
cannot run because `torch_npu` is unavailable prints `SKIP` rather than
silently passing.
