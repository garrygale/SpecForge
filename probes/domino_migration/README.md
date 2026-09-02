# Domino dflare migration probes

This folder contains CPU-only checks for the migrated verifiedBase Domino
path. They do not require SGLang, Mooncake, torch_npu, or an NPU.

## Run

```bat
cd /d C:\Users\g\Desktop\codeAgents\SpecForge
set PYTHONPATH=C:\Users\g\Desktop\codeAgents\SpecForge
C:\Users\g\Desktop\codeAgents\phi-GNNv2\.venv\Scripts\python.exe probes\domino_migration\probe_domino_dflare_cpu.py
```

The probe exercises:

* dflare layer fusion with `target_hidden_size != hidden_size`
* heterogeneous K/V projections (`k_proj_target` / `v_proj_target`)
* per-layer sliding-window masks
* Domino GRU correction head dimensions
* online training forward/loss
* W4A8 QAT replacement
* `spec_generate(..., return_acceptance_stats=True)` against a fake target

Downloaded official target config/tokenizer metadata used for migration is
under `target_configs/`. No safetensors were downloaded.

Running the migrated config/key tests:

```bat
C:\Users\g\Desktop\codeAgents\phi-GNNv2\.venv\Scripts\python.exe -m unittest tests.test_utils.test_domino_dflare_cpu -v
```
