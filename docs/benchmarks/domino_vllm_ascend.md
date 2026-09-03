# Domino on vLLM + vLLM-Ascend (Qwen3.6 / Qwen3.8)

This is the deployment contract for the migrated verifiedBase Domino draft.
It assumes the branches:

* `vllm` `codex/DRAFT_qwen36_35B`
* `vllm-ascend` `codex/DRAFT_qwen36_35B`

The service validation is strict: the draft config's `block_size` must equal
the launched `num_speculative_tokens`.  For the intended seven-token service,
the retrained draft config must therefore use `block_size: 7`.  The checked-in
migration configs still use `block_size: 16` for the original training recipe;
do not launch those with `--spec-tokens 7` without retraining at 7.

## Qwen3.6-35B-A3B

```bash
vllm serve /path/to/Qwen3.6-35B-A3B \
  --quantization ascend \
  --tensor-parallel-size 2 \
  --max-model-len 262144 \
  --served-model-name qwen3.6-35b \
  --speculative-config '{
    "method": "domino",
    "model": "/path/to/exported/domino-draft/qwen3.6-35b-a3b",
    "num_speculative_tokens": 7,
    "draft_tensor_parallel_size": 1
  }'
```

`--enforce-eager` can be added to the speculative config for the initial
debug run:

```bash
  --speculative-config '{
    "method": "domino",
    "model": "/path/to/exported/domino-draft/qwen3.6-35b-a3b",
    "num_speculative_tokens": 7,
    "draft_tensor_parallel_size": 1,
    "enforce_eager": true
  }'
```

## Qwen3.8-27B

Only the model path and auxiliary layer IDs change. The Qwen3.8 checkpoint
uses the same `Qwen3_5`/`Qwen3_5Moe` architecture metadata as Qwen3.6:

```bash
vllm serve /path/to/Qwen3.8-27B \
  --quantization ascend \
  --tensor-parallel-size 2 \
  --max-model-len 262144 \
  --served-model-name qwen3.8-27b \
  --speculative-config '{
    "method": "domino",
    "model": "/path/to/exported/domino-draft/qwen3.8-27b",
    "num_speculative_tokens": 7,
    "draft_tensor_parallel_size": 1
  }'
```

`draft_tensor_parallel_size=2` is also allowed when the target is TP=2.

## Draft config contract

The exported draft `config.json` must use:

* `architectures: ["Qwen3DominoModel"]` or `["DominoDraftModel"]` (the
  service accepts the training-time `DominoDraftModel` alias and normalizes
  it to `Qwen3DominoModel`)
* `block_size: 7` (must match `num_speculative_tokens`)
* `dflash_config.fusion_mode: "flare"`
* `dflash_config.heterogeneous_kv: true`
* `dflash_config.target_layer_ids`: `[1, 7, 13, 19, 25, 31, 37]` for
  Qwen3.6-35B-A3B, `[1, 11, 21, 31, 41, 51, 61]` for Qwen3.8-27B
* `dflash_config.target_hidden_size`: 2048 / 5120 respectively
* `dflash_config.sliding_window`: the per-layer list if the training config
  used one
* `dflash_config.qat_w_bit`: 4 (W4A8 bulk) with `qat_w4a4_layers`

The target auxiliary hidden states are looked up under
`model.language_model.embed_tokens.weight` in the exporter and service.
