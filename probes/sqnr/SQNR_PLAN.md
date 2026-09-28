# Plan: SQNR analysis of DFlash-vs-DFlare target-hidden fusion

Companion to `SQNR_ANALYSIS_V0.md` (the lens, validated) and
`probe_sqnr_budget.py` (the probe, running). This file is the plan for scaling
it up, plus the decisions that need your call before S1 starts.

Objective: turn "flare fuses in a quantizable way, fc does not" from an
observation into a **predictive, falsifiable account** -- and use it to say what
to build next (in the draft architecture and in the quantizer).

---

## 0. Where v0 stands

Done and runnable now (this session, CPU, seconds):

| artefact | what it gives |
|---|---|
| `probes/probe_sqnr_budget.py --laws` | Laws 1-5 on synthetic ground truth, with measured-vs-predicted errors |
| `probes/probe_sqnr_budget.py --config ... --random` | the fc-vs-flare budget: ledger, both paths in the same units, audit ratios |
| `reports/sqnr_laws.json`, `reports/sqnr_budget.json` | machine-readable outputs |
| `SQNR_ANALYSIS_V0.md` | the lens, the validated laws, the mechanism, the predictions |

State of the claim: the mechanism is pinned to exact code paths
(`dflash.py:1292` vs `:1397`, `wxay.py:52`, `wxay.py:200`), the four laws that
carry it are validated to <0.1 dB outside the annihilation regime, and the
synthetic budget reproduces the sign and order of magnitude of the measured
18.4 dB real gap. What is **not** yet established: the real-data ledger, the
SQNR-to-acceptance bridge, and the intervention.

## 1. Stages, gates, and what each one can kill

**S0 -- lens + laws + probe. DONE (v0).** Output: the framework above.

**S1 -- real-data SQNR ledger (no training; hours).**
Run the probe on real captured target hidden states and both trained
checkpoints; sweep a_bit in {2,3,4,5,6,8,16}. Produce:
1. the block ledger (per-block RMS, disparity, spikiness tax, disparity tax,
   `b_eff` per block, `blocks_with_le_1_eff_bit`) at each bit width;
2. fc: SQNR at the quantized point AND at the output, law-predicted vs measured,
   audit ratios, `destroyed_signal_fraction`, and **`pi` of the trained fc vs the
   uniform random-init baseline** (prediction 3 in the analysis: trained `pi`
   shifts toward the loud blocks);
3. flare: fused SQNR per draft layer, dilution available
   (`w_max/||w||_2`) vs realised, `inter_layer_cosine_mean`;
4. the dose-response curve: SQNR(bits) for concat / fc output / fused, so the
   two paths can be compared along the whole precision axis (the synthetic run
   says the gap peaks near 8-bit and closes by 4-bit -- either confirm or kill
   that).
*Gate G1:* Law 2 reproduces the fc output SQNR within ~1 dB on real data, and
`blocks_with_le_1_eff_bit` at 8-bit is >= ~4 of 9. If Law 2 misses badly, the
block-diagonal assumption is what broke -- report the audit ratios and revise
before spending GPU time.
*Cost:* ~1 h CPU for 8k tokens; a few GPU-minutes if the hidden states have to
be re-captured from Qwen3-8B.

**S2 -- SQNR -> damage transfer (fake-quant, no QAT; 1-3 GPU-days).**
The step that turns dB into acceptance. Four measurements, in order of value:
1. **Dose-response.** Sweep activation/weight bits; measure acceptance length
   (and draft-target top-1 agreement as a cheap offline proxy) for both fusion
   modes. Overlay SQNR(bits) from S1.
2. **The annihilation ablation (decisive).** In fc mode, instead of quantizing,
   *zero* the blocks with `b_eff <= 1` (and leave the rest exact). If zeroing
   alone reproduces most of the w8a8 decay, the mechanism is information
   destruction, not noise -- which is the framework's central claim and the
   cleanest possible discriminator.
3. **The no-context floor.** Zero the whole target feature. This is the worst
   case QAT cannot beat; it calibrates whether w8a8 fc acceptance sits near the
   floor (annihilated) or far above it (merely noisy).
4. **The intervention check (cheap, no retrain).** Give the concat per-block
   scales (one scale per 4096-block, i.e. group-wise activation quantization
   with group = one target layer) and re-measure. The framework predicts this
   alone recovers most of the 8-bit gap.
*Gate G2:* either one monotone function of SQNR fits **both** modes (SQNR is
sufficient -- then the whole story is quantitative and the report writes itself),
or the modes need different curves (SQNR is necessary but not sufficient -- then
`destroyed_signal_fraction` is the missing second axis and the two-number
framing of §2 in the analysis is the result).

**S3 -- QAT + acceptance validation (the expensive part; existing pipeline).**
1. Reproduce the fc w8a8 >10% acceptance decay under the current recipe.
2. Apply the S2 winner, in cheapest-first order: per-block (group-wise) scales on
   the concat -> exclude `fc` activation quantization (w8a16 input) -> rotation
   (Hadamard/QuaRot-style) on the concat blocks -> flare (the known-good control).
3. Success criterion: fc + intervention at w8a8 within ~2% of the BF16
   acceptance, and its measured SQNR within ~3 dB of the flare path's.
4. Write the result as a rule, not an anecdote: *quantize after the mixing, or
   give the mix its own per-block scales; never share one per-token scale across
   a concatenation of heteroscaled tensors.*
*Cost:* bounded by your existing QAT loop; the interventions are config-level.

**S4 -- presentation.** See §3.

## 2. Decisions I need from you

**D-1 Which repo is authoritative?** I found two: `my-specforge` (branch `main`,
holds the probes, `ANALYSIS.md`, the dflare configs, `specforge/layers/wxay.py`,
`core/domino.py`) and `SpecForge` (branch `codex/DRAFTv2`, recent commits on the
folded readout / routed outer experts, and its own `probes/domino_service_npu/`).
v0 is written into `my-specforge/probes/` because that is where the evidence base
for these experiments lives. Confirm, or tell me to port to `SpecForge` / a
different branch.

**D-2 Hidden states for S1.** Options: (a) reuse hidden states captured by the
earlier NPU run if they were persisted; (b) re-capture from Qwen3-8B with
`--target-model-path` using the *same corpus the QAT run trained on* (most
representative); (c) capture from 2-3 domains (chat/code/math) to test whether
the SQNR verdict is corpus-dependent. My recommendation: (b) for the headline,
(c) as a robustness row.

**D-3 Do you have an fc-mode trained checkpoint?** `pi` of the *trained* fc is
prediction 3 and needs one; a flare checkpoint has no `fc.weight` (the probe warns
and falls back to random init). If none exists, we can (i) drop prediction 3,
(ii) take `pi` from the fc model at the *start* of the failed w8a8 run if a step-0
snapshot exists, or (iii) read `pi` from a short fc-mode warmup. Which is available?

**D-4 Acceptance harness.** S2's dose-response needs acceptance length on
quantized checkpoints. `SpecForge/probes/domino_service_npu/` has
`check_acceptance_replacement_rate.py` and `probe_acceptance_over_generation.py`;
which one is the current ground truth, and is it usable at this stage?

## 3. How should the full scale be presented?

* **P-1 Lab notebook** -- extend `probes/ANALYSIS.md` + `probes/README.md` +
  JSON reports in `probes/reports/`, register the probe in
  `run_all_probes.py` so the ledger is one command. Matches the existing house
  style; cheapest; good for continuing work.
* **P-2 Technical report** -- new `reports/sqnr-fusion-<date>.md`: theory
  section (the five laws with their proofs sketched), the real-data ledger
  tables, dose-response and SQNR-vs-bits figures (matplotlib PNG in
  `reports/figures/`), the intervention result, reproducibility appendix with
  exact commands.
* **P-3 Both** -- P-1 as the working record plus P-2 as the readable artefact,
  and `run_all_probes.py` gaining a `--only sqnr_budget` entry so anyone can
  regenerate every number in the report.

My recommendation: **P-3**, with P-2 written only after G2 -- the report is worth
writing once we know whether SQNR alone is sufficient, because that changes the
headline from "the fc concat is 18 dB worse" to a quantified two-axis account.

## 4. Open technical questions worth deciding early

1. **Which quantity is the dependent variable?** S1a has already narrowed this.
   The energy aggregates (energy SQNR, energy-weighted destroyed fraction) are
   *structurally blind* to the fc failure: with the official trained `fc` putting
   `pi = (0.492, 0.409, 0.100)` on `(L1, L17, L33)`, driving the quiet-tap ratio
   `q` from 0.02 to 0.01 at 8-bit makes the aggregate SQNR *rise* 26.5 -> 30.6 dB
   while the highest-`pi` pathway drops below **zero effective bits**. So S2
   should treat as competing predictors: (a) energy SQNR, (b) per-token-mean
   SQNR, (c) **`min_pi_weighted_effective_bits`**, (d) **per-block functional
   SNR**, and (e) the per-token SQNR *lower tail* (p1/p5), since token-localized
   spikes mean the mean and the tail disagree by ~5 dB. Fit all five against
   acceptance length and keep the winner; this is a one-run experiment and it
   decides the report's headline metric.
2. **Is the bridge sited at the concat or at the KV input?** The quantized point
   is the concat, but the damage lands on `k_proj_target`/`v_proj_target` inputs.
   Law 2 covers the transfer; S2 should confirm there is no additional
   amplification inside the KV projections (their weights are per-channel
   quantized, so `pi`-weighting applies again).
3. **Does the annihilation regime have a training-side fix?** If content is
   destroyed rather than noised, then a spike/scale regularizer on the concat (or
   a per-layer whitening ahead of `fc`) is a design lever that pure quantizer
   work cannot reach. Worth one ablation in S3 once S2 establishes which
   mechanism dominates.
4. **Does the framework predict `embed_proj`?** Its input is the other
   concat-into-matmul in the model, and it is the only QAT exclusion. Running the
   ledger on it is nearly free and is a strong out-of-sample test of the whole
   framework.
5. **Noise level vs SENSITIVITY -- the gap in everything measured so far.**
   Every number produced so far (SQNR, effective bits, destroyed fraction)
   measures how large the perturbation is. None of them measures how far
   acceptance moves per unit of perturbation. The two are different axes, and
   the reported evidence points at the second: if excluding `fc` removes a noise
   source and acceptance does not move, the loss is dominated by sensitivity.
   Concretely: acceptance loss ~ (noise) x (sensitivity), and `down_proj`/`o_proj`
   are the two points known to be *noisy* (un-normalized inputs), while the
   fc-mode graph is the suspect for being *sensitive*.
   *What to run instead of another SQNR map:* a **sensitivity map**. Perturb one
   quantized point at a time at a fixed relative perturbation (or sweep bits) and
   measure the change in the draft's output distribution / acceptance -- per
   point, in both fusion modes, including `down_proj` input, `o_proj` input and
   the target-conditioning feature. `probe_layer_quant_sensitivity.py` already
   does this at layer granularity; extending it to per-point answers both
   "which points are noisy" and "which points are consequential", which the SQNR
   map cannot separate.
6. **Are `down_proj`/`o_proj` and the fc collapse the same cause?** Same *family*
   (a matmul whose activation input is not normalized before the per-token
   quantizer: the gated product, the attention output, the multi-block concat),
   and the measured ordering supports the family claim (concat 2.9 dB at 4-bit <
   `down_proj` input ~9 dB < post-norm points ~16 dB, ordering the same as their
   spikiness). But it cannot be the *cause of the mode difference*: both points
   are equally hard in the flare model, which passes. Discriminating experiment:
   apply the fixes already built for those points (SiTU-GLU for `down_proj`,
   post-sublayer norm for `o_proj`) to the **fc-mode** model and retrain briefly.
   Acceptance recovers -> same root cause; it does not -> the fc-mode graph is
   the cause and the sensitivity framing in item 5 is the right one.
7. **Before any of the above:** confirm the `fc` exclusion actually matched.
   `replace_linear_with_quantized` prints `[QAT] <path>: skipped (excluded)` on a
   hit and only *warns* (`qat_exclude names not found in model`) on a miss, so a
   mis-typed name silently excludes nothing -- which would invalidate the
   "excluding fc does not help" result that the retraction rests on.

## 5. Immediate next actions (on your go-ahead)

1. Confirm D-1..D-4.
2. S1 run on the server:
   `python probes/probe_sqnr_budget.py --config configs/qwen3-8b-domino-dflare-verifiedBase.json --target-model-path <Qwen3-8B> --draft-ckpt <flare ckpt> --B 32 --S 256`
   plus the same with `--draft-ckpt <fc ckpt>` (and without, to get the random-`pi`
   baseline).
3. Report the S1 tables back here; I revise `SQNR_ANALYSIS_V0.md` into the S1
   version and we pick the S2 intervention order from the gate results.

## 5. Immediate next actions (on your go-ahead)

1. Confirm D-1..D-4.
2. S1 run on the server:
   `python probes/probe_sqnr_budget.py --config configs/qwen3-8b-domino-dflare-verifiedBase.json --target-model-path <Qwen3-8B> --draft-ckpt <flare ckpt> --B 32 --S 256`
   plus the same with `--draft-ckpt <fc ckpt>` (and without, to get the random-`pi`
   baseline).
3. Report the S1 tables back here; I revise `SQNR_ANALYSIS_V0.md` into the S1
   version and we pick the S2 intervention order from the gate results.
