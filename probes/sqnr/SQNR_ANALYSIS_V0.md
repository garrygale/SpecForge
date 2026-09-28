# SQNR lens on DFlash-vs-DFlare target-hidden fusion (v0, short version)

Why the convex (`fusion_mode="flare"`) target-hidden fusion makes the Domino
draft model quantizable -- including w4a8 with a w4a4 subset -- while the
matmul (`fusion_mode="fc"`) fusion loses **>10% acceptance length even at w8a8
QAT**.

This is the *short* version: the lens, the laws it rests on, and the numbers the
laws already produce. It deliberately stops short of the training-scale
experiment; see `SQNR_PLAN.md` for how the full study should run and be
presented. Probe: `probe_sqnr_budget.py`. Reports: `reports/sqnr_laws.json`,
`reports/sqnr_budget.json`.

---

## 1. The two paths, exactly as implemented

From `specforge/modeling/draft/dflash.py` (config
`configs/qwen3-8b-domino-dflare-verifiedBase.json`: T=9 target layers
`[1,5,9,13,17,21,25,29,33]`, `target_hidden_size`=4096, draft `hidden_size`=2560,
7 draft layers, `heterogeneous_kv`=true):

```
flare (line 1397):  layer_target_i = hidden_norm( sum_j softmax(W_fuse[i])_j * h_j )   # (B,S,4096)
fc    (line 1292):  target_hidden  = hidden_norm( fc( concat_j h_j ) )                 # fc: Linear(36864 -> 2560)
```

Both then feed the KV projections (flare: `k_proj_target`/`v_proj_target` over
4096 dims; fc: the shared `k_proj`/`v_proj` over 2560 dims).

The quantizer (`specforge/layers/wxay.py`) is **per-token symmetric**:

```python
qmax  = 2**(bits-1) - 1
scale = x.abs().amax(dim=-1, keepdim=True) / qmax      # one scale per TOKEN
x_q   = round(x / scale).clamp(-qmax, qmax) * scale
```

and `replace_linear_with_quantized()` swaps **every** `nn.Linear` except
`qat_exclude` (only `embed_proj` here). The fusion operators are `nn.Linear` in
one case and a softmax weighted sum in the other -- so this single line decides
*where the quantizer sits relative to the multi-layer mixing*:

| | fc | flare |
|---|---|---|
| quantized tensor | the **pre-mixing concat** of 9 blocks, one shared per-token scale | the **post-mixing** fused vector, single block |
| blocks sharing a scale | 9 (RMS disparity 20x synthetic; ~5e4 reported on real data) | 1 |
| fusion operator | unconstrained matmul (can amplify) | convex combination (contraction) |
| per-draft-layer feature | one shared feature | its own routed feature |

That difference -- not the matmul arithmetic per se -- is the whole effect.

## 2. The lens: two numbers, not one

The per-token quantizer gives **every** coordinate the same absolute step
`s_t = max_t / qmax`, where `max_t` is the max over the whole token vector.
Therefore:

**(2.1) Energy SQNR** -- the aggregate that governs downstream error energy is

```
SQNR_E = 10 log10 ( sum_t ||x_t||^2 / sum_t ||x_t - Q(x_t)||^2 )
```

Aggregated by **power**, not by averaging per-token dB values. These are not
interchangeable: per-token spikiness is heavy-tailed, so `E[1/spik^2] !=
1/E[spik]^2`. Measured on the synthetic concat at 8-bit: energy SQNR 20.6 dB vs
per-token-mean SQNR 26.0 dB -- **5.4 dB of flattery**. (The existing
`summarize_tensor` reports the per-token mean; both are now reported side by
side, since `probes/ANALYSIS.md` leaned on the mean.)

**(2.2) Destroyed-signal fraction** -- the share of content that is not merely
noisy but *gone*: coordinates inside the dead zone `|x| < s_t/2` round to zero.
Measured on the synthetic concat: 0.62% at 8-bit, 23.0% at 4-bit. This is the
number SQNR cannot express, and it is the one that matters once blocks are
annihilated (Law 1, third tensor: 98% of coordinates zeroed while the SQNR still
reads a healthy 18.1 dB).

**Effective bits per block.** With `r_{t,j}` the RMS of block `j` on token `t`:

```
b_eff,j = log2( 2 r_{t,j} / s_t ) = bits - log2(max_t / r_{t,j})
        = [bits - log2(max_t / r_max,t)] - log2(r_max,t / r_{t,j})
          \______ spikiness tax ______/   \__ block-disparity tax __/
```

A block with `b_eff <= 1` has at most one reliable bit. The spikiness tax is paid
by whatever tensor the quantizer sees, so it is unavoidable; the **disparity tax
exists only when several blocks must share one scale -- it is what the fc fusion
buys and flare does not pay**. Synthetic ledger: spikiness tax 3.47 bits;
disparity tax 4.3 bits for the quietest block down to 0.0 for the loudest; at
8-bit, 1.61 of 9 blocks sit at `b_eff <= 1` (8.98 of 9 at 4-bit).

## 3. The laws, and how well they hold

All checked on synthetic ground truth by `--laws` (`reports/sqnr_laws.json`).

**Law 1 -- noise floor.** For surviving coordinates the rounding error is
statistically uniform on `(-s_t/2, s_t/2)`, so `E[e^2] <= s_t^2/12` per
coordinate. Measured noise/(s²/12) = **1.000** on Gaussian input at 4 and 8 bits
(dead-zone coordinate fraction 21% at 4-bit -- so the bound is tight, not
conservative). *Validity domain:* when a token is spike-dominated, the same
quantity collapses to 0.015 and 99.98% of coordinates are zeroed while the SQNR
still reads 10.3 dB. **SQNR stops being informative in the annihilation
regime** -- hence 2.2.

**Law 2 -- matmul transfer.** With `pi_j = ||W_j||_F^2 / ||W||_F^2` the
column-**block** energy share of `W`, a matmul performs an energy-weighted
average of per-block quality: it cannot repair a destroyed block and it dilutes
good blocks by bad ones. Measured against the partial-survival model
(per token/block, empirical `p` = surviving-coordinate fraction, `z` = energy
fraction inside the dead zone):

```
signal = sum_{t,j} pi_j E_{t,j} (1 - z_{t,j})                 retained content
noise  = sum_{t,j} pi_j ( p_{t,j} H s_t^2/12 + z_{t,j} E_{t,j} )   rounding + zeroing
SQNR   = signal / noise
```

| W | bits | SQNR in (concat) | SQNR out measured | law | error |
|---|---|---|---|---|---|
| random uniform | 8 | 37.59 | 37.62 | 37.60 | **-0.01 dB** |
| energy on quiet block | 8 | 37.59 | 25.07 | 25.05 | **-0.02 dB** |
| energy on loud block | 8 | 37.59 | 41.94 | 41.95 | **+0.02 dB** |
| random uniform | 4 | 13.80 | 13.80 | 13.89 | +0.09 dB |
| energy on quiet block | 4 | 13.80 | 6.54 | 5.58 | -0.96 dB |

The audit ratios (`signal_meas/model`, `noise_meas/model`) land at 1.00-1.04.
The law is accurate to **<0.1 dB** outside the annihilation regime and <1 dB
inside it. Two consequences that matter:

* **The fc output SQNR is set by where the trained `fc` puts its column
  energy** -- 25.05 dB vs 41.95 dB at 8-bit for two W's with identical input
  SQNR (37.59 dB). A 17 dB swing from the weight layout alone.
* At 8-bit with energy on the quiet block, `pi`-weighting drives the *output* to
  **12.5 dB below the concat's own SQNR** -- the quantized point looks better
  than the transmitted feature.

**Law 3 -- concat penalty.** `SQNR_E(concat) ≈ 12 qmax^2 mean_j(r_j^2) /
mean_t(max_t^2)`. Measured vs law, synthetic concat, all rows:

| disparity | bits | measured | law | error |
|---|---|---|---|---|
| 1x | 4 / 8 | 16.00 / 41.17 | 15.98 / 41.17 | -0.01 / -0.00 dB |
| 10x | 4 / 8 | 11.84 / 36.59 | 11.88 / 36.59 | +0.04 / +0.00 dB |
| 100x | 4 / 8 | 12.35 / 34.72 | 12.38 / 34.79 | +0.03 / +0.07 dB |

**Law 4 -- convex fusion: a contraction, and dilution only for incoherent
spikes.** For `w` on the simplex, `max_i |y_i| <= max_j max_i |h_{j,i}|` --
verified (max amplification 1.000 one-hot, 0.223 uniform). Dilution is
`20 log10(||w||_2 / w_max)`, and it exists **only if the layers spike at
independent (token, dim) pairs**:

| weights | spikes | predicted | measured |
|---|---|---|---|
| uniform | independent | +9.54 dB | **+9.35 dB** |
| uniform | coherent (same dim) | +9.54 dB | **-0.17 dB** |
| peaked 0.85 | independent | +0.02 dB | +0.02 dB |

This corrects the dilution story, and it corrects an earlier version of this
document. The trained fusion weights have entropy 0.56-1.69 nats (uniform
= 2.20). Under the uniform-over-`k` model the *available* dilution is
`10 log10(k) = 4.343 H` dB, so the trained weights could in principle access
**2.43 to 7.34 dB** -- not the <1 dB an earlier draft claimed. The random-init
schedule (`dilution_gain_pred_db_per_layer` = 0.59 dB, `w_max/||w||_2` = 0.934)
is the peaked case, not the trained one.

So why is the measured `sqnr4_gain_vs_dominant_source` only -1.4..+0.3 dB
(`probes/ANALYSIS.md` §8.1)? Because the incoherence condition fails: the
synthetic check shows the same uniform weights give **+9.35 dB for independent
spikes and -0.17 dB for coherent spikes**. Residual-stream layers spike at
correlated positions, so the dilution that the weight spread would license is not
collectable. **Flare's realised advantage is quantizer placement and per-layer
routing, not smoothing** -- dilution is a potential, and
`inter_layer_cosine_mean` is the diagnostic that decides how much of it a given
checkpoint can access (synthetic independent layers: 0.000; real residual streams
are strongly correlated, so expect behaviour near the coherent case).

**Law 5 -- RMSNorm is per-token SQNR-invariant.** Measured per-token mean at
4 and 8 bits: raw 4.762314796447754 vs normed 4.762314796447754 -- identical to
the last digit. The energy aggregate can still drift ~1.5 dB when per-token RMS
varies. Practical reading: what matters is *where the norm sits relative to
where the quantizer sits*, which is why fc's quantized concat is untouched by
the `hidden_norm` that follows it.

## 4. The budget: fc vs flare in the same units

Synthetic target hidden states, random-init weights
(`probe_sqnr_budget.py --config ... --random --B 8 --S 128`); 8.9k tokens,
structural not trained:

| | 8-bit | 4-bit |
|---|---|---|
| fc, SQNR at the quantized point (concat) | 20.61 dB | 5.05 dB |
| fc, SQNR at the output (what the KV projections receive) | 20.57 dB (law 20.73) | 5.03 dB (law 4.20) |
| fc, relative distortion of the conditioning feature | **9.4%** | 56.0% |
| fc, destroyed-signal fraction | 0.62% | 23.0% |
| fc, blocks at `b_eff <= 1` (of 9) | 1.61 | 8.98 |
| flare, fused SQNR (mean over 7 draft layers) | **32.23 dB** | 8.91 dB |
| flare, relative distortion | **2.4%** | 35.8% |
| flare, destroyed-signal fraction | 0.011% | 6.8% |
| **gap (flare - fc output)** | **+11.66 dB** | +3.88 dB |

Two things to read off this table:

1. **The gap is largest at 8-bit, not at 4-bit.** At 4-bit both paths collapse
   (flare's own spikiness tax, 3.47 bits, eats most of the 4 bits available), so
   the *mode* difference compresses. This is a non-obvious prediction: the
   fc-vs-flare separation is a mid-precision phenomenon, which is consistent
   with fc failing at w8a8 while flare is comfortable there.
2. **The per-token-mean aggregate hides the damage.** At 8-bit the concat's
   per-token mean is 26.0 dB vs 20.6 dB energy SQNR. A probe reading only
   `sqnr_8bit_mean` would report the fc path as 6 dB healthier than it is.

## 5. Why this explains ">10% acceptance decay at w8a8"

Prior measurements (`probes/ANALYSIS.md` §8, NPU run, QAT bf16 checkpoint,
w4a4 on gate+up@{0,2,4} and q/k/v/k_proj_target/v_proj_target@L0): concat
sqnr8 = **17.7 dB**, fused = **36.1 dB** (an 18.4 dB gap, matching the +11.7 dB
the synthetic run predicts in sign and order of magnitude). Chain:

1. `fc` is an `nn.Linear`, so w8a8 QAT quantizes the concat. The per-token scale
   is spent on the loudest block; quiet blocks lose `log2(r_max/r_j)` bits each
   (Law 3). At 17.7 dB the fc *output* carries **~13% relative distortion**
   (`10^(-17.7/20) = 0.13`).
2. That output is the **only conditioning path** for the target context, shared
   by all 7 draft layers, after which it becomes K and V for every draft
   position. 13% relative distortion is roughly 3-bit fidelity on the
   information the draft exists to condition on.
3. `hidden_norm` does not help (Law 5): the norm is downstream of the quantizer,
   so it rescales signal and noise together.
4. QAT can partly rescue *SQNR* -- by moving `pi` toward the loud, well-quantized
   blocks (Law 2: a 17 dB lever). But that rescue is paid for in
   `destroyed_signal_fraction`: the quiet layers' content is discarded, not
   cleaned. So QAT narrows the damage without restoring the missing conditioning
   -- which is precisely a residual acceptance loss after adaptation rather than
   a divergence.
5. Flare never quantizes the concat at all: its quantized point is the
   post-mixing, single-block, unit-scale fused feature (36.1 dB real / 32.2 dB
   synthetic), i.e. **~2.4-1.6% distortion**, and each draft layer gets its own
   routed feature. w4a8 is comfortable, and the spare budget buys the w4a4 set.

Corroborating structural evidence already in the repo: the *other* concat-into-
matmul point in this model is `embed_proj` in the Domino correction head
(`specforge/core/domino.py:220`, concatenating draft hidden + GRU prefix states +
fused target), and it is **the only module excluded from QAT** (`qat_exclude:
["embed_proj"]`). The two places where a matmul consumes a concatenation of
heterogeneously-scaled tensors are exactly the two known pain points.

## 6. Falsifiable predictions

If the framework is right, the real-data ledger (S1) must show:

1. `blocks_with_le_1_eff_bit` >= ~4 of 9 on the real concat at 8-bit, and the
   destroyed-signal fraction at 8-bit is percent-level or above -- not a
   noise-only story.
2. `fc_output_sqnr` predicted by Law 2 within ~1 dB of measured, on the real
   checkpoint, at 4 and 8 bits.
3. **The trained fc checkpoint's `pi` is shifted toward the loud (high-RMS)
   target layers relative to the uniform 1/T random-init value**, and the shift
   size correlates with how much acceptance QAT recovered. (Testable from an
   existing checkpoint -- no retraining.)
4. `inter_layer_cosine_mean` on real hidden states is high (>= ~0.5), confirming
   that dilution is unavailable and the measured ~0 dB fusion gain is expected.
5. Giving the concat **per-block scales** (group-wise activation quantization,
   one scale per 4096-block instead of per token) should recover most of the gap
   at w8a8 with no architecture change -- the sharpest single prediction.

## 7. Limitations of v0 (what is deliberately not claimed)

* **No acceptance-length link yet.** v0 quantifies damage to the conditioning
  signal. Turning SQNR into predicted acceptance length needs the fake-quant
  dose-response and the acceptance harness (S2). Do not read the dB gap as a
  percentage.
* **Synthetic hidden states and random-init weights.** The numbers in §4 are
  structural (scale disparity 20x, independent per-layer spikes). The real
  disparity is reported as ~5e4, which would make the fc ledger *worse*, not
  better -- but that figure needs confirming in S1 before it is quoted.
* **`pi` is unavailable for a trained fc checkpoint locally.** With
  `fusion_mode="fc"` a flare checkpoint has no `fc.weight`, and the probe prints
  a warning and falls back to random init; `fc_weight_from_checkpoint` records
  which happened.
* **Law 2 assumes the input covariance is block-diagonal** (cross-block terms
  dropped) and that surviving coordinates round uniformly. The audit ratios
  expose both; they read 1.00-1.04 in the checks, but should be re-read on real
  data.
* **The optimizer is out of scope.** v0 says nothing about whether QAT *should*
  have been able to adapt around the noise; it only measures the information
  available to adapt to.

---

# S1a -- ledger of the OFFICIAL trained fusion weights

Added after v0. `analyze_fusion_weights.py` + `fetch_hf_tensors.py` read the two
official Qwen3-8B drafts directly from HuggingFace (no target model, no GPU, no
full checkpoint download: the safetensors header gives byte ranges, so only
`layer_fusion_weights` [7,9] and `fc.weight` [4096,12288] are transferred).

* DFlash (fc fusion): [`shanjiaz/dflash-qwen3-8b`](https://huggingface.co/shanjiaz/dflash-qwen3-8b),
  T=3, `target_layer_ids=[1,17,33]`.
* DFlare (flare fusion): [`AngelSlim/Qwen3-8b-dflare`](https://huggingface.co/AngelSlim/Qwen3-8b-dflare),
  T=9, `target_layer_ids=[1,5,...,33]`.

## S1a.1 DFlare's trained fusion is a router, not a mixer

| draft L | entropy (nats) | `w_max/||w||_2` | available dilution | dominant tap |
|---|---|---|---|---|
| 0 | 1.828 | 0.660 | +3.61 dB | L21 |
| 1 | 1.470 | 0.919 | +0.73 dB | L21 |
| 2 | 0.994 | 0.937 | +0.56 dB | L21 |
| 3 | 1.484 | 0.900 | +0.91 dB | L1 |
| 4 | 1.187 | 0.963 | +0.33 dB | L1 |
| 5 | 1.341 | 0.873 | +1.18 dB | L1 |
| 6 | 1.101 | 0.898 | +0.93 dB | L33 |

Uniform reference: 2.197 nats, +9.54 dB. **Mean available dilution is +1.18 dB**
(max +3.61 dB), and the seven draft layers route almost exclusively to just
**three of the nine taps: L1, L21, L33** -- the same shallow/mid/deep structure
DFlash chose independently (`[1,17,33]`).

So the trained fusion is closer to a hard router than to a mixer, and Law 4's
dilution is worth ~1 dB at most -- and ~0 dB once the incoherence condition is
accounted for. This settles §3's corrected reading with trained numbers: **flare's
quantization advantage comes from quantizer placement (the concat is never
quantized) plus per-layer routing, not from convex averaging.**

## S1a.2 DFlash's trained `fc` reads its energy from the taps that quantization destroys

`pi_j` over the three column blocks, versus uniform 1/3:

| block | target layer | `pi_j` | `||W_j||_F` | spectral norm |
|---|---|---|---|---|
| 0 | L1 (shallowest) | **0.4919** | 825.2 | 266.3 |
| 1 | L17 | **0.4085** | 752.1 | 151.3 |
| 2 | L33 (deepest) | 0.0995 | 371.2 | 42.8 |

Row norms spread 3.73x, `cond(W)` = 886, `||W||_F/||W||_2` = 4.24.

The BF16 task optimum puts **90% of the column energy on the two shallow taps
(L1, L17)** and only 10% on the deepest one. In a residual stream the deepest tap
carries the largest RMS and therefore sets the per-token maximum `max_t` -- which
is exactly what the shared per-token scale is spent on. So the trained `fc` reads
mostly from the *quiet* blocks, and the quiet blocks are the ones the quantizer
starves.

## S1a.3 The aggregate SQNR is blind to this -- a correction to v0's headline metric

Applying Law 2 with an assumed geometric depth ramp (`q` = quietest/loudest block
RMS ratio, `rho = max_t/sigma_loudest = 4`) at 8-bit:

| q | aggregate SQNR | destroyed (energy) | L1 pathway: pi / eff bits / local SNR |
|---|---|---|---|
| 0.50 | 37.13 dB | 0.0000 | 0.49 / 5.0 bits / 34.8 dB |
| 0.20 | 33.86 dB | 0.0000 | 0.49 / 3.7 bits / 26.8 dB |
| 0.05 | 31.66 dB | 0.0000 | 0.49 / **1.7 bits** / 14.8 dB |
| 0.02 | **26.54 dB** | 0.0018 | 0.49 / **0.3 bits** / 6.8 dB |
| 0.01 | **30.55 dB** | 0.0005 | 0.49 / **-0.7 bits** / 0.8 dB |

Read the last two rows together: **as the quiet block is annihilated more
thoroughly, the aggregate SQNR gets *better* (26.5 -> 30.6 dB) and the
energy-weighted destroyed fraction gets *smaller* (0.0018 -> 0.0005) -- because
the destroyed content was low-energy.** Meanwhile the pathway carrying half the
trained `fc`'s column energy goes from 0.3 effective bits to *zero*. Both
energy-based aggregates in §2 are structurally blind to this regime; energy is
the wrong currency for a pathway that carries little energy but decisive
information.

The metrics that do not have this blind spot, and which S2 should fit against
acceptance length:

1. **`min_pi_weighted_effective_bits`** -- the effective bit width of the
   highest-`pi` taps (0.3 bits at q=0.02, -0.7 at q=0.01, at a nominal 8).
2. **per-block functional SNR** -- `12 qmax^2 u_j^2 / rho^2`, i.e. how wrong each
   pathway's *own* contribution is, which no global energy average can dilute.

This is the sharpest statement the framework can currently make about "even w8a8
QAT loses >10% acceptance length": at w8a8 the concat's aggregate SQNR can read a
comfortable 26-31 dB while the tap supplying ~half of the trained `fc`'s input
energy has **less than one effective bit**. A QAT run can compensate for noise; it
cannot recover content that was never transmitted.

**Pending (S1b):** `q` and `rho` are assumed here, not measured. They require the
real per-layer hidden-state RMS profile of Qwen3-8B, which needs the target model
(or previously captured states). Once `q` and `rho` are pinned, this bracket
collapses to a single predicted number per bit width and prediction 2 becomes a
hard test of Law 2 on real data.

## S1a.4 A fetch bug worth recording

The first version of `fetch_hf_tensors.py` read safetensors `data_offsets`
relative to byte 8, but they are relative to the **end of the header**
(`8 + header_len`). The leading tensors were therefore decoded from the header
JSON itself: `hidden_norm.weight` -- which must be ~1.0 -- read as 1e37 with
`mean=inf`, and the identical bogus magnitude appeared in several tensors. The
sanity check on a known-value tensor is what caught it, and the fix is in the
file. Any future range-fetch of a safetensors file should re-run that check
(`hidden_norm.weight` mean ~0.8-1.0) before trusting the numbers.

---

# S1b -- RETRACTION: the fusion operator is not the cause

Two things killed the §4/§5 explanation of the fc-mode acceptance collapse. Both
are recorded here so the earlier sections are not read as the current position.

## Retraction 1 -- excluding `fc` from quantization does not help (reported result)

If `fc` is excluded from QAT and everything else runs w8a8, fc-mode acceptance
still collapses. When `fc` is excluded the concat never reaches a quantizer, so
the "the per-token scale spends itself on the loudest block" mechanism cannot be
the cause of the collapse. Within their own codebase both fusion modes are
config-switchable on one architecture, so this is the cleanest available
comparison and it has to be believed over the analysis.

## Retraction 2 -- the fc OUTPUT is *quantization-friendlier* than the flare output

Measured directly, using the two **real trained operators** (official
`z-lab/Qwen3-8B-DFlash-b16` `fc.weight` [4096,20480] and the trained DFlare
`layer_fusion_weights` [7,9]) applied to the same synthetic spiky target hidden
states, post-RMSNorm, at the point each consumer actually quantizes:

| operator | spikiness | SQNR 4-bit | SQNR 8-bit |
|---|---|---|---|
| flare (trained softmax, convex) | 10.4-11.0 | 7.0-9.5 dB | 30.5-32.2 dB |
| fc (official trained matmul) | **4.69** | **13.9 dB** | **39.1 dB** |
| source target layers | 10.5-11.0 | 8.6-9.0 dB | 25.6-25.8 dB |
| fc: `||W||_inf` (max row L1) | 9017 | -- | -- |

The fc output is smoother and ~7-8 dB *better* in SQNR than the flare fused
feature, and better than the raw source layers. Law 5 (per-token SQNR
invariance under RMSNorm) makes this a fair comparison at the consumption point,
so the fusion output is not the problem either. The `||W||_inf` = 9017 worst-case
amplification budget is a vacuous bound over 20480 inputs and should **not** be
quoted as a mechanism -- the empirical measurement above is what counts.

## Where this leaves the investigation

Nothing about the fusion *operator* -- input or output -- explains the
acceptance collapse. The collapse must come from something the fusion mode
changes **downstream**, and every comparison so far confounds at least these two
switches (they are tied together by the `heterogeneous_kv` flag):

1. **Shared vs dedicated KV projections.** fc mode shares `k_proj`/`v_proj`
   between the draft stream and the target context; flare mode has dedicated
   `k_proj_target`/`v_proj_target`. With shared weights, QAT cannot rebalance the
   relative gain between the draft-stream term and the target-context term -- one
   matrix serves both. With dedicated weights the conditioning path gets its own
   parameters and can be trained quantization-robust.
2. **One fused feature reused by every draft layer vs a per-layer feature.**
   fc hands the same vector to all layers, so any corruption of it is
   common-mode across layers and cannot average out; flare gives each layer its
   own routed read, so errors across layers are closer to independent.

Both are config-level, and in the official z-lab config (`hidden_size` ==
`target_hidden_size` == 4096) the matmul fusion *can* be paired with
`heterogeneous_kv=True`, which is not possible in the local 8B domino config
where `hidden_size`=2560. That yields the experiment that actually discriminates:

| | shared KV | dedicated KV |
|---|---|---|
| **matmul fusion** | A = official DFlash (fails) | B = `heterogeneous_kv=True` on DFlash |
| **convex/router fusion** | C = flare with `heterogeneous_kv=False` | D = domino-DFlare (works) |

If B passes at w8a8, the culprit is the shared KV path, not the fusion operator.
If B still fails while C passes, the culprit is the matmul operator's *effect on
how the conditioning is consumed*, and the question becomes why a single shared
feature is fragile rather than why a matmul is.

Until that table is run, the honest position is: **the SQNR budget of the fusion
path does not explain the DFlash fc-mode collapse; it is a property of the graph
the fc mode forces, and the two candidate graph properties are still
confounded.** The machinery built here (per-token effective bits, the
partial-survival law, per-pathway SNR) is still the right toolkit -- it just has
to be pointed at the context-KV path and the attention output instead of at the
fusion.

## Artifact caveat

`shanjiaz/dflash-qwen3-8b` (3 layers, taps `[1,17,33]`) is a **community** upload
and was used for the first version of the S1a numbers. The official DFlash
release is [`z-lab/Qwen3-8B-DFlash-b16`](https://huggingface.co/z-lab/Qwen3-8B-DFlash-b16):
5 layers, taps `[1,9,17,25,33]`, `hidden_size` 4096, full attention,
`fc.weight` [4096,20480]. Official `pi` over those taps is
`(0.3727, 0.1868, 0.2524, 0.1503, 0.0378)` -- still strongly non-uniform, with
the deepest tap at 3.8%, but that finding is now only a property of the trained
fusion weights, not an explanation of any acceptance number.

Related public artifacts worth knowing about: `SubSir/QAT-Qwen3-4B-DFLASH`
(NVFP4 QAT, `exclude_modules: []`, fc quantized) and
`SubSir/QAT-Qwen3-4B-DFLASH-wofc` (NVFP4 QAT, `exclude_modules: ["fc"]`) -- the
same with/without-`fc` split, published by someone else, with no acceptance
numbers attached.

---

# S1c -- the fc-input scale spread, and the retraction of the ~3x inference

The S1a text above inferred the per-tap input RMS ratio from the trained `fc`'s
column energies (~3x). **That inference is wrong** and is retracted here.

It assumed training balanced each tap's contribution (``pi_j * rms_j^2``
constant). The trained `fc` does not do that: it gives the deepest, largest-norm
tap the *smallest* column energy (0.038), so the output is dominated by the deep
tap and the column energies say nothing about the input scales.

Measured directly (`probe_residual_scale_profile.py` on Qwen3-0.6B, 28 layers,
taps [1,9,17,25,33] of a 36-layer target mapped to matched relative depth,
padding and attention-sink token removed):

| statistic | value |
|---|---|
| per-token across-tap RMS disparity (median) | **54.6x** |
| same, p10 / p90 | 43.5x / 65.0x |
| ratio of per-tap RMS medians | 52.5x |
| ratio of per-tap RMS means | 86.2x (outlier-dominated -- do not quote) |

Per-tap median RMS grows monotonically with depth: 0.343 / 0.804 / 1.649 / 4.532
/ 18.001 for L1 / L9 / L17 / L25 / L33. Note the distribution is extremely heavy
tailed -- `rms_mean` at the massive-activation onset layer reads 16.0 while
`rms_p90` is 0.489, and 7.8% of tokens carry a tap RMS above 100 -- so **means
over tokens are unusable here** and every statistic above is a median.

## Effective bits per tap under one shared per-token scale

`b_eff = bits - log2(max_t / r)`, with `max_t ~ rho * rms_loudest`, `rho = 15`
(measured per-tap spikiness is 12.9-18.8), and `pi` from the official trained
`fc`:

| tap | rms | pi (column energy) | b_eff @8-bit | tap relative error |
|---|---|---|---|---|
| L1 | 0.343 | **0.3727** | **-1.63** | 179% |
| L9 | 0.804 | **0.1868** | **-0.40** | 76% |
| L17 | 1.649 | **0.2524** | 0.63 | 37% |
| L25 | 4.532 | 0.1503 | 2.09 | 14% |
| L33 | 18.001 | 0.0378 | 4.08 | 3% |

**81.2% of the fc's column energy sits on taps with <= 1 effective bit at w8a8**;
at w4a4 it is 100%. The result is insensitive to `rho`: at `rho = 1` (i.e. no
spikes at all) L1 still only reaches 2.3 bits, and at `rho = 4` it is 0.29.

This is the fc-input failure mode in one table, and it is the ~52x spread -- not
the matmul arithmetic -- that produces it. DFlare has no concat and therefore no
instance of it.

## The caveat that keeps this honest

Those starved taps contribute little **energy** to the fc output (the deep tap is
52x larger), which is why every energy-weighted aggregate in this document --
energy SQNR, energy-weighted destroyed fraction -- looks healthy while 81% of the
column energy sits on 0-bit pathways. With the measured Qwen3-8B profile the
split is: the three starved taps (L1, L9, L17) contribute **7.3%** of the fc
output energy and L33 alone contributes **67.1%**. "81% of column energy" and
"7.3% of output energy" are both true and measure different things: what the
model wanted to read versus how much energy it got. Any acceptance-length metric
should be fitted against the **pathway** metrics
(`min_pi_weighted_effective_bits`, per-tap functional SNR), not the energy ones.

Absolute 52x is from a 0.6B proxy (28 layers vs the 8B's 36); the shape (monotone
growth plus an early massive-activation onset) is what transfers. Confirming the
number on Qwen3-8B is one forward pass with `output_hidden_states=True`.

## CONFIRMED on Qwen3-8B (real target, `--device npu`)

Ran on the server against the real Qwen3-8B. The proxy prediction held:

| | Qwen3-0.6B proxy | **Qwen3-8B (real)** |
|---|---|---|
| tap RMS medians (L1/9/17/25/33) | 0.343 / 0.803 / 1.644 / 4.531 / 18.00 | **0.268 / 0.931 / 1.556 / 4.338 / 13.98** |
| per-token across-tap disparity (median) | 54.6x | **58.5x** |
| `b_eff` @8-bit | -1.63 / -0.40 / 0.63 / 2.09 / 4.08 | **-1.62 / 0.17 / 0.91 / 2.39 / 4.08** |
| `pi` on taps with ``b_eff <= 1`` | 0.812 | **0.812** |

So on the real model, with one shared per-token scale over the fc input:
**three of the five taps have at most one effective bit at a nominal 8, and those
three carry 81.2% of the trained `fc`'s column energy.** L1 (pi = 0.373) sits at
-1.62 bits, inside the dead zone. Even the loudest tap only reaches 4.08 bits,
which is exactly the spikiness tax `8 - log2(rho)` with `rho ~ 15` -- the
spikiness tax is paid by everyone, the disparity tax only by the quiet taps.

Note the `pi` row is identical in both runs by construction (it is a property of
the same `fc.weight`), so it is not evidence; the evidence is the RMS profile and
the resulting `b_eff`.

## The next experiment: separate annihilation from noise, in BF16

`b_eff` predicts that the quantizer annihilates content. It does not yet show
that the annihilation costs acceptance -- the starved taps are also the
low-energy ones. Two PTQ-level ablations (no retraining) settle it, both on the
fc-mode draft:

1. **Annihilation ablation.** In BF16, zero the taps with `b_eff <= 1`
   (L1, L9, L17) in the concat. If acceptance barely moves, the content of those
   taps is not needed and the concat is not the acceptance killer; if it
   collapses, the 0-bit pathways are exactly the damage.
2. **Noise-only ablation.** Keep all taps, but quantize *only* those three blocks
   with the shared per-token scale (i.e. inject the quantization error without
   removing the signal elsewhere). This isolates "the shared scale's noise"
   from "the lost content".

Prediction from this analysis: (1) removes 7.3% of the fc output energy, so it
costs whatever that content is worth -- bounded, but not obviously negligible;
(2) costs more, because the noise those blocks inject is scaled by the *loudest*
tap's step rather than their own. If both come out harmless, the concat is
exonerated as the acceptance cause and the remaining suspect is the graph the fc
mode forces (shared KV vs dedicated, single shared feature vs per-layer routing).

## Actionable consequence

Give each tap (each concat block) its own activation scale -- group-wise
activation quantization with the group equal to one target layer. With per-tap
scales every tap lands at `~8 - log2(rho_tap) ~ 4.09` effective bits, so:

| | L1 | L9 | L17 | L25 | L33 | pi-weighted |
|---|---|---|---|---|---|---|
| b_eff now | -1.62 | 0.17 | 0.91 | 2.39 | 4.08 | **0.171** |
| b_eff with per-tap scales | 4.09 | 4.09 | 4.09 | 4.09 | 4.09 | **4.093** |
| gain | +5.71 | +3.92 | +3.18 | +1.70 | +0.01 | **+3.92 bits (~+23.6 dB)** |

with **no retraining**. Note it does NOT reach 8 bits: per-tap scales remove the
*disparity* tax, while the *spikiness* tax (~3.9 bits at `rho = 15`) remains for
every tap. The alternative (RMSNorm each tap before concatenating) attacks both
but changes the architecture and needs a retrain.




