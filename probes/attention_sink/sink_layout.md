# Sink layout (A1: n=1, GQA)

One learned pair per layer per kv-head: `sink_k, sink_v ∈ R^head_dim`, raw
constants (no k_norm, no RoPE, no input dependence). Per-query visible set
becomes band ∪ own block ∪ {sink}:

    out = λ(q)·softmax_real(q)·V + (1 − λ(q))·v_sink,  λ = S/(S+B),
    B = exp(q·k_sink)

λ senses match quality against the real keys; a static gate is absorbable
(folded-softmax lesson), a q-only gate is key-blind. Zero-init keeps the
sink a zero-logit zero-value entry at init; gradients flow only to diffuse
rows.

## Panel A — what one query sees (training view; compressed: block=4, W=8,
anchor a=20; real config block=16, W>=512)

    KV cols:   …… c13 c14 c15 c16 c17 c18 c19 | A(20) m1(21) m2(22) m3(23) | SINK
              (old ctx that slid out)  band   |   own block (draft slots)   |
    K source:   k_proj_target(fused target h)  | k_proj(draft h)             | learned
    position:   real, roped                    | a..a+3, roped               | none
    ------------------------------------------------------------------------
    q o=0 (20):  .  1  1  1  1  1  1 | 1  1  1  1 | 1
    q o=1 (21):  .  .  1  1  1  1  1 | 1  1  1  1 | 1
    q o=2 (22):  .  .  .  1  1  1  1 | 1  1  1  1 | 1
    q o=3 (23):  .  .  .  .  1  1  1 | 1  1  1  1 | 1
                 ^ left edge = a+o-(W-1), rises with o  | domino block-bidi | always

The band's left edge slides up with the query position; position 0 leaves
every layer once anchor > W. The sink column is the only one no slide can
mask. Training packing: KV rows = [0..S-1] ctx | [S..S+N*16-1] blocks |
[tail n_sink] — one sink row shared by all N packed blocks.

## Panel B — serving physical layout (per sequence per layer)

    draft KV cache (DominoDraftAttention: full, never evicted; window at
    compute time):
    +----------------------------+---------------------------+--------------+
    | ctx rows pos 0 .. a-1      | block slots pos a .. a+15 | SINK slot    |
    | k_proj_target(fused tgt h) | k_proj(draft h)           | sink_k const |
    | k_norm yes  RoPE@pos yes   | k_norm yes RoPE@pos yes   | raw direct   |
    | written for new tokens     | slot-overwrite per step   | bytes fixed, |
    | each round                 | (transient)               | pos a+16     |
    +----------------------------+---------------------------+--------------+

    Serving positions are CONTIGUOUS 0..seq_len (slots come from the draft
    block table by position), so there is no free address below the block.
    The sink therefore occupies the position exactly ONE PAST the draft
    block (seq_len before extension): reachable only because domino layers
    attend block-bidirectionally, and within every band since
    |q - (a+16)| <= 16 << W. This is why dflash-causal variants reject
    attn_sink in serving v1.

## Panel C — round-to-round

    round r (anchor a=20):            verify: accept 20,21,22 + target 23
      [ctx …… 13|14 15 16 17 18 19] [blk: 20 21 22 23] [SINK X, pos 36=a+16]
       intra-block autoreg: slot 20 written, query 20; slot 21 overwritten,
       query 21 … sink untouched, visible at every micro-step
                                          |
                                          v
    round r+1 (anchor a'=24):
      [ctx …… 16|17 …… 22 23 (new, from target)] [blk reset: 24 .. 39]
      [SINK X, pos 40=a'+16]
                                          ^
      Content X is byte-identical across rounds AND sequences; only the
      hosted position advances with the tail. Each round the speculator
      extends draft seq_lens by num_sinks and rewrites the sink row at the
      tail slot (same eager pass as the context K/V precompute). The tail
      slot is derived with the same block-table clamp the prep kernel uses
      for query slots, with a loud collision check against this round's
      query slots.

## Panel D — row provenance (train vs serve)

    row   K/V content                    k_norm  RoPE      across rounds  across seqs
    ctx   k_proj_target(fused target h)  yes     yes@pos   rewritten      per-seq
    block k_proj(draft h)                yes     yes@pos   per-step ow    per-seq
    sink  learned parameter              no      no        bytes fixed    IDENTICAL

The sink is the only KV entry bit-identical between training and serving
and immune to prefix caching / mooncake transfer (a stale sink row is
exactly the correct row).

## Visibility invariants (acceptance checks)

For any query q in [a, a+15], any window W >= 32 (real min 512):

    context causal cut   kv < a            a-1 < a            ok
    band lower bound     q-(W-1) <= kv     q-(a-1) <= 16 <= W-1  ok
    domino forward reach kv <= q+W         a-1 <= q           ok
    dflash causal block  kv_off <= o       sink outside block, visible indep  ok
