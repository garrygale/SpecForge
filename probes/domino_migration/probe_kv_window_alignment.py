#!/usr/bin/env python3
"""Train-vs-serve draft-attention (KV visibility) alignment probe.

Compares, for one draft block, the KV positions each query may read:

* training mask built by ``create_dflash_sdpa_mask`` (SpecForge), and
* the mask the vLLM / vLLM-Ascend draft attention actually applies at serving
  time, namely an Ascend FIA / CUDA window ``(pre_tokens=W, next_tokens=W)``
  for non-causal sliding layers (``sparse_mode=4`` band) and
  ``(pre_tokens=W, next_tokens=0)`` for causal sliding layers.

Both live on the same absolute-position axis: the block's element ``j`` sits at
absolute position ``anchor + j`` (``anchor`` == the position of the block's
first/bonus token) and the context occupies positions ``[0, anchor - 1]``.

Run from the SpecForge root:

    set PYTHONPATH=C:/Users/g/Desktop/codeAgents/SpecForge
    C:/Users/g/Desktop/codeAgents/phi-GNNv2/.venv/Scripts/python.exe \
        probes/domino_migration/probe_kv_window_alignment.py
"""

from __future__ import annotations

import torch

from specforge.algorithms.common.dflash_family_model import create_dflash_sdpa_mask


def training_visible_sets(
    *,
    anchor: int,
    block_size: int,
    sliding_window: int | None,
    sliding_draft_causal: bool = False,
) -> list[set[int]]:
    """Absolute KV positions visible per query, per SpecForge's training mask.

    The training sequence is ``context = [0, anchor)`` followed by one draft
    block whose element ``k`` sits at absolute position ``anchor + k`` --
    the same absolute axis the serving KV cache uses.

    ``sliding_draft_causal=True`` selects the legacy causal draft block; the
    default (``False``) is the served band that Domino runs.
    """
    seq_len = anchor
    anchors = torch.tensor([[anchor]], dtype=torch.long)
    keep = torch.ones((1, 1), dtype=torch.bool)
    mask = create_dflash_sdpa_mask(
        anchor_positions=anchors,
        block_keep_mask=keep,
        S=seq_len,
        block_size=block_size,
        device=torch.device("cpu"),
        sliding_window=sliding_window,
        sliding_draft_causal=sliding_draft_causal,
    )[0, 0]  # (block_size, seq_len + block_size)

    # Column -> absolute position: [0, seq_len) are context, [seq_len, ..) the
    # block's own elements anchored at `anchor`.
    absolute = list(range(seq_len)) + [anchor + k for k in range(block_size)]
    return [
        {absolute[col] for col in range(len(absolute)) if bool(mask[q, col])}
        for q in range(block_size)
    ]


def served_band_visible_sets(
    *,
    anchor: int,
    block_size: int,
    sliding_window: int | None,
    causal: bool,
) -> list[set[int]]:
    """Absolute KV positions visible per query with the served FIA band mask.

    The draft's KV cache holds ``[0, anchor - 1]`` (target-derived context) plus
    the current block ``[anchor, anchor + block_size - 1]`` (draft-derived) and
    ``seq_lens`` ends at the block's last position, so the band is clipped to
    the union of both.
    """
    cache = set(range(anchor)) | {anchor + k for k in range(block_size)}
    out: list[set[int]] = []
    for j in range(block_size):
        q = anchor + j
        if sliding_window is None:
            lo, hi = 0, max(cache)
        elif causal:
            # pre_tokens=W, next_tokens=0 -> [q - W + 1, q]
            lo, hi = q - sliding_window + 1, q
        else:
            # pre_tokens=W, next_tokens=W -> [q - W + 1, q + W]
            lo, hi = q - sliding_window + 1, q + sliding_window
        out.append({p for p in cache if lo <= p <= hi})
    return out


def served_unmasked_visible_sets(
    *,
    anchor: int,
    block_size: int,
) -> list[set[int]]:
    """SpecForge's in-process ``spec_generate`` path: no mask is passed at all."""
    cache = set(range(anchor)) | {anchor + k for k in range(block_size)}
    return [set(cache) for _ in range(block_size)]


def proposed_training_visible_sets(
    *,
    anchor: int,
    block_size: int,
    sliding_window: int,
    forward_cap: bool = True,
) -> list[set[int]]:
    """Candidate training mask: reproduce the served band on the draft block.

    Context half (unchanged from today's builder):
        ``kv < anchor`` and ``kv >= q - (W - 1)``
    Draft half (the fix):
        same block and ``kv >= q - (W - 1)`` and (optionally) ``kv <= q + W``

    ``forward_cap=False`` mirrors the shape of the Draft-OPD replay fix on
    ``upstream/codex/draft-opd-replay`` (band on the left, whole block on the
    right), which is a superset of the served band when ``W < block_size``.
    """
    out: list[set[int]] = []
    for j in range(block_size):
        q = anchor + j
        lower = q - (sliding_window - 1)
        visible = {p for p in range(anchor) if lower <= p < anchor}
        for k in range(block_size):
            p = anchor + k
            if p < lower:
                continue
            if forward_cap and p > q + sliding_window:
                continue
            visible.add(p)
        out.append(visible)
    return out


def check_shipped_mask_matches_served(
    *,
    anchor: int,
    block_size: int,
    windows: tuple[int, ...],
) -> None:
    """Verify the shipped default mask reproduces the served band exactly."""
    print("\n--- shipped training mask (default) vs served band (non-causal) ---")
    for window in windows:
        served = served_band_visible_sets(
            anchor=anchor,
            block_size=block_size,
            sliding_window=window,
            causal=False,
        )
        shipped = training_visible_sets(
            anchor=anchor, block_size=block_size, sliding_window=window
        )
        left_only = proposed_training_visible_sets(
            anchor=anchor,
            block_size=block_size,
            sliding_window=window,
            forward_cap=False,
        )
        exact_ok = all(a == b for a, b in zip(shipped, served))
        left_ok = all(b <= a for a, b in zip(left_only, served))
        extra = sum(len(a - b) for a, b in zip(left_only, served))
        print(
            f"W={window:>4}: create_dflash_sdpa_mask default == served: {exact_ok}; "
            f"OPD-style (no forward cap) is a superset: {left_ok} "
            f"(+{extra} block KVs over all queries)"
        )
        assert exact_ok, f"training mask does not match serving for W={window}"


def compare(
    *,
    anchor: int,
    block_size: int,
    sliding_window: int | None,
    causal: bool,
    sliding_draft_causal: bool = True,
) -> dict[str, object]:
    train = training_visible_sets(
        anchor=anchor,
        block_size=block_size,
        sliding_window=sliding_window,
        sliding_draft_causal=sliding_draft_causal,
    )
    serve = served_band_visible_sets(
        anchor=anchor,
        block_size=block_size,
        sliding_window=sliding_window,
        causal=causal,
    )
    unmasked = served_unmasked_visible_sets(anchor=anchor, block_size=block_size)

    rows = []
    extra_total = missing_total = train_total = 0
    for j, (t, s, u) in enumerate(zip(train, serve, unmasked)):
        extra = s - t
        missing = t - s
        extra_total += len(extra)
        missing_total += len(missing)
        train_total += len(t)
        rows.append(
            {
                "j": j,
                "train": len(t),
                "serve": len(s),
                "extra": len(extra),
                "extra_pct": 100.0 * len(extra) / max(1, len(t)),
                "missing": len(missing),
                "nomask_extra": len(u - t),
                "extra_rel": sorted(p - (anchor + j) for p in extra),
                "missing_rel": sorted(p - (anchor + j) for p in missing),
            }
        )
    return {
        "rows": rows,
        "extra_total": extra_total,
        "missing_total": missing_total,
        "train_total": train_total,
    }


def short_report(
    *,
    anchor: int,
    block_size: int,
    sliding_window: int | None,
    causal: bool,
    sliding_draft_causal: bool = True,
) -> None:
    res = compare(
        anchor=anchor,
        block_size=block_size,
        sliding_window=sliding_window,
        causal=causal,
        sliding_draft_causal=sliding_draft_causal,
    )
    rows = res["rows"]
    probe_offsets = sorted({0, block_size // 4, block_size // 2, block_size - 1})
    pct = "  ".join(
        f"j={rows[j]['j']}: {rows[j]['train']}->{rows[j]['serve']} "
        f"(+{rows[j]['extra']} / -{rows[j]['missing']})"
        for j in probe_offsets
    )
    print(
        f"W={str(sliding_window):>4} served_causal={str(causal):>5} "
        f"legacy_draft_causal={str(sliding_draft_causal):>5}: {pct}"
    )
    print(
        f"        |  aggregate over all {block_size} queries: "
        f"sum(serve\\train)={res['extra_total']} "
        f"({100.0 * res['extra_total'] / res['train_total']:.1f}% of trained KVs), "
        f"sum(train\\serve)={res['missing_total']} "
        f"({100.0 * res['missing_total'] / res['train_total']:.1f}%)"
    )


def verbose_report(
    *,
    anchor: int,
    block_size: int,
    sliding_window: int | None,
    causal: bool,
    label: str,
    sliding_draft_causal: bool = True,
) -> None:
    res = compare(
        anchor=anchor,
        block_size=block_size,
        sliding_window=sliding_window,
        causal=causal,
        sliding_draft_causal=sliding_draft_causal,
    )
    print(
        f"\n=== {label} (W={sliding_window}, causal={causal}) ===\n"
        f"context = [0, {anchor}), block = [{anchor}, {anchor + block_size})"
    )
    print(
        f"{'q_off':>5} {'|train|':>8} {'|serve|':>8} {'extra':>6} "
        f"{'extra%':>7} {'missing':>8} {'nomask_extra':>13}"
    )
    for row in res["rows"]:
        print(
            f"{row['j']:>5} {row['train']:>8} {row['serve']:>8} "
            f"{row['extra']:>6} {row['extra_pct']:>6.1f}% "
            f"{row['missing']:>8} {row['nomask_extra']:>13}"
        )
        if row["j"] in (0, block_size // 2, block_size - 1):
            if row["extra_rel"]:
                print(f"      extra served KVs (rel. to q): {row['extra_rel']}")
            if row["missing_rel"]:
                print(f"      training-only KVs (rel. to q): {row['missing_rel']}")


def main() -> None:
    import sys

    verbose = "--verbose" in sys.argv
    # Long enough context that a 2k window is fully populated, as in serving.
    anchor = 4096
    block_size = 16
    print(
        "Block element j is at absolute position anchor+j, matching vLLM's "
        "'query_pos = last_valid_pos + 1 + j' (anchor == last_valid_pos + 1)."
    )
    print(
        "'train' = SpecForge training mask (create_dflash_sdpa_mask); "
        "'serve' = FIA/CUDA draft mask "
        "(pre_tokens=W, next_tokens=W for non-causal, =0 for causal).\n"
        "'nomask_extra' = extra KVs the in-process spec_generate evaluator "
        "sees (it passes no attention mask at all).\n"
        "The tables below use the LEGACY causal draft block "
        "(sliding_draft_causal=True), which is what training did before this "
        "probe's fix; the shipped default (False = served band) is verified at "
        "the end of the run."
    )

    print(
        "\n--- legacy training mask vs Domino serving "
        "(sliding layers non-causal, band) ---"
    )
    for window in (2, 4, 8, 16, 24, 32, 64, 128, 512, 1024, 2048):
        short_report(
            anchor=anchor,
            block_size=block_size,
            sliding_window=window,
            causal=False,
        )

    print(
        "\n--- legacy training mask vs DFlash-style serving "
        "(sliding layers causal) ---"
    )
    for window in (16, 512, 2048):
        short_report(
            anchor=anchor,
            block_size=block_size,
            sliding_window=window,
            causal=True,
        )

    print("\n--- Full-attention layer (no window) ---")
    short_report(
        anchor=anchor,
        block_size=block_size,
        sliding_window=None,
        causal=False,
    )

    check_shipped_mask_matches_served(
        anchor=anchor,
        block_size=block_size,
        windows=(1, 2, 4, 8, 16, 24, 32, 64, 512, 2048),
    )

    if verbose:
        for window in (4, 512):
            verbose_report(
                anchor=4096,
                block_size=block_size,
                sliding_window=window,
                causal=False,
                label="sliding layer, Domino serving (band)",
            )
        verbose_report(
            anchor=4096,
            block_size=block_size,
            sliding_window=None,
            causal=False,
            label="full-attention layer",
        )


if __name__ == "__main__":
    main()
