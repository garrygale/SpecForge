#!/usr/bin/env python3
"""Probe the GDN accepted-token / varlen boundary reported as issue #9956.

The custom Ascend GDN recurrent/conv ops validate every row with
``acceptedTokenNum > seqLen``.  This probe constructs the same
``spec_query_start_loc`` / ``num_accepted_tokens`` metadata contract used by
``vllm_ascend/ops/gdn_attn_builder.py`` and reports rows that would be
rejected by the C++ op, including partial final rounds.

Run on CPU or NPU; this probe is numerical, not operator-execution based.
"""

from __future__ import annotations

import torch


def _rows(
    spec_query_start_loc: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
) -> list[tuple[int, int, bool]]:
    rows: list[tuple[int, int, bool]] = []
    lengths = torch.diff(spec_query_start_loc)
    for i, (accepted, length) in enumerate(
        zip(num_accepted_tokens.tolist(), lengths.tolist(), strict=True)
    ):
        valid = accepted > 0 and accepted <= length
        rows.append((i, accepted, valid))
    return rows


def main() -> None:
    # Full seven-token draft round: each row is exactly 7 query tokens.
    qsl_full = torch.tensor([0, 7, 14, 21], dtype=torch.long)
    accepted_full = torch.tensor([1, 7, 2], dtype=torch.long)
    full_rows = _rows(qsl_full, accepted_full)

    # End-of-sequence round: the final row is truncated to 3 tokens while the
    # accepted count is still 4 (bonus + three accepted drafts).  This is the
    # shape that the upstream issue reports as 4 exceeds segment length 3.
    qsl_eos = torch.tensor([0, 7, 14, 17], dtype=torch.long)
    accepted_eos = torch.tensor([7, 4, 4], dtype=torch.long)
    eos_rows = _rows(qsl_eos, accepted_eos)
    bad = [row for row in eos_rows if not row[2]]
    print("full round rows:", full_rows)
    print("EOS/partial round rows:", eos_rows)
    if bad:
        print(
            "diagnosis: accepted > spec_query_start_loc segment length only "
            "in the partial final round; a full all-accepted round stays "
            "within the row length."
        )
    else:
        print("diagnosis: no accepted > segment length row found in this case.")

    # Clamp now applied by the service before the custom conv/recurrent ops.
    lengths = torch.diff(qsl_eos)
    clamped = accepted_eos.clamp(max=lengths)
    print(
        "service clamp output:",
        clamped.tolist(),
        " (active rows are still at least 1; padded rows are filled later)",
    )


if __name__ == "__main__":
    main()
