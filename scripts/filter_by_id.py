#!/usr/bin/env python3
# coding=utf-8
"""Keep only the JSONL samples whose ``id`` is present in another JSONL file.

This is the post-processing helper for data produced by
``scripts/regen_responses.py``.  Two regeneration runs over the same input
subset (different model / settings / retry) only contain the samples that
succeeded, so the two output files share most ids but not all of them.  This
script filters one file against the other: each line of ``--input`` is kept iff
its ``id`` value also occurs in ``--other``; everything else is dropped.  The
filtered file is written to a new path, never in place by default.

Kept lines are copied verbatim from the input (only the line ending is
normalized to ``\\n``), so they stay byte-identical to the regenerated rows.
The reference file is streamed once; only its ids are kept in memory.

Usage::

    # after setting INPUT_PATH / OTHER_PATH in the Configuration block below
    python scripts/filter_by_id.py

    # keep the samples of run_a_regen.jsonl that also exist in run_b_regen.jsonl
    python scripts/filter_by_id.py \\
        --input ./regenerated/run_a_regen.jsonl \\
        --other ./regenerated/run_b_regen.jsonl

    # explicit output path and a different id field
    python scripts/filter_by_id.py --input a.jsonl --other b.jsonl \\
        --output ./regenerated/a_common.jsonl --id-key source_id

    # preview the counts without writing anything
    python scripts/filter_by_id.py --input a.jsonl --other b.jsonl --dry-run

Edit the defaults in the ``Configuration`` block at the top of this file, or
override them via CLI arguments (see ``--help``).

To filter the other direction, swap ``--input`` and ``--other``.  The default
output is ``<input without .jsonl>_filtered.jsonl`` next to the input file;
an existing output is only replaced with ``--overwrite``.

Both files are expected to be JSONL with one JSON object per line.  A line
without the id field (or with a null id) can never match and is dropped; the
summary counts those separately.  Malformed JSON aborts the run unless
``--skip-invalid`` is given, in which case the bad lines are counted and
skipped.

Exit codes: 0 on success (even if nothing is kept), 1 on any error.
Requires Python 3.10+ and only the standard library.
"""

# ---- Configuration ---------------------------------------------------------
# Edit these defaults to run the script without any CLI arguments.  Every value
# can still be overridden on the command line (see --help).  Relative paths are
# resolved from the directory the script is run in.

INPUT_PATH = "./regenerated/run_a_regen.jsonl"  # JSONL file to filter (trimmed)
OTHER_PATH = "./regenerated/run_b_regen.jsonl"  # JSONL file with the keep-list ids
OUTPUT_PATH = None  # None = <INPUT_PATH without .jsonl>_filtered.jsonl
ID_KEY = "id"  # JSON field holding the sample id in both files
DRY_RUN = False  # True = only report the counts, write no output file
OVERWRITE = False  # True = allow replacing an existing output file
SKIP_INVALID = False  # True = skip malformed JSON lines instead of aborting
PROGRESS_EVERY = 100_000  # print progress every N rows; 0 disables
QUIET = False  # True = no progress lines (the final summary is still printed)
# ----------------------------------------------------------------------------

import argparse
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple


@dataclass
class Counts:
    """Row counters shared by the reference scan and the filter pass."""

    lines: int = 0  # non-blank lines read
    blank: int = 0  # blank / whitespace-only lines
    invalid: int = 0  # lines that are not a JSON object
    no_id: int = 0  # rows without a usable id value
    kept: int = 0  # filter pass: rows written to the output
    dropped: int = 0  # filter pass: rows whose id is not in the reference
    duplicate_ids: int = 0  # reference: repeated ids


# ---------------------------------------------------------------------------
#  JSONL helpers
# ---------------------------------------------------------------------------


def sample_id_key(value: Any) -> Optional[str]:
    """Return the comparison key for an id value, or None when there is no id.

    IDs are strings in the datasets this script was written for, but the key
    is canonicalized through JSON so that, e.g., an integer id ``42`` in one
    file still matches the string id ``"42"`` in the other.  ``None`` (a
    missing field and a JSON ``null`` are treated the same) never matches.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def iter_jsonl(
    path: str,
    *,
    skip_invalid: bool,
    counts: Counts,
    progress_every: int = 0,
    progress_label: str = "rows",
) -> Iterator[Tuple[int, Dict[str, Any], str]]:
    """Yield ``(line_number, parsed_object, raw_line)`` for every data line.

    Blank lines are counted and skipped.  Lines that are not a JSON object are
    counted and either skipped (``skip_invalid=True``) or raised as a
    ``SystemExit`` error naming the file and line.
    """
    with open(path, "r", encoding="utf-8-sig") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                counts.blank += 1
                continue
            counts.lines += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                counts.invalid += 1
                if skip_invalid:
                    continue
                raise SystemExit(
                    f"error: {path}:{lineno}: invalid JSON "
                    f"({exc.msg} at column {exc.colno})"
                ) from None
            if not isinstance(row, dict):
                counts.invalid += 1
                if skip_invalid:
                    continue
                raise SystemExit(
                    f"error: {path}:{lineno}: expected a JSON object, "
                    f"got {type(row).__name__}"
                ) from None
            if progress_every > 0 and counts.lines % progress_every == 0:
                print(
                    f"  ... {counts.lines:,} {progress_label} read",
                    file=sys.stderr,
                )
            yield lineno, row, line.rstrip("\r\n")


def load_reference_ids(
    path: str,
    id_key: str,
    *,
    skip_invalid: bool,
    progress_every: int = 0,
) -> Tuple[set, Counts]:
    """Read *path* once and return the set of its ids plus scan counters."""
    counts = Counts()
    ids: set = set()
    for _, row, _ in iter_jsonl(
        path,
        skip_invalid=skip_invalid,
        counts=counts,
        progress_every=progress_every,
        progress_label="reference rows",
    ):
        key = sample_id_key(row.get(id_key))
        if key is None:
            counts.no_id += 1
        elif key in ids:
            counts.duplicate_ids += 1
        else:
            ids.add(key)
    return ids, counts


# ---------------------------------------------------------------------------
#  Filtering
# ---------------------------------------------------------------------------


def _default_file_mode() -> int:
    """File mode a plain ``open(..., "w")`` would use, i.e. 0666 minus umask."""
    current_umask = os.umask(0)
    os.umask(current_umask)
    return 0o666 & ~current_umask


def filter_file(
    input_path: str,
    output_path: str,
    reference_ids: set,
    id_key: str,
    *,
    skip_invalid: bool,
    dry_run: bool,
    progress_every: int = 0,
) -> Counts:
    """Write the kept rows of *input_path* to *output_path*; return counters.

    The output is written to a temporary file in the destination directory and
    atomically renamed into place, so an interrupted run never leaves a
    truncated file that looks like a finished one.
    """
    counts = Counts()
    out_fh = None
    tmp_path: Optional[str] = None
    try:
        if not dry_run:
            out_dir = os.path.dirname(output_path) or os.curdir
            os.makedirs(out_dir, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(
                dir=out_dir,
                prefix=os.path.basename(output_path) + ".",
                suffix=".tmp",
            )
            # mkstemp creates the file as 0600; give the output the normal
            # permissions a plain open() would have used instead.
            try:
                os.chmod(tmp_path, _default_file_mode())
            except OSError:
                pass
            out_fh = os.fdopen(fd, "w", encoding="utf-8", newline="\n")

        for _, row, raw_line in iter_jsonl(
            input_path,
            skip_invalid=skip_invalid,
            counts=counts,
            progress_every=progress_every,
            progress_label="input rows",
        ):
            key = sample_id_key(row.get(id_key))
            if key is None:
                counts.no_id += 1
            elif key in reference_ids:
                counts.kept += 1
                if out_fh is not None:
                    out_fh.write(raw_line + "\n")
            else:
                counts.dropped += 1

        if out_fh is not None:
            out_fh.close()
            out_fh = None
            os.replace(tmp_path, output_path)
            tmp_path = None
    finally:
        if out_fh is not None:
            out_fh.close()
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    return counts


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter a JSONL file, keeping only the samples whose id appears "
            "in another JSONL file."
        )
    )
    parser.add_argument(
        "--input",
        default=INPUT_PATH,
        help=(
            "JSONL file to filter (the file that gets trimmed) "
            f"(default: {INPUT_PATH})"
        ),
    )
    parser.add_argument(
        "--other",
        default=OTHER_PATH,
        help=(
            "JSONL file whose ids form the keep-list (the reference file) "
            f"(default: {OTHER_PATH})"
        ),
    )
    output_default = (
        OUTPUT_PATH
        if OUTPUT_PATH is not None
        else "<input without .jsonl>_filtered.jsonl"
    )
    parser.add_argument(
        "--output",
        default=OUTPUT_PATH,
        help=f"Output path (default: {output_default})",
    )
    parser.add_argument(
        "--id-key",
        default=ID_KEY,
        help=(
            "JSON field holding the sample id in both files "
            f"(default: {ID_KEY})"
        ),
    )
    parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action=argparse.BooleanOptionalAction,
        default=DRY_RUN,
        help=(
            "Only report the counts, write no output file "
            f"(default: {DRY_RUN})"
        ),
    )
    parser.add_argument(
        "--overwrite",
        dest="overwrite",
        action=argparse.BooleanOptionalAction,
        default=OVERWRITE,
        help=f"Allow replacing an existing output file (default: {OVERWRITE})",
    )
    parser.add_argument(
        "--skip-invalid",
        dest="skip_invalid",
        action=argparse.BooleanOptionalAction,
        default=SKIP_INVALID,
        help=(
            "Skip (and count) malformed JSON lines instead of aborting "
            f"(default: {SKIP_INVALID})"
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=PROGRESS_EVERY,
        help=(
            "Print a progress line to stderr every N rows; 0 disables "
            f"(default: {PROGRESS_EVERY})"
        ),
    )
    parser.add_argument(
        "-q",
        "--quiet",
        dest="quiet",
        action=argparse.BooleanOptionalAction,
        default=QUIET,
        help=(
            "Do not print progress lines; the final summary is still printed "
            f"(default: {QUIET})"
        ),
    )
    return parser.parse_args(argv)


def _print_scan(
    title: str,
    path: str,
    counts: Counts,
    unique_ids: Optional[int] = None,
) -> None:
    print(f"{title}: {path}")
    print(f"  rows: {counts.lines:,}   blank lines: {counts.blank:,}")
    if unique_ids is not None:
        print(
            f"  unique ids: {unique_ids:,}   "
            f"duplicate-id rows: {counts.duplicate_ids:,}   "
            f"rows without id: {counts.no_id:,}"
        )
    else:
        print(
            f"  kept: {counts.kept:,}   "
            f"dropped (id not in reference): {counts.dropped:,}   "
            f"rows without id: {counts.no_id:,}"
        )
    if counts.invalid:
        print(f"  skipped invalid JSON lines: {counts.invalid:,}")


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    input_path = os.path.abspath(args.input)
    other_path = os.path.abspath(args.other)
    if not os.path.isfile(input_path):
        raise SystemExit(
            f"error: input file not found: {input_path}\n"
            "       set INPUT_PATH at the top of the script or pass --input"
        )
    if not os.path.isfile(other_path):
        raise SystemExit(
            f"error: reference file not found: {other_path}\n"
            "       set OTHER_PATH at the top of the script or pass --other"
        )
    if os.path.normcase(input_path) == os.path.normcase(other_path):
        raise SystemExit("error: the input and reference files must differ")

    if args.output is None:
        output_path = os.path.splitext(input_path)[0] + "_filtered.jsonl"
    else:
        output_path = os.path.abspath(args.output)
    if os.path.normcase(output_path) in (
        os.path.normcase(input_path),
        os.path.normcase(other_path),
    ):
        raise SystemExit("error: output must not overwrite the input or reference file")
    if not args.dry_run and os.path.exists(output_path) and not args.overwrite:
        raise SystemExit(
            f"error: output already exists: {output_path}\n"
            "       pass --overwrite to replace it"
        )

    progress_every = 0 if args.quiet else max(args.progress_every, 0)

    print(f"Loading ids from {other_path} ...")
    reference_ids, reference_counts = load_reference_ids(
        other_path,
        args.id_key,
        skip_invalid=args.skip_invalid,
        progress_every=progress_every,
    )
    if not reference_ids:
        print(
            "WARNING: the reference file contains no usable ids with "
            f"id field {args.id_key!r}; the filtered file will be empty.",
            file=sys.stderr,
        )

    print(f"Filtering {input_path} ...")
    counts = filter_file(
        input_path,
        output_path,
        reference_ids,
        args.id_key,
        skip_invalid=args.skip_invalid,
        dry_run=args.dry_run,
        progress_every=progress_every,
    )
    if counts.lines and counts.no_id == counts.lines:
        print(
            "WARNING: every input row is missing the id field "
            f"{args.id_key!r}; check ID_KEY / --id-key. Nothing was kept.",
            file=sys.stderr,
        )

    print("-" * 60)
    _print_scan("Reference", other_path, reference_counts, len(reference_ids))
    print("-" * 60)
    _print_scan("Filter", input_path, counts)
    print("-" * 60)
    if args.dry_run:
        print(f"Dry run: kept {counts.kept:,} rows; no file written.")
    else:
        print(f"Output: {output_path} ({counts.kept:,} rows)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, UnicodeError) as exc:
        sys.exit(f"error: {exc}")
