"""Set CPU BLAS threading env vars BEFORE torch / OpenBLAS initialize.

Some OpenBLAS builds (common in torch Linux/conda CPU wheels; notably on
big many-core servers) emit a stream of "BLAS: bad memory unallocation"
errors when their threaded memory pool races under high thread counts.
The reliable workaround is pinning OPENBLAS_NUM_THREADS to a small value
before the BLAS library initializes (i.e. before the first matmul).

This module must be imported BEFORE torch (and numpy) in every probe script.
Defaults:
    OPENBLAS_NUM_THREADS = 1, OMP_NUM_THREADS = 1
Overrides (in priority order):
    SPECFORGE_PROBE_BLAS_THREADS=<n>   (force a specific thread count)
    OPENBLAS_NUM_THREADS=<n>           (already-set env is respected)
    OMP_NUM_THREADS=<n>                (already-set env is respected)

If you set OPENBLAS_NUM_THREADS=4..8 and the errors come back, go back to 1.
"""

import os

_threads = os.environ.get("SPECFORGE_PROBE_BLAS_THREADS", "1")

forced = []
if os.environ.get("OPENBLAS_NUM_THREADS") is None:
    os.environ["OPENBLAS_NUM_THREADS"] = _threads
    forced.append("OPENBLAS_NUM_THREADS=" + _threads)
if os.environ.get("OMP_NUM_THREADS") is None:
    os.environ["OMP_NUM_THREADS"] = _threads
    forced.append("OMP_NUM_THREADS=" + _threads)

if forced:
    print("[probes] " + ", ".join(forced)
          + " (OpenBLAS memory-pool race workaround; override with "
          + "SPECFORGE_PROBE_BLAS_THREADS or the env vars)")
