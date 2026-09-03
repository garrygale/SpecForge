#!/usr/bin/env bash
# Run all Domino service NPU probes from the SpecForge repository root.
#
# Usage:
#   bash probes/domino_service_npu/run_all_probes.sh
#   PYTHON=/path/to/target/python bash probes/domino_service_npu/run_all_probes.sh

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="${PYTHON:-python}"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "ERROR: Python executable not found: $PYTHON" >&2
    exit 2
fi

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"

status=0

run_probe() {
    local name="$1"
    shift
    echo
    echo "=================================================================="
    echo "RUN: $name"
    echo "=================================================================="
    if "$PYTHON" "$ROOT/probes/domino_service_npu/$name" "$@"; then
        echo "PASS: $name"
    else
        status=1
        echo "FAIL: $name"
    fi
}

run_probe probe_domino_config_and_7steps.py
run_probe probe_gdn_accepted_boundary.py
run_probe probe_quant_paths.py --real-dims
run_probe probe_grouped_fused_kv.py --real-dims
run_probe probe_non_causal_band.py --window 512 --tokens 2048

echo
if [ "$status" -eq 0 ]; then
    echo "All probes passed."
else
    echo "One or more probes failed."
fi
exit "$status"
