#!/usr/bin/env bash
set -euo pipefail

root=${1:-/data2/auto-infer-multistep-mtp-20260720}
device=${2:-1}
mkdir -p "$root/results"
export PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"

for depth in 1 2; do
    python "$root/scripts/mimo_mtp_graph_batched.py" "$device" "$depth" \
        2>&1 | tee "$root/results/mimo-graph-mtp-k${depth}.log"
done
