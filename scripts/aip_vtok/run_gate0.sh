#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=4
PYTHON=.venv-native/bin/python
mkdir -p results/aip_vtok/tests

# Do not disrupt another workload. Every GPU phase below runs sequentially.
if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
    echo 'GPU is occupied; leaving all existing processes untouched.' >&2
    exit 3
fi
"$PYTHON" -m aip_vtok.evaluation.audit > results/aip_vtok/audit.log
"$PYTHON" -m pytest -q tests/aip_vtok --junitxml=results/aip_vtok/tests/pytest.xml \
    | tee results/aip_vtok/tests/gate0_green.txt
if [ ! -f results/aip_vtok/baseline.json ]; then
    "$PYTHON" -m aip_vtok.evaluation.gate0 baseline | tee results/aip_vtok/gate0_baseline.log
fi
for width in 128 256; do
    if [ ! -f "results/aip_vtok/projector_w${width}/evaluation.json" ]; then
        resume=()
        if [ -f "checkpoints/aip_vtok/projector_w${width}/latest.pt" ]; then
            resume=(--resume)
        fi
        "$PYTHON" -m aip_vtok.evaluation.gate0 train --width "$width" "${resume[@]}" \
            | tee -a "results/aip_vtok/gate0_projector_w${width}.log"
    fi
    if "$PYTHON" -m aip_vtok.evaluation.report --require-pass; then
        exit 0
    fi
done
"$PYTHON" -m aip_vtok.evaluation.report --require-pass
