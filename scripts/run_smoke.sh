#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$PWD/.venv/bin:$PATH"
export OMP_NUM_THREADS=2
python -m pytest cp_adaptok/tests -q
python -m cp_adaptok.cli.train --config configs/s1_tokenizer.yaml --max-steps 100 --smoke --allow-untrained
python -m cp_adaptok.cli.eval --config configs/s1_tokenizer.yaml --budgets 128 256 512 1024 2048 --checkpoint checkpoints/s1_smoke/latest.pt --allow-untrained --output results/smoke_eval
python -m cp_adaptok.cli.stream_demo --mode localhost --progressive --checkpoint checkpoints/s1_smoke/latest.pt --allow-untrained
