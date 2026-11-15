#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$PWD/.venv/bin:$PATH"
python -m cp_adaptok.cli.eval --config configs/s0_baseline.yaml --baseline native
python -m cp_adaptok.cli.eval --config configs/s1_tokenizer.yaml --budgets 128 256 512 1024 2048
python -m cp_adaptok.cli.eval --config configs/ablation_spatial.yaml --budgets 128 256 512 1024 2048
python -m cp_adaptok.cli.eval --config configs/ablation_8192.yaml --budgets 128 256 512 1024 2048 4096 8192
