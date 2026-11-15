#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$PWD/.venv/bin:$PATH"
python -m cp_adaptok.cli.benchmark --resolution 1920 1080 --measure actual-bitstream --budgets 128 2048
python -m cp_adaptok.cli.benchmark --resolution 1920 1080 --measure actual-bitstream --budgets 128 256 512 1024 2048 --output results/benchmark_all_prefixes
