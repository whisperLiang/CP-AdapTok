#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=4
exec .venv-adaptok-official/bin/python -m experiments.original_adaptok_1080p.cli "$@"
