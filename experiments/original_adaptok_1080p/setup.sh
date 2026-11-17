#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
uv venv --python 3.10 .venv-adaptok-official --allow-existing
uv pip install --python .venv-adaptok-official/bin/python \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121
uv pip install --python .venv-adaptok-official/bin/python \
  -r experiments/original_adaptok_1080p/requirements.txt
