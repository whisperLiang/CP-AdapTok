#!/usr/bin/env bash
# Build pinned upstream extensions in a separate environment; no system CUDA install.
set -euo pipefail
CP_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$CP_ROOT"
CP_UV="${CP_UV:-$(command -v uv || true)}"
if [[ -z "$CP_UV" ]]; then CP_UV="$HOME/.local/bin/uv"; fi
test -x "$CP_UV"
mkdir -p results .build
if [[ ! -x .venv-native/bin/python ]]; then
  "$CP_UV" venv --python 3.12 .venv-native
fi
"$CP_UV" pip install --python .venv-native/bin/python torch==2.9.1 torchvision==0.24.1 \
  --index-url https://download.pytorch.org/whl/cu130 > results/native_environment_install.log 2>&1
"$CP_UV" pip install --python .venv-native/bin/python \
  wheel pybind11 ninja psutil pyyaml scipy opencv-python-headless pytest pytest-cov \
  radon ruff pytorch-msssim lpips DISTS-pytorch matplotlib bd-metric tqdm \
  > results/native_dependencies_install.log 2>&1
"$CP_UV" pip install --python .venv-native/bin/python --target .build/cuda-packages --no-deps \
  nvidia-cuda-nvcc==13.0.88 nvidia-nvvm==13.0.88 nvidia-cuda-crt==13.0.88 \
  nvidia-cuda-runtime==13.0.96 nvidia-cuda-cccl==13.0.85 \
  > results/native_toolkit_install.log 2>&1
if [[ ! -d .build/cutlass/.git ]]; then
  git clone --depth 1 --branch v4.4.1 https://github.com/NVIDIA/cutlass.git .build/cutlass \
    > results/native_cutlass_clone.log 2>&1
fi
test "$(git -C .build/cutlass rev-parse HEAD)" = "$(git -C .build/cutlass rev-parse v4.4.1)"
.venv-native/bin/python - <<'PY'
from pathlib import Path
import shutil
root = Path.cwd()
for relative in ['src/cpp', 'src/layers/extensions/inference']:
    shutil.copytree(root/'third_party/DCVC'/relative, root/'.build/native-dcvc'/relative,
                    dirs_exist_ok=True, ignore=shutil.ignore_patterns('build', '*.egg-info', '__pycache__'))
links = {
    root/'.build/native-dcvc/third_party/cutlass': root/'.build/cutlass',
    root/'.build/cuda-packages/nvidia/cu13/lib64': root/'.build/cuda-packages/nvidia/cu13/lib',
    root/'.build/cuda-packages/nvidia/cu13/lib/libcudart.so': root/'.build/cuda-packages/nvidia/cu13/lib/libcudart.so.13',
}
for link, target in links.items():
    link.parent.mkdir(parents=True, exist_ok=True)
    if not link.exists():
        link.symlink_to(target)
PY
patch --directory=.build/native-dcvc/src/layers/extensions/inference -p1 \
  < native_patches/decoded_latent.patch > results/native_decoded_latent_patch.log
export CUDA_HOME="$CP_ROOT/.build/cuda-packages/nvidia/cu13"
export CPLUS_INCLUDE_PATH="$CP_ROOT/.venv-native/lib/python3.12/site-packages/nvidia/cu13/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
export MAX_JOBS="${MAX_JOBS:-4}"
"$CP_UV" pip install --python .venv-native/bin/python --no-build-isolation .build/native-dcvc/src/cpp \
  > results/native_cpp_build.log 2>&1
"$CP_UV" pip install --python .venv-native/bin/python --no-build-isolation \
  .build/native-dcvc/src/layers/extensions/inference > results/native_cuda_build.log 2>&1
"$CP_UV" pip freeze --python .venv-native/bin/python > results/native_environment_freeze.txt
.venv-native/bin/python - <<'PY'
import json
import hashlib
import sys
import subprocess
import torch
import MLCodec_extensions_cpp
import inference_extensions_cuda
from pathlib import Path
info = dict(torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
            python=sys.version, cpp=MLCodec_extensions_cpp.__file__, inference=inference_extensions_cuda.__file__,
            decoded_latent_api=hasattr(inference_extensions_cuda.DMCIProxy, 'get_decoded_latent') and
                               hasattr(inference_extensions_cuda.DMCLDProxy, 'get_decoded_latent'),
            patch_sha256=hashlib.sha256(Path('native_patches/decoded_latent.patch').read_bytes()).hexdigest(),
            upstream=subprocess.check_output(['git','-C','third_party/DCVC','rev-parse','HEAD'],text=True).strip())
info['patched_source_sha256'] = {
    name: hashlib.sha256((Path('.build/native-dcvc/src/layers/extensions/inference')/name).read_bytes()).hexdigest()
    for name in ['dmci_proxy.h', 'dmc_ld_proxy.h', 'bind.cpp']}
Path('results/native_environment.json').write_text(json.dumps(info, indent=2)+'\n')
print(json.dumps(info, indent=2))
PY
