#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
project_root="$PWD"
python_bin="${PYTHON_BIN:-python}"
cutlass="$(realpath "${CUTLASS_HOME:?set CUTLASS_HOME to the pinned CUTLASS checkout}")"
toolkit="$(realpath "${CUDA_HOME:?set CUDA_HOME to the CUDA toolkit}")"
patch_file="$project_root/configs/hla_tok/hts/native_state_audit.patch"
native_root="$project_root/.build/hts/native"
upstream="$project_root/third_party/DCVC/src"
proxy="$native_root/src/layers/extensions/inference"
env_path="$("$python_bin" -c 'import sys; print(sys.prefix)')"
torch_path="$("$python_bin" -c 'from pathlib import Path; import torch; print(Path(torch.__file__).parent)')"
python_headers="$("$python_bin" -c 'from sysconfig import get_path; print(get_path("include"))')"
suffix="$("$python_bin" -c 'from sysconfig import get_config_var; print(get_config_var("EXT_SUFFIX"))')"
nvidia_include="$("$python_bin" -c 'from sysconfig import get_path; print(get_path("purelib") + "/nvidia/cu13/include")')"
export CPLUS_INCLUDE_PATH="$nvidia_include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
export CUDA_HOME="$toolkit"
export PATH="$env_path/bin:$toolkit/bin:$PATH"
export MAX_JOBS="${MAX_JOBS:-12}"
mkdir -p "$proxy" "$native_root"/{src/cpp,third_party,site}
ln -sfn "$cutlass" "$native_root/third_party/cutlass"
cp -a "$upstream/cpp/." "$native_root/src/cpp/"
cp -a "$upstream/layers/extensions/inference/." "$proxy/"
patch --batch --directory="$proxy" -p1 --fuzz=0 < "$patch_file"

if [[ -z "${HTS_NATIVE_CACHE:-}" ]]; then
  (cd "$native_root/src/cpp" && "$python_bin" setup.py build_ext --build-lib "$native_root/site")
  (cd "$proxy" && "$python_bin" setup.py build_ext --build-lib "$native_root/site")
  exit
fi

# Reuse only the pinned build's unchanged translation units. Recompile every
# translation unit whose class definitions or binding are different.
cache_src="$(realpath "$HTS_NATIVE_CACHE")/src"
cache_proxy="$cache_src/layers/extensions/inference"
diff -qr --exclude=build --exclude='*.egg-info' --exclude=__pycache__ --exclude='*.so' \
  "$upstream/cpp" "$cache_src/cpp"
diff -qr --exclude=build --exclude='*.egg-info' --exclude=__pycache__ --exclude='*.so' \
  --exclude=bind.cpp --exclude=dmc_ld_proxy.h --exclude=dmci_proxy.h \
  "$upstream/layers/extensions/inference" "$cache_proxy"
object_dir="$cache_proxy/build/temp.linux-x86_64-cpython-312"
mapfile -d '' objects < <(find "$object_dir" -type f -name '*.o' \
  ! -name 'bind.o' ! -name 'dmc_ld_proxy.o' ! -name 'dmci_proxy.o' \
  ! -name 'dmc_hts_proxy.o' ! -name 'dmc_htl_proxy.o' -print0 | sort -z)
test "${#objects[@]}" -ge 14
sha256sum "${objects[@]}" > "$native_root/cached_objects.sha256"
jobs=()
for stem in bind dmci_proxy dmc_ld_proxy dmc_hts_proxy dmc_htl_proxy; do
  c++ -c -std=c++17 -O3 -DNDEBUG -fPIC -pthread -Wno-deprecated-declarations \
    -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=inference_extensions_cuda \
    -I"$torch_path/include" -I"$torch_path/include/torch/csrc/api/include" -I"$toolkit/include" \
    -I"$native_root/src/cpp/py_rans" -I"$cutlass/include" -I"$cutlass/tools/util/include" -I"$python_headers" \
    "$proxy/$stem.cpp" -o "$native_root/$stem.o" &
  jobs+=("$!")
  objects+=("$native_root/$stem.o")
done
for job in "${jobs[@]}"; do wait "$job"; done
c++ -shared "${objects[@]}" -L"$torch_path/lib" -L"$toolkit/lib64" \
  -lc10 -ltorch -ltorch_cpu -ltorch_python -lc10_cuda -ltorch_cuda -lcudart \
  -o "$native_root/site/inference_extensions_cuda$suffix"
cp "$cache_src/cpp/build/lib.linux-x86_64-cpython-312/MLCodec_extensions_cpp$suffix" "$native_root/site/"
