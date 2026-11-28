#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

variant="${1:-pht}"
case "$variant" in
  official|pht|dual) ;;
  *) printf '%s\n' 'usage: build.sh [official|pht|dual]' >&2; exit 2 ;;
esac
python_bin="${PYTHON_BIN:-python}"
cutlass="$(realpath "${CUTLASS_HOME:?set CUTLASS_HOME to the pinned CUTLASS checkout}")"
export CUDA_HOME="$(realpath "${CUDA_HOME:?set CUDA_HOME to the pinned CUDA toolkit}")"
export PATH="$(dirname "$python_bin"):$CUDA_HOME/bin:$PATH"
export MAX_JOBS="${MAX_JOBS:-6}"
build="$PWD/.build/pht-tok/$variant"
source_dir="$PWD/third_party/DCVC/src"
proxy="$build/src/layers/extensions/inference"
env_version="$("$python_bin" -c 'from sysconfig import get_config_var; print(get_config_var("py_version_nodot"))')"
torch_dir="$("$python_bin" -c 'import torch; from pathlib import Path; print(Path(torch.__file__).parent)')"
headers="$("$python_bin" -c 'from sysconfig import get_path; print(get_path("include"))')"
suffix="$("$python_bin" -c 'from sysconfig import get_config_var; print(get_config_var("EXT_SUFFIX"))')"
cuda_include="$("$python_bin" -c 'from sysconfig import get_path; print(get_path("purelib") + "/nvidia/cu13/include")')"
export CPLUS_INCLUDE_PATH="$cuda_include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
mkdir -p "$proxy" "$build"/{src/cpp,third_party,site}
ln -sfn "$cutlass" "$build/third_party/cutlass"
cp -a "$source_dir/cpp/." "$build/src/cpp/"
cp -a "$source_dir/layers/extensions/inference/." "$proxy/"

if [[ "$variant" != official ]]; then
  patch --batch --directory="$proxy" -p1 --fuzz=0 < experiments/pht_tok/native/native_state_audit.patch
  patch --batch --directory="$proxy" -p1 --fuzz=0 < experiments/pht_tok/native/hts_adapter.patch
fi
if [[ "$variant" == dual ]]; then
  patch --batch --directory="$proxy" -p1 --fuzz=0 < experiments/pht_tok/native/ld_adapter.patch
fi

if [[ -z "${PHT_NATIVE_CACHE:-}" ]]; then
  (cd "$build/src/cpp" && "$python_bin" setup.py build_ext --build-lib "$build/site")
  (cd "$proxy" && "$python_bin" setup.py build_ext --build-lib "$build/site")
  exit
fi

cache="$(realpath "$PHT_NATIVE_CACHE")/src"
cache_proxy="$cache/layers/extensions/inference"
diff -qr --exclude=build --exclude='*.egg-info' --exclude=__pycache__ --exclude='*.so' \
  "$source_dir/cpp" "$cache/cpp"
diff -qr --exclude=build --exclude='*.egg-info' --exclude=__pycache__ --exclude='*.so' \
  --exclude=bind.cpp --exclude=dmc_ld_proxy.h --exclude=dmci_proxy.h \
  "$source_dir/layers/extensions/inference" "$cache_proxy"

mapfile -d '' objects < <(find "$cache_proxy/build/temp.linux-x86_64-cpython-$env_version" \
  -type f -name '*.o' ! -name 'bind.o' ! -name '*_proxy.o' ! -name 'memory_pool.o' -print0 | sort -z)
test "${#objects[@]}" -ge 11
sha256sum "${objects[@]}" > "$build/cached_objects.sha256"
jobs=()
for stem in bind memory_pool layers_proxy dmci_proxy dmc_ld_proxy dmc_hts_proxy dmc_htl_proxy; do
  c++ -c -std=c++17 -O3 -DNDEBUG -fPIC -pthread -Wno-deprecated-declarations \
    -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=inference_extensions_cuda \
    -I"$torch_dir/include" -I"$torch_dir/include/torch/csrc/api/include" -I"$CUDA_HOME/include" \
    -I"$build/src/cpp/py_rans" -I"$cutlass/include" -I"$cutlass/tools/util/include" -I"$headers" \
    "$proxy/$stem.cpp" -o "$build/$stem.o" &
  jobs+=("$!")
  objects+=("$build/$stem.o")
done
for job in "${jobs[@]}"; do wait "$job"; done
c++ -shared "${objects[@]}" -L"$torch_dir/lib" -L"$CUDA_HOME/lib64" \
  -lc10 -ltorch -ltorch_cpu -ltorch_python -lc10_cuda -ltorch_cuda -lcudart \
  -o "$build/site/inference_extensions_cuda$suffix"
cp "$cache/cpp/build/lib.linux-x86_64-cpython-$env_version/MLCodec_extensions_cpp$suffix" "$build/site/"
