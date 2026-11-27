#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
case "${1:-}" in
  official) bash benchmarks/pht_tok/native/build.sh official ;;
  pht)
    bash benchmarks/pht_tok/native/build.sh pht
    bash benchmarks/pht_tok/native/build_entropy.sh
    ;;
  d1)
    bash baselines/hla_tok_uf_v2/build_native.sh
    bash baselines/hla_tok_uf_v2/build_records.sh
    bash baselines/hla_tok_uf_v2/build_legacy_rans.sh
    ;;
  *) printf '%s\n' 'usage: scripts/build_native.sh {official|pht|d1}' >&2; exit 2 ;;
esac
