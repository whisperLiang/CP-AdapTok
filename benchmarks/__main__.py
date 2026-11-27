"""Entry points for the retained native 1080p benchmarks."""

import argparse
import importlib
import sys

SUITES = {
    "pht": "benchmarks.pht_tok.study",
    "pht-codec": "benchmarks.pht_tok.examples",
    "pht-validation": "benchmarks.pht_tok.validation",
    "pht-latency": "benchmarks.pht_tok.measure",
    "network": "benchmarks.pht_tok.network",
    "dcvc": "baselines.dcvc_uf.evaluate",
    "adaptok": "baselines.adaptok.cli",
    "adaptok-latency": "baselines.adaptok.benchmark",
    "d1": "baselines.hla_tok_uf_v2.evaluate",
    "d1-closure": "baselines.hla_tok_uf_v2.dynamic",
    "d1-incremental": "baselines.hla_tok_uf_v2.incremental_bench",
    "smoke-native": "benchmarks.smoke_native",
    "smoke-d1": "benchmarks.smoke_d1",
    "smoke-adaptok": "benchmarks.smoke_adaptok",
    "compare": "benchmarks.compare",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=SUITES)
    parser.add_argument("args", nargs=argparse.REMAINDER, help="arguments accepted by the selected suite")
    options = parser.parse_args()
    sys.argv = [f"{parser.prog} {options.suite}", *options.args]
    importlib.import_module(SUITES[options.suite]).main()


if __name__ == "__main__":
    main()
