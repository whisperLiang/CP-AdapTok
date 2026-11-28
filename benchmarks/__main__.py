"""Entry points for the retained native 1080p benchmarks."""

import argparse
import importlib
import sys

SUITES = {
    "pht": "experiments.pht_tok.study",
    "pht-evaluate": "experiments.pht_tok.evaluate",
    "pht-codec": "experiments.pht_tok.examples",
    "pht-validation": "experiments.pht_tok.validation",
    "pht-latency": "experiments.pht_tok.measure",
    "network": "experiments.pht_tok.network",
    "dcvc": "baselines.dcvc_uf.evaluate",
    "adaptok": "baselines.adaptok.cli",
    "adaptok-latency": "baselines.adaptok.benchmark",
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
