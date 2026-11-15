"""Regenerate curves and reports from actual recorded evidence."""

import argparse

from cp_adaptok.large_scale import experiment
from cp_adaptok.large_scale_analysis import build_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    manifest, _ = experiment(args.run_id)
    summary = build_report(manifest)
    print(f"run_id={summary['run_id']} final_test_complete={summary['final_test_complete']}")


if __name__ == "__main__":
    main()
