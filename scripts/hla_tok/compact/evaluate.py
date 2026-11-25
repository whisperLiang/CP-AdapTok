"""Evaluate the frozen Compact profiles and build byte-matched tables."""

from pathlib import Path

from hla_tok.compression.profiles import PROFILES_BY_LABEL
from hla_tok.evaluation.io import write_csv
from hla_tok.streaming.compact_wire import max_budget
from hla_tok.streaming.continuous import MAX_K
from scripts.hla_tok.compact.calibrate import quantized_assets
from scripts.hla_tok.compact.fixed_reference import HEADER_BYTES as FIXED13_HEADER_BYTES
from scripts.hla_tok.compact.inputs import OUTPUT, compact_inputs
from scripts.hla_tok.compact.matrix import Case, evaluate, summarize_matrix
from scripts.hla_tok.continuous.inputs import Inputs, locked_output, run_arguments


def formal_cases(config: dict, assets: dict) -> list[Case]:
    wanted = set(config["quantization"]["profiles"])
    return [Case(key, "O2", budget) for key, spec in assets.items() if spec["profile"] in wanted
            for budget in config["budgets"]]


def _caps(config: dict, order: str) -> list[tuple[float, int]]:
    allowed = config["quantization"]["o1_caps"] if order == "O1" else config["frame_wire_byte_caps"]
    bands = zip(config["bandwidth_mbps"], config["frame_wire_byte_caps"], strict=True)
    return [(mbps, cap) for mbps, cap in bands if cap in allowed]


def plan(config: dict, assets: dict) -> list[dict]:
    settings = []
    payload_cap = config["equal_payload"]["cap_bytes"]
    wanted = set(config["quantization"]["profiles"])
    for key, spec in assets.items():
        if spec["profile"] not in wanted:
            continue
        profile = PROFILES_BY_LABEL[spec["profile"]]
        for order in ("O2", "O1"):
            for mbps, cap in _caps(config, order):
                settings.append({"asset": key, "order": order, "budget": max_budget(cap, profile),
                                 "basis": "total_wire", "cap_bytes": cap, "target_mbps": mbps})
            settings.append({"asset": key, "order": order, "basis": "payload_only", "cap_bytes": payload_cap,
                             "budget": max_budget(payload_cap, profile, payload_only=True), "target_mbps": ""})
            full_cap = payload_cap + FIXED13_HEADER_BYTES
            settings.append({"asset": key, "order": order, "basis": "total_wire_f13w", "cap_bytes": full_cap,
                             "budget": max_budget(full_cap, profile), "target_mbps": ""})
    return settings


def _rate_row(point: dict, data: dict, wire_sizes: list[int]) -> dict:
    cap = point["cap_bytes"]
    actual = max(wire_sizes)
    excess = sum(size > cap for size in wire_sizes)
    if excess:
        raise ValueError("measured packet exceeds the frozen byte cap")
    return data | point | {"max_bytes_per_frame": actual, "unused_bytes": cap - actual,
                           "overflow_frame_fraction": excess / len(wire_sizes),
                           "rate_feasibility": "SATURATED" if point["budget"] == MAX_K else "FEASIBLE"}


def _bandwidth_tables(output: Path, points: list[dict], rows: list[dict], summary: list[dict]) -> None:
    from collections import defaultdict

    lengths: dict[tuple, list[dict]] = defaultdict(list)
    def key(row: dict) -> tuple:
        return row["asset"], row["order"], row["budget"]

    aggregate = {key(row): row for row in summary}
    for row in rows:
        lengths[key(row)].append(row)
    scored = []
    for point in points:
        condition = key(point)
        basis = "payload_bytes" if point["basis"] == "payload_only" else "total_bytes"
        scored.append(_rate_row(point, aggregate[condition], [r[basis] for r in lengths[condition]]))
    write_csv(output / "bandwidth_constrained_quality.csv", [r for r in scored if r["basis"] == "total_wire"])
    write_csv(output / "compact_equal_caps.csv", [r for r in scored if r["target_mbps"] == "" or r["cap_bytes"] == 26_520])


def run(inputs: Inputs, config: dict, output: Path) -> None:
    assets = quantized_assets(inputs, config, output)
    formal = formal_cases(config, assets)
    grid = plan(config, assets)
    cases = formal + [Case(row["asset"], row["order"], row["budget"]) for row in grid]
    frames = evaluate(inputs, config, assets, output, "scalar", cases)
    stats = summarize_matrix(frames)
    write_csv(output / "scalar_quantization_per_frame.csv", frames)
    write_csv(output / "scalar_quantization_rd.csv", stats)
    write_csv(output / "scalar_quantization_by_video.csv", summarize_matrix(frames, by_video=True))
    _bandwidth_tables(output, grid, frames, stats)
    inputs.finish(output / "scalar_inputs.json")


def main() -> None:
    parser = run_arguments(__doc__)
    parser.set_defaults(output=OUTPUT)
    args = parser.parse_args()
    with locked_output(args.output):
        inputs, config = compact_inputs(args.v3, args.output)
        run(inputs, config, args.output)


if __name__ == "__main__":
    main()
