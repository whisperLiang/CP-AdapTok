"""Replay the frozen CP frontiers and C1 on exactly the 174 admitted Validation frames."""

from pathlib import Path

import torch

from hla_tok.evaluation.io import write_csv, write_json
from hla_tok.streaming.continuous import MAX_K, NATIVE, encode_continuous
from hla_tok.streaming.continuous_wire import wire_rate
from hla_tok.streaming.fixed13 import pack_tokens_13bit
from scripts.hla_tok.compact.fixed_reference import FixedWindow, open_window
from scripts.hla_tok.compact.inputs import OUTPUT as ARCHIVE
from scripts.hla_tok.compact.inputs import compact_inputs
from scripts.hla_tok.compact.matrix import Case, Matrix
from scripts.hla_tok.compact.references import verify_archive
from scripts.hla_tok.continuous.inputs import METRICS, load_model, run_arguments, verify_model_state
from scripts.hla_tok.continuous.matrix import compute, groups, validation_frames
from scripts.hla_tok.continuous.quality import summarize
from scripts.hla_tok.v2.protocol import ROOT, csv_at, json_at

TEMPORAL = ROOT / "results/hla_tok/temporal"
ADAPTER = "T1_Q8-4_rank_seed64997_step1000"


def _signed(row: dict) -> tuple[float, float, float]:
    return float(row["global_psnr"]), -float(row["lpips"]), -float(row["dists"])


def _dominates(a: dict, b: dict) -> bool:
    gains = tuple(x - y for x, y in zip(_signed(a), _signed(b), strict=True))
    return min(gains) >= 0 and max(gains) > 0


def comparison_points() -> list[dict]:
    bandwidth = csv_at(ARCHIVE / "bandwidth_constrained_quality.csv") + csv_at(ARCHIVE / "training_final_bandwidth.csv")
    points = []
    for row in bandwidth:
        row["cap_bytes"] = int(row.get("cap_bytes") or row["cap"])
        if row["basis"] != "total_wire":
            continue
        cap = row["cap_bytes"]
        required = row["order"] == "O2" and row["asset"] in {"Q16-4_rank", ADAPTER}
        peer = [r for r in bandwidth if int(r.get("cap_bytes") or r["cap"]) == cap and r["basis"] == "total_wire"]
        if required or (cap in {20833, 26520, 28080, 41666, 83333, 166666}
                        and not any(_dominates(other, row) for other in peer)):
            points.append(row)
    for basis, cap in (("payload_only", 26520), ("total_wire_f13w", 26544)):
        peers = [r for r in csv_at(ARCHIVE / "compact_equal_caps.csv") if r["basis"] == basis]
        for row in peers:
            if row["asset"] == "Q16-4_rank" and row["order"] == "O2" or not any(_dominates(r, row) for r in peers):
                points.append(row | {"cap_bytes": cap})
    adapter_payload, = [r for r in bandwidth if r["basis"] == "payload_only" and r["asset"] == ADAPTER]
    return points + [adapter_payload]


def _key(row: dict) -> tuple[str, str, int]:
    return row["asset"], row["order"], int(row["budget"])


def _audit(expected: dict, rows: list[dict], tolerances: dict) -> dict:
    measured = summarize(rows)
    errors = {metric: abs(float(expected[metric]) - measured[metric]) for metric in METRICS}
    payload = sum(entry["payload_bytes"] for entry in rows)
    total = sum(entry["total_bytes"] for entry in rows)
    parity = all(value <= tolerances[metric] for metric, value in errors.items())
    bound = "payload_bytes" if expected["basis"] == "payload_only" else "total_bytes"
    cap = int(expected["cap_bytes"])
    physical = max(row[bound] for row in rows) <= cap and (payload, total) == tuple(
        int(expected[field]) for field in ("payload_bytes", "total_bytes"))
    return measured | wire_rate(payload, total, len(rows)) | {
        "expected": {metric: float(expected[metric]) for metric in METRICS},
        "absolute_delta": errors, "cap_bytes": cap, "basis": expected["basis"],
        "status": "PASS" if parity and physical and len(rows) == 174 else "FAIL",
    }


@torch.inference_mode()
def replay(v3: Path, native_replay: Path, output: Path) -> None:
    inputs, config = compact_inputs(v3, native_replay)
    c1, checkpoint = load_model(v3, "C1", inputs.config)
    decoder = c1.receiver()
    specs = json_at(ARCHIVE / "compact_profile_manifest.json") | {
        ADAPTER: json_at(ARCHIVE / "training_selection.json")["spec"],
    }
    points = comparison_points()
    matrix = Matrix(inputs, config, specs, [Case(*_key(row)) for row in points], ARCHIVE)
    rows, fixed = [], []
    for video, count in inputs.cohort.items():
        for _, frame, rgb in validation_frames(inputs, ((video, i) for i in range(count))):
            matrix.quality.prepare(rgb)
            with compute("bf16"):
                full = encode_continuous(inputs.model, rgb)
                ordered = {name: matrix.orders[name].apply(full) for _, name in matrix.groups}
                ids = c1.encode_training(rgb, MAX_K, quantize=True).ids
                if ids is None:
                    raise RuntimeError("frozen C1 did not return Fixed13 indices")
                raw = FixedWindow(frame, 1, 1, 1, MAX_K).packet(pack_tokens_13bit(ids[0].cpu().tolist()))
                _, payload = open_window(raw)
                pixels = decoder.decode_bytes(payload, MAX_K, NATIVE)
            fixed.append({"video": video, "frame": frame, "payload_bytes": len(payload), "total_bytes": len(raw)}
                         | matrix.quality.score(pixels))
            for key, budgets in matrix.groups.items():
                rows.extend(matrix._score_cases(key, budgets, ordered[key[1]], video, frame))
            if (frame + 1) % 16 == 0:
                print(f"CP/C1 replay {video} {frame + 1}/{count}", flush=True)
    write_csv(output / "baselines_per_frame.csv", rows)
    write_csv(output / "c1_per_frame.csv", fixed)
    indexed = {_key(tags): items for tags, items in groups(rows, ("asset", "order", "budget"))}
    tolerances = inputs.config["reproduction"]["tolerances"]
    checked = [{"asset": row["asset"], "order": row["order"], "budget": int(row["budget"])}
               | _audit(row, indexed[_key(row)], tolerances) for row in points]
    expected, = (row for row in csv_at(ARCHIVE / "single_frame_references.csv") if row["method"] == "C1 Fixed13")
    c1_quality = _audit(expected | {"basis": "total_wire", "cap_bytes": 26544}, fixed, tolerances)
    verify_model_state(c1, checkpoint["state_dict_sha256"])
    inputs.finish(output / "baseline_inputs.json")
    original = verify_archive(inputs, config)
    archived_original = {name: json_at(ARCHIVE / name) for name in ("original_temporal_scope.json",)}
    result = {"status": "PASS" if all(row["status"] == "PASS" for row in [*checked, c1_quality]) else "FAIL",
              "c0": json_at(native_replay / "reproduction.json"), "c1_fixed13": c1_quality,
              "compact": checked, "original_fixed13_archive": original | archived_original,
              "tolerances": tolerances, "test_pixel_reads": inputs.reader.counts["test"]}
    write_json(output / "baseline_reproduction.json", result)
    if result["status"] != "PASS":
        raise RuntimeError("CP or C1 differs from the frozen replay")


if __name__ == "__main__":
    parser = run_arguments(__doc__)
    parser.set_defaults(output=TEMPORAL)
    parser.add_argument("--native-replay", type=Path, default=Path(".build/temporal/baseline"))
    options = parser.parse_args()
    replay(options.v3, options.native_replay, options.output)
