"""Build the reproducibility manifests and rate-distortion admission evidence."""

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from statistics import mean

import numpy as np
from scipy.interpolate import PchipInterpolator

from experiments.pht_tok.study import ROOT, write_json

UPSTREAM = "cbdae87a5445114cdc7f48816da63ea80bdeac40"
OUT = ROOT / "results/pht_tok"
UPSTREAM_FILES = (
    "README.md", "training.md", "LICENSE.txt", "NOTICE.txt", "test_video.py", "test_compress_time.py",
    "src/models/video_model_ht.py", "src/models/common_model.py", "src/models/entropy_models.py",
    "src/utils/video_reader.py", "src/utils/transforms.py", "src/utils/stream_helper.py",
    "src/layers/extensions/inference/dmc_hts_proxy.cpp", "src/layers/extensions/inference/dmc_hts_proxy.h",
    "src/layers/extensions/inference/memory_pool.h", "src/cpp/py_rans/rans.cpp",
    "DCVC-family/DCVC-RT/src/layers/extensions/inference/common.h",
)
HISTORY = "7d9bbb471fd71555b963f4ecb51241363a9954dd"


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def fingerprint(path: Path) -> dict:
    return {"filename": path.name, "bytes": path.stat().st_size, "sha256": digest(path)}


def shell(*command: str) -> str:
    return subprocess.check_output(command, text=True).strip()


def assets(work: Path, weights: Path, cutlass: Path, nvcc: Path, history: Path) -> dict:
    import torch

    repo = ROOT / "third_party/DCVC"
    if shell("git", "-C", str(repo), "rev-parse", "HEAD") != UPSTREAM or shell(
        "git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"
    ):
        raise ValueError("official submodule identity changed")
    expected = read(ROOT / "configs/hla_tok/hts/official_weights.json")
    models = {key: fingerprint(weights / info["filename"]) | {"published_item": info["onedrive_item_id"]}
              for key, info in expected["weights"].items()}
    if any(models[k]["sha256"] != item["sha256"] for k, item in expected["weights"].items()):
        raise ValueError("official checkpoint SHA256 differs")
    historical_report = history / "results/hla_tok/hts_v2_1/final_report.md"
    history_sha = shell("git", "-C", str(history), "rev-parse", "HEAD")
    if history_sha != HISTORY:
        raise ValueError("historical branch HEAD changed")
    on_git = subprocess.check_output(["git", "-C", str(history), "show",
                                     "HEAD:results/hla_tok/hts_v2_1/final_report.md"])
    if hashlib.sha256(on_git).hexdigest() != digest(historical_report):
        raise ValueError("historical report differs from its commit")
    return {"official_repository": "https://github.com/microsoft/DCVC", "official_commit": UPSTREAM,
            "licenses": {"source": "MIT", "notice": "NOTICE.txt", "stream_helper": "Apache-2.0"},
            "weights_release": expected["official_share"], "weights": models,
            "source_files": {name: digest(repo / name) for name in UPSTREAM_FILES},
            "builds": {folder: [fingerprint(lib) for lib in sorted((work / folder).glob("*.so"))]
                       for folder in ("official/site", "pht/site", "pht-unoptimized-site", "rt-site")},
            "rt_weights": [fingerprint(path) for path in sorted((work / "rt-weights").glob("*.tar"))],
            "rt_weights_release": ("https://1drv.ms/f/c/2866592d5c55df8c/"
                                   "Esu0KJ-I2kxCjEP565ARx_YB88i0UnR6XnODqFcvZs4LcA?e=by8CO8"),
            "cutlass_commit": shell("git", "-C", str(cutlass), "rev-parse", "HEAD"),
            "toolchain": {"torch": torch.__version__, "torch_cuda": torch.version.cuda,
                          "gpu": torch.cuda.get_device_name(),
                          "driver": shell("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"),
                          "nvcc": shell(str(nvcc), "--version"), "python": sys.version,
                          "compiler": shell("c++", "--version").splitlines()[0]},
            "experiment_base": "e90ceda2d2875cd99061b8eb20946ebf67820a91",
            "historical_experiment": {"branch": "experiment/hla-tok-hts-v2.1", "head": history_sha,
                                      "final_report": fingerprint(historical_report)},
            "dataset_manifests": {key: digest(ROOT / f"data/manifests/cpat_{key}.json")
                                  for key in ("train", "validation", "test")},
            "uvg": {"source": "https://ultravideo.fi/dataset.html", "content": "ShakeNDry",
                    "license": "CC BY-NC 3.0, Digiturk", "tested_frames": 129, "original_frames": 300,
                    "resolution": [1080, 1920], "fps": 120,
                    "archive": fingerprint(work / "media/uvg-complete.7z"),
                    "yuv": fingerprint(work / "media/ShakeNDry_1920x1080_120fps_420_8bit_YUV.yuv")},
            "public_trace": {"repository": "https://github.com/ravinet/mahimahi",
                             "commit": "f1346c38c415abfc98626dd82450f49a57033603",
                             "artifact": fingerprint(work / "verizon-lte.down")}}


def baseline_rows(report: dict, dataset: str, archived: Path | None = None) -> list[dict]:
    results = []
    for model, entry in report["models"].items():
        for qp, points in entry["results"].items():
            seconds = sum(p["seconds"] for p in points)
            frames = sum(p["valid_frames"] for p in points)
            bits = sum(p["physical_bits"] for p in points)
            matches: list[bool] | None = [] if archived is not None else None
            if matches is not None and archived is not None:
                for index, point in enumerate(points):
                    prior = archived / model / f"official-i{int(qp):02}-p{int(qp):02}" / f"{index:02}.bin"
                    matches.append(prior.is_file() and digest(prior) == point["sha256"])
            results.append({"dataset": dataset, "model": model, "qp": int(qp), "source_frames": frames,
                            "physical_bytes": bits // 8, "source_seconds": seconds, "mbps": bits / seconds / 1e6,
                            "bpp": bits / (frames * 1080 * 1920),
                            "mean_official_psnr": mean(p["official"]["ave_all_frame_psnr"] for p in points),
                            "physical_archive_matches": matches, "sequences": points})
    return results


def rt_rows(work: Path) -> list[dict]:
    report = read(work / "rt-official.json")["Calibration"]
    rows = []
    for seq, points in report.items():
        for record in points.values():
            qp = (0, 16, 32, 48, 63)[record["rate_idx"]]
            raw = work / "RT-Calibration/Calibration" / f"{seq}_q{qp}.bin"
            frames = record["i_frame_num"] + record["p_frame_num"]
            bytes_sent = raw.stat().st_size
            rows.append({"dataset": "DAVIS Calibration", "model": "rt", "source": seq, "qp": qp,
                         "physical_bytes": bytes_sent, "source_frames": frames, "source_fps": 30,
                         "mbps": bytes_sent * 8 / (frames / 30) / 1e6,
                         "mean_official_psnr": record["ave_all_frame_psnr"], "sha256": digest(raw)})
    return rows


def baseline(work: Path, archive: Path) -> dict:
    calibration = read(work / "calibration.json")
    uvg = read(work / "uvg.json")
    rows = baseline_rows(calibration, "DAVIS Calibration", archive / "native_audit")
    agreement = sum(sum(row["physical_archive_matches"]) for row in rows)
    if agreement != 60:
        raise ValueError(f"only {agreement} of the 60 official Calibration streams match their archive")
    archive_study = read(archive / "native_audit/native_comparison.json")
    return {"status": "PASS", "matched_archived_streams": agreement,
            "calibration_sources": calibration["sources"], "uvg_sources": uvg["sources"],
            "uf_points": rows + baseline_rows(uvg, "UVG ShakeNDry"),
            "rt_points": rt_rows(work) if (work / "rt-official.json").is_file() else [],
            "official_commands": {group: {model: data["command"] for model, data in report["models"].items()}
                                  for group, report in (("calibration", calibration), ("uvg", uvg))},
            "historical_context": archive_study["options"], "validation_pixel_reads": 0, "test_pixel_reads": 0}


def official_curves(path: Path) -> list[dict]:
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    result = []
    for row in rows:
        if row["profile"] != "official" or row["qp_p"] != row["qp_i"]:
            continue
        result.append({"method": "official" if row["model"] == "hts" else f"official_{row['model']}",
                       "qp": int(row["qp_p"]), "stage": 4, "frames": int(row["frames"]),
                       "mbps": float(row["actual_mbps_source_fps"]), "bpp": float(row["bpp"]),
                       "global_psnr": float(row["global_psnr"]), "lpips": float(row["lpips"]),
                       "dists": float(row["dists"]), "origin": "archived official diagonal QP"})
    return result


def envelope(rows: list[dict], metric: str) -> tuple[list[dict], list[dict]]:
    sign = 1 if metric == "global_psnr" else -1
    accepted: list[dict] = []
    dominated: list[dict] = []
    ordered = sorted(rows, key=lambda row: (row["bpp"], -sign * row[metric]))
    for row in ordered:
        if accepted and (sign * row[metric] <= sign * accepted[-1][metric] or row["bpp"] == accepted[-1]["bpp"]):
            dominated.append(row)
        else:
            accepted.append(row)
    return accepted, dominated


def log_rate_curve(rows, metric):
    sign = 1 if metric == "global_psnr" else -1
    return np.array([sign * row[metric] for row in rows]), np.log([row["bpp"] for row in rows])


def bd_rate(reference: list[dict], candidate: list[dict], metric: str) -> dict:
    control, control_dominated = envelope(reference, metric)
    chosen, candidate_dominated = envelope(candidate, metric)
    a, x = log_rate_curve(control, metric)
    b, y = log_rate_curve(chosen, metric)
    observation = {"official_points": len(a), "candidate_points": len(b),
                   "dominated_official": [item["qp"] for item in control_dominated],
                   "dominated_candidate": [[item["qp"], item["stage"]] for item in candidate_dominated]}
    if min(len(a), len(b)) < 4 or max(a[0], b[0]) >= min(a[-1], b[-1]):
        return observation | {"status": "INSUFFICIENT_OVERLAP"}
    lo, hi = max(a[0], b[0]), min(a[-1], b[-1])
    extra = PchipInterpolator(b, y).integrate(lo, hi) - PchipInterpolator(a, x).integrate(lo, hi)
    sign = 1 if metric == "global_psnr" else -1
    return observation | {"status": "MEASURED", "quality_range": sorted((sign * lo, sign * hi)),
                          "delta_percent": float(np.expm1(extra / (hi - lo)) * 100)}


def equal_rate(reference, candidate, metric):
    control, _ = envelope(reference, metric)
    a, x = log_rate_curve(control, metric)
    sign = 1 if metric == "global_psnr" else -1
    quality = sign * candidate[metric]
    if not a[0] <= quality <= a[-1]:
        return None
    return float((candidate["bpp"] / np.exp(np.interp(quality, a, x)) - 1) * 100)


def local_nonmonotonic(units):
    stage_rows = {(r["source"], r["qp"], r["start"], r["method"], r["stage"]): r
                  for r in units if not r["intra"] and r["method"] != "official"}
    result = {}
    for method in ("MASK", "PRECISION"):
        result[method] = {}
        for metric, direction in (("psnr", 1), ("lpips", -1), ("dists", -1)):
            regressions = []
            for (*identity, stage), row in stage_rows.items():
                if identity[-1] != method or stage == 1:
                    continue
                before = stage_rows[(*identity, stage - 1)]["frame_metrics"]
                after = row["frame_metrics"]
                regressions.extend(direction * (b[metric] - a[metric]) < 0
                                   for a, b in zip(before, after, strict=True))
            result[method][metric] = {"fraction": mean(regressions), "comparisons": len(regressions)}
    return result


def rd(work: Path, archive: Path):
    fixed = read(work / "fixed.json")
    measured = fixed["curves"]
    old = official_curves(archive / "native_audit/qp_sweep.csv")
    official = [row for row in old if row["method"] == "official"]
    candidates = [row for row in measured if row["method"] != "official"]
    metrics = ("global_psnr", "lpips", "dists")
    matched = [{"method": row["method"], "qp": row["qp"], "stage": row["stage"],
                **{metric: equal_rate(official, row, metric) for metric in metrics}} for row in candidates]
    costs = []
    for row in candidates:
        if row["stage"] != 4:
            continue
        native = next(r for r in measured if r["method"] == "official" and r["qp"] == row["qp"])
        costs.append({"method": row["method"], "qp": row["qp"], "official_bytes": native["bytes"],
                      "progressive_bytes": row["bytes"], "extra_ratio": row["bytes"] / native["bytes"] - 1})
    low = local_nonmonotonic(fixed["units"])
    analysis = {"source": "128 Calibration frames, four sequences, Train pixels only",
                "curves": measured, "mid_equal_quality_rate_penalty_percent": matched,
                "bd_rate_by_stage": {method: {str(stage): {
                    metric: bd_rate(official, [r for r in candidates if r["method"] == method and
                                                r["stage"] == stage], metric) for metric in metrics}
                    for stage in range(1, 5)} for method in ("MASK", "PRECISION")},
                "local_adjacent_regression": low, "full_overhead": costs,
                "training": "frozen official checkpoints"}
    fields = ("method", "qp", "stage", "frames", "bytes", "bpp", "mbps", "global_psnr", "lpips", "dists",
              "ms-ssim", "origin")
    with (OUT / "rate_distortion.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(old + [row | {"origin": "PHT Calibration rerun"} for row in measured])
    ledgers = [row for row in fixed["units"] if "ledger" in row]
    for row in ledgers:
        a = row["ledger"]
        if a["physical_bytes"] != a["header_bytes"] + a["layer_header_bytes"] + sum(a["payload_bytes"]):
            raise ValueError("PHT physical byte accounting mismatch")
    write_json(OUT / "bitstream_ledger.json",
               {"complete_P_chunks": len(ledgers), "units": ledgers, "full_overhead": costs,
                "flush_bytes_in_payload_per_record": 4, "alignment_bytes": 0, "auxiliary_bits": 0,
                "reference_and_I_accounted_in_rates": True})
    write_json(OUT / "progressive_diagnosis.json", analysis)
    write_json(OUT / "calibration_units.json", fixed["units"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=ROOT / ".build/pht-tok")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--cutlass", type=Path, required=True)
    parser.add_argument("--nvcc", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    provenance = assets(args.work, args.weights, args.cutlass, args.nvcc, args.history)
    audition = baseline(args.work, args.archive)
    write_json(OUT / "upstream_manifest.json", provenance)
    write_json(OUT / "official_baseline_audit.json", audition)
    write_json(OUT / "baseline_manifest.json", {
        "status": audition["status"], "official_commands": audition["official_commands"],
        "weights": provenance["weights"], "datasets": provenance["dataset_manifests"],
        "calibration": "four native DAVIS2017 Train clips, 128 frames, nominal 30 FPS",
        "public_control": provenance["uvg"], "rt_native": bool(audition["rt_points"]),
        "untouched_history": provenance["historical_experiment"]})
    rd(args.work, args.archive)


if __name__ == "__main__":
    main()
