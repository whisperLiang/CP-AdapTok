"""Replay the published Original AdapTok window codec on Calibration."""

import argparse
from pathlib import Path

import torch
import yaml

from hla_tok.evaluation.io import write_csv, write_json
from hla_tok.evaluation.metrics import metric_hashes
from hla_tok.evaluation.official import CLIP_LENGTH, OfficialBaseline
from scripts.hla_tok.compact.references import OFFICIAL_CHECKPOINT, _official_summary, _official_window
from scripts.hla_tok.continuous.matrix import compute
from scripts.hla_tok.continuous.quality import NativeQuality
from scripts.hla_tok.uf.inputs import inputs
from scripts.hla_tok.v2.protocol import COHORT, ROOT, json_at, setup


def replay_clip(reader, clip, model, metric, config, budgets) -> tuple[list[dict], list[dict]]:
    reference = config["reference"]
    frames: list[dict] = []
    windows: list[dict] = []
    for start in range(clip.start, clip.start + clip.length, CLIP_LENGTH):
        stop = min(start + CLIP_LENGTH, clip.start + clip.length)
        images = [reader.read(clip.split, clip.video, i) for i in range(start, stop)]
        padded = torch.stack(images + [images[-1]] * (CLIP_LENGTH - len(images)))
        with compute(config["compute_dtype"]):
            target, decoded = _official_window(model, padded, start, len(images), reference, budgets)
        windows.extend({"video": clip.video, **entry} for _, entry in decoded)
        frames.extend(score_window(metric, clip.video, start, len(images), target, decoded))
    return frames, windows


def score_window(metric, video, start, length, original, decoded) -> list[dict]:
    rows = []
    for index in range(length):
        metric.prepare(original[index:index + 1])
        for rgb, entry in decoded:
            tags = {key: entry[key] for key in ("budget_per_group", "tokens_per_tile")}
            rows.append(tags | {"video": video, "frame": start + index}
                        | metric.score(rgb[index:index + 1]))
    return rows


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    setup()
    if metric_hashes() != json_at(COHORT)["metric_weights"]:
        raise ValueError("Original AdapTok must use the frozen perceptual weights")
    reader, plans = inputs()
    config = yaml.safe_load((ROOT / "configs/hla_tok/compact/protocol.yaml").read_text())
    manifest = json_at(OFFICIAL_CHECKPOINT)
    model = OfficialBaseline(ROOT / "third_party/AdapTok", Path(manifest["directory"]), manifest)
    reference = config["reference"]
    budgets = sorted({k if k <= model.ranks else k // model.groups for k in reference["original_budgets"]})
    metric = NativeQuality()
    rows: list[dict] = []
    windows: list[dict] = []
    with torch.cuda.stream(torch.cuda.Stream()):
        for clip in plans["calibration"]:
            clip_frames, clip_windows = replay_clip(reader, clip, model, metric, config, budgets)
            rows.extend(clip_frames)
            windows.extend(clip_windows)
            print(clip.video, clip.length, flush=True)
    write_csv(args.output / "original_per_frame.csv", rows)
    write_csv(args.output / "original_window_ledger.csv", windows)
    write_csv(args.output / "original_rd.csv", _official_summary(rows, windows))
    write_json(args.output / "pixel_reads.json", reader.counts)


if __name__ == "__main__":
    main()
