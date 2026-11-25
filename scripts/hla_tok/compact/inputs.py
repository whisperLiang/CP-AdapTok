"""Freeze calibration samples before opening Validation pixels."""

from collections.abc import Iterator
from pathlib import Path
from random import Random

from hla_tok.evaluation.io import write_json
from hla_tok.evaluation.metrics import sha256_file
from hla_tok.progressive.geometry import NATIVE_SHAPE
from scripts.hla_tok.continuous.inputs import RESULTS as CONTINUOUS
from scripts.hla_tok.continuous.inputs import Inputs, open_inputs, require_digest
from scripts.hla_tok.continuous.ordering import load_orders
from scripts.hla_tok.v2.protocol import ROOT, frozen_data, json_at
from scripts.hla_tok.v3.protocol import read_config

CONFIG = ROOT / "configs/hla_tok/compact/protocol.yaml"
TRAIN_PLAN = CONFIG.with_name("train_frames.json")
OUTPUT = ROOT / "results/hla_tok/compact"


def _round_robin(catalog: dict, videos: list[str], rng: Random) -> list[dict]:
    indices = {video: list(range(len(catalog[video]["frame_hashes"]))) for video in videos}
    for values in indices.values():
        rng.shuffle(values)

    def frames() -> Iterator[dict]:
        for position in range(max(map(len, indices.values()), default=0)):
            for video, values in indices.items():
                if position < len(values):
                    i = values[position]
                    yield {"video": video, "frame": i, "source_sha256": catalog[video]["frame_hashes"][i]["sha256"]}

    return list(frames())


def train_plan(catalog: dict, knobs: dict) -> dict:
    rng = Random(knobs["seed"])
    videos = sorted(video for video, entry in catalog.items()
                    if entry["role"] == "train" and not entry["resized"] and entry["frame_hashes"]
                    and tuple(reversed(entry["resolution"])) == NATIVE_SHAPE)
    if not videos:
        raise ValueError("no eligible native Train frames")
    rng.shuffle(videos)
    held = min(knobs["holdout_sources"], len(videos) - 1)
    fit = _round_robin(catalog, videos[held:], rng)
    holdout = _round_robin(catalog, videos[:held], rng)
    if not held:
        split = min(knobs["holdout_frames"], max(1, len(fit) // 4))
        holdout, fit = fit[:split], fit[split:]
    if not fit or not holdout:
        raise ValueError("independent calibration requires at least two Train frames")
    fit_count = min(len(fit), knobs["fit_frames"])
    test_count = min(len(holdout), knobs["holdout_frames"])
    spare = knobs["fit_frames"] + knobs["holdout_frames"] - fit_count - test_count
    fit_extra = min(spare, len(fit) - fit_count)
    return {"fit": fit[:fit_count + fit_extra], "holdout": holdout[:test_count + spare - fit_extra]}


def frozen_plan() -> dict:
    reader, _, _ = frozen_data()
    config = read_config(CONFIG)
    return train_plan(reader.catalog("train"), config["calibration"]) | {
        "split": "train", "manifest_sha256": reader.hashes["train"], "native_shape": list(NATIVE_SHAPE),
        "compute_dtype": config["compute_dtype"], "protocol_sha256": sha256_file(CONFIG),
    }


def write_train_plan(path: Path = TRAIN_PLAN) -> None:
    plan = frozen_plan()
    if path.exists() and json_at(path) != plan:
        raise ValueError("Train selection differs from the frozen calibration plan")
    write_json(path, plan)


def compact_inputs(v3: Path, output: Path) -> tuple[Inputs, dict]:
    inputs = open_inputs(v3)
    replay = json_at(output / "reproduction.json")
    if replay["status"] != "PASS" or any(replay[key] != value for key, value in inputs.provenance().items()):
        raise ValueError("native replay must pass with the same frozen model and data")
    if json_at(TRAIN_PLAN) != frozen_plan():
        raise ValueError("the Train plan must be frozen before Validation")
    config = read_config(CONFIG)
    require_digest(CONTINUOUS / "rank_order_manifest.json", config["rank_order_manifest_sha256"])
    return inputs, config


def compact_orders(device: str = "cuda"):
    return {order.name.name: order for order in load_orders(CONTINUOUS, device)}


if __name__ == "__main__":
    write_train_plan()
