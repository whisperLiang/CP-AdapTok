"""Frozen data, validation clips, and deployment assets for HLT v1."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from random import Random

import numpy as np
import torch

from hla_tok.evaluation.benchmark import transfer
from hla_tok.evaluation.data import AuditedFrames
from hla_tok.evaluation.io import write_json
from hla_tok.evaluation.metrics import sha256_file
from hla_tok.models.local import LocalTokenDecoder
from hla_tok.models.spatial import RGBDecoder
from hla_tok.streaming.continuous import encode_continuous
from hla_tok.temporal.allocation import base_scan, scan_table
from hla_tok.temporal.profile import Profile
from hla_tok.temporal.transform import NativeTransform, TemporalSynthesis
from hla_tok.training.checkpoint import read_checkpoint
from scripts.hla_tok.compact.inputs import OUTPUT as COMPACT
from scripts.hla_tok.compact.inputs import TRAIN_PLAN, compact_orders
from scripts.hla_tok.continuous.inputs import Inputs, require_digest
from scripts.hla_tok.continuous.matrix import compute
from scripts.hla_tok.v2.protocol import ROOT, json_at

CONFIG = ROOT / "configs/hla_tok/temporal/protocol.json"
RESULTS = ROOT / "results/hla_tok/temporal"
WORK = ROOT / ".build/temporal"


@dataclass(frozen=True)
class Clip:
    split: str
    video: str
    start: int
    length: int

    def manifest(self, reader: AuditedFrames) -> dict:
        row = reader.catalog(self.split)[self.video]
        selection = row["frame_hashes"][self.start:self.start + self.length]
        indexes = [int(Path(item["file"]).stem) for item in selection]
        if len(indexes) != self.length or indexes != list(range(indexes[0], indexes[0] + self.length)):
            raise ValueError("temporal clip has a frame gap or crosses the source boundary")
        if row["resized"] or row["resolution"] != [1920, 1080]:
            raise ValueError("temporal clip must be native 1080p")
        return vars(self) | {"fps": row["fps"], "fps_nominal": row["fps_nominal"], "frames": [
            {"index": self.start + i, "source_index": index, "timestamp_seconds_nominal": index / row["fps"],
             "sha256": selection[i]["sha256"]} for i, index in enumerate(indexes)
        ]}


def training_plan(reader: AuditedFrames, config: dict) -> dict[str, list[Clip]]:
    original = json_at(TRAIN_PLAN)
    rng = Random(config["seed"])
    frames = max(config["train_window_lengths"])
    catalog = reader.catalog("train")
    chosen = {}
    for group in ("fit", "holdout"):
        sources = sorted({sample["video"] for sample in original[group]
                          if len(catalog[sample["video"]]["frame_hashes"]) >= frames})
        selected = rng.sample(sources, config["train_calibration_sources"][group])
        chosen[group] = [Clip("train", video, rng.randrange(len(catalog[video]["frame_hashes"]) - frames + 1), frames)
                         for video in selected]
    if {c.video for c in chosen["fit"]} & {c.video for c in chosen["holdout"]}:
        raise ValueError("temporal fitting and calibration must have different Train sources")
    return chosen


def validation_plan(inputs: Inputs) -> tuple[list[Clip], list[Clip]]:
    primary = [Clip("validation", video, 0, frames) for video, frames in inputs.cohort.items()]
    catalog = inputs.reader.catalog("validation")
    extra = [Clip("validation", video, 0, len(catalog[video]["frame_hashes"]))
             for video in ("DAVIS2017:soapbox", "DAVIS2017:dance-twirl")]
    return primary, extra


def freeze_clips(inputs: Inputs, output: Path, config: dict) -> dict[str, list[Clip]]:
    train = training_plan(inputs.reader, config)
    primary, long = validation_plan(inputs)
    timestamp = {"source_order": "ascending original frame index", "timestamp_basis": "manifest nominal fps"}
    plans = {group: [Clip(c.split, c.video, c.start + offset, width).manifest(inputs.reader)
                     for c in selected for width in config["train_window_lengths"]
                     for offset in range(0, c.length, width)] for group, selected in train.items()}
    path = output / "sequence_manifest.json"
    manifest = {"dataset_sha256": inputs.reader.hashes, "training": plans,
                "validation": [c.manifest(inputs.reader) for c in primary],
                "long_validation": [c.manifest(inputs.reader) for c in long]} | timestamp
    if path.exists() and json_at(path) != manifest:
        raise ValueError("temporal manifest identity differs")
    write_json(path, manifest)
    return train


def frozen_transform(device: str = "cuda") -> tuple[NativeTransform, str]:
    manifest = json_at(COMPACT / "projection_assets.json")
    spec = manifest["PCA32-BF16_rank"]
    path = COMPACT / "profiles" / spec["filename"]
    digest = require_digest(path, spec["sha256"])
    with np.load(path, allow_pickle=False) as asset:
        transform = NativeTransform(torch.from_numpy(asset["center"]), torch.from_numpy(asset["projection"]))
    return transform.to(device), digest


@torch.no_grad()
def project_source(inputs: Inputs, transform: NativeTransform, rgb: torch.Tensor) -> np.ndarray:
    with compute("bf16"):
        native = encode_continuous(inputs.model, rgb)
    return transform.project(native)[0].cpu().numpy()


def source_frame(reader: AuditedFrames, clip: Clip, offset: int) -> torch.Tensor:
    return transfer(reader.read(clip.split, clip.video, clip.start + offset)[None])


@dataclass(frozen=True)
class Setting:
    label: str
    base: str
    scan: str
    predictor: str
    sparse: bool = True
    bits: int = 4
    block_size: int = 256
    refresh: int = 32
    scene: bool = False
    score_basis: str | None = None


class Assets:
    def __init__(self, output: Path, device: str = "cuda"):
        self.transform, pca = frozen_transform(device)
        self.config = json_at(CONFIG)
        self.identity = json_at(output / "temporal_assets.json")
        self.path = output / "lattice.npz"
        file_sha = require_digest(self.path, self.identity["lattice_sha256"])
        order = COMPACT.parent / "continuous/rank_order_manifest.json"
        order_sha = require_digest(order, self.identity["rank_order_sha256"])
        if self.identity["pca_sha256"] != pca or self.identity["protocol_sha256"] != sha256_file(CONFIG):
            raise ValueError("deployment transform or protocol identity differs")
        immutable_parts = (file_sha, pca, order_sha, self.identity["protocol_sha256"], self.identity["model_sha256"])
        self.binding = sha256(b"".join(bytes.fromhex(digest) for digest in immutable_parts)).digest()
        with np.load(self.path, allow_pickle=False) as archive:
            self.arrays = {name: archive[name] for name in archive.files}
        self.order = compact_orders("cpu")["O2"]
        self.device = device

    def profile(self, setting: Setting) -> Profile:
        knobs = self.config["base_profiles"][setting.base]
        base = base_scan(self.order, knobs["cap"], knobs["ranks"], knobs["components"],
                         setting.bits, setting.block_size)
        key = setting.score_basis or ("absolute" if setting.predictor == "absolute" else "temporal")
        score = None if setting.scan == "S0" else self.arrays[f"{setting.scan}_{key}"]
        scan = scan_table(self.order, base, score)
        scene = self.config["refresh_scene_threshold"] if setting.scene else None
        return Profile(int(setting.base[1:]), scan, len(base), self.arrays[f"i{setting.bits}"],
                       self.arrays[f"p{setting.bits}"], self.binding, setting.bits, setting.block_size,
                       setting.sparse, setting.predictor, setting.refresh, scene,
                       self.config["linear_bound_steps"], knobs["cap"])

    def synthesis(self, model) -> TemporalSynthesis:
        return TemporalSynthesis.from_model(model, self.transform,
                                            torch.as_tensor(self.arrays["scale"], device=self.device))

    def receive_only(self, checkpoint: Path) -> TemporalSynthesis:
        require_digest(checkpoint, self.identity["checkpoint_sha256"])
        state = read_checkpoint(checkpoint).state["model"]
        local, rgb = LocalTokenDecoder(), RGBDecoder("D0")
        for name, module in (("local_decoder.", local), ("rgb_decoder.", rgb)):
            module.load_state_dict({key.removeprefix(name): value for key, value in state.items()
                                    if key.startswith(name)}, strict=True)
        decoder = TemporalSynthesis(local, rgb, self.transform, torch.as_tensor(self.arrays["scale"], device=self.device))
        return decoder.eval().requires_grad_(False).to(device=self.device, memory_format=torch.channels_last)
