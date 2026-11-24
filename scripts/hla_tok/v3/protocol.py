"""Immutable run definitions and pre-generated Train sample chains."""

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import yaml

from hla_tok import Geometry
from hla_tok.progressive.geometry import NATIVE_SHAPE
from hla_tok.training.train import TrainingFrames
from hla_tok.training.v3 import stratified_budget
from scripts.hla_tok.v2.protocol import ROOT

PROTOCOL = ROOT / "configs/hla_tok/v3/protocol.yaml"
OUTPUT = ROOT / "results/hla_tok/v3"
CHECKPOINTS = ROOT / "checkpoints/hla_tok/v3"
BASE = "50bd9cc6f28c6cde0e65f82723af2e7b11e48bd9"
STAGES = {"dense": ("A0", "A1"), "continuous": ("B0",),
          "quantized": ("C0", "C1"), "progressive": ("D",)}
CROP = Geometry(512, 512)
NATIVE = Geometry(*NATIVE_SHAPE)


@dataclass(frozen=True)
class Leg:
    updates: int
    geometry: Geometry
    freeze_spatial: bool
    learning_rate: float


@dataclass(frozen=True)
class Trial:
    stage: str
    arm: str
    seed: int
    variant: str

    def __post_init__(self):
        if self.arm not in STAGES[self.stage] or self.seed not in (1, 2) or self.variant not in ("D0", "D1"):
            raise ValueError("unknown trial outside the preregistered arms")

    @property
    def relative(self) -> Path:
        return Path(self.stage, f"seed{self.seed}", self.arm)

    @property
    def quantized(self) -> bool:
        return self.arm in ("C1", "D")


def read_config(path: Path) -> dict:
    return yaml.safe_load(path.read_bytes())


def curriculum(config: dict, stage: str) -> tuple[Leg, ...]:
    key = "progressive_schedule" if stage == "progressive" else stage
    return tuple(Leg(row["updates"], CROP if row["resolution"] == "crop" else NATIVE,
                     row.get("freeze_spatial", False),
                     config["native_learning_rate" if row["resolution"] == "native" else "learning_rate"])
                 for row in config[key])


def encode_record(sample: dict) -> bytes:
    return json.dumps(sample, sort_keys=True, separators=(",", ":")).encode()


def extend_chain(chain: str, sample: dict) -> str:
    return sha256(bytes.fromhex(chain) + encode_record(sample)).hexdigest()


def planned_crops(geometry: Geometry, seed: int, update: int) -> list[list[int]]:
    top = (17 * update + seed) % (geometry.height - 255)
    if geometry == NATIVE:
        half = geometry.width // 2
        return [[top, (71 * update + seed) % (half - 255)],
                [(top + 397) % (geometry.height - 255), half + (113 * update + seed) % (half - 255)]]
    return [[top, (71 * update + seed) % (geometry.width - 255)]]


class SamplePlan:
    def __init__(self, reader, schedule, seed, config, *, progressive=False):
        self.frames = TrainingFrames(reader, schedule, seed)
        self.records = [
            self.frames.plan(update) | {
                "split": "train",
                "perceptual_crops_yx": planned_crops(geometry, seed, update),
                "native_budget": stratified_budget(seed, update, tuple(config["progressive"]["proportions"]))
                if progressive else config["native_budget"],
            } for update, geometry in enumerate(schedule)
        ]
        self.chains = [bytes(32).hex()]
        for record in self.records:
            self.chains.append(extend_chain(self.chains[-1], record))

    def save(self, path: Path):
        payload = b"".join(encode_record(record) + b"\n" for record in self.records)
        if path.is_file():
            if path.read_bytes() != payload:
                raise ValueError("existing Train plan does not match the pre-generated sample chain")
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(payload)
        temporary.replace(path)

    def _load(self, record):
        if record["split"] != "train":
            raise PermissionError("optimization only accepts the frozen Train plan")
        return self.frames.load(record)

    def batches(self, start: int, stop: int):
        if start >= stop:
            return
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(self._load, self.records[start])
            for update in range(start + 1, stop):
                current = pending.result()
                pending = pool.submit(self._load, self.records[update])
                yield current
            yield pending.result()
