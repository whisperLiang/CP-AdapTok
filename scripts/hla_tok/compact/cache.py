"""Native Train feature cache with explicit complete and partial Rank storage."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.lib.format import open_memmap
from torch import Tensor

from hla_tok.evaluation.benchmark import transfer
from hla_tok.evaluation.io import write_json
from hla_tok.evaluation.metrics import sha256_file
from hla_tok.models.local import TOKEN_DIM
from hla_tok.streaming.continuous import FULL_RANKS, REGIONS, TAIL_REGIONS, encode_continuous
from scripts.hla_tok.compact.inputs import TRAIN_PLAN
from scripts.hla_tok.continuous.inputs import Inputs
from scripts.hla_tok.continuous.matrix import compute
from scripts.hla_tok.v2.protocol import ROOT, json_at


@dataclass(frozen=True)
class TrainTokens:
    full: Tensor
    tail: Tensor
    fit_frames: int

    def native(self) -> Tensor:
        full = self.full.transpose(1, 2).flatten(1, 2)
        return torch.cat((full, self.tail.squeeze(2)), dim=1)

    def rank(self, index: int) -> Tensor:
        return self.tail[:, :, 0] if index == FULL_RANKS else self.full[:, :, index]


def _identity(inputs: Inputs, config: dict) -> dict:
    paths = ["hla_tok/models/codec.py", "hla_tok/models/local.py", "hla_tok/models/spatial.py",
             "hla_tok/streaming/continuous.py", "scripts/hla_tok/compact/cache.py"]
    return inputs.provenance() | {
        "sample_plan_sha256": sha256_file(TRAIN_PLAN), "compute_dtype": config["compute_dtype"],
        "storage_dtype": "bf16 bit patterns in little-endian uint16",
        "geometry": config["native_shape"], "torch": str(torch.__version__),
        "gpu": torch.cuda.get_device_name(), "source_sha256": {p: sha256_file(ROOT / p) for p in paths},
    }


@torch.inference_mode()
def _write_cache(inputs: Inputs, config: dict, folder: Path, frames: list[dict]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    shapes = [(len(frames), REGIONS, FULL_RANKS, TOKEN_DIM), (len(frames), TAIL_REGIONS, 1, TOKEN_DIM)]
    full, tail = (open_memmap(folder / f"{name}.tmp.npy", mode="w+", dtype="<u2", shape=shape)
                  for name, shape in zip(("full", "tail"), shapes, strict=True))
    for i, row in enumerate(frames):
        pixels = transfer(inputs.reader.read("train", row["video"], row["frame"])[None])
        with compute(config["compute_dtype"]):
            tokens = encode_continuous(inputs.model, pixels).to(torch.bfloat16)
        words = tokens[0].view(torch.int16).cpu().numpy().view(np.uint16)
        complete = FULL_RANKS * REGIONS
        full[i] = words[:complete].reshape(FULL_RANKS, REGIONS, TOKEN_DIM).transpose(1, 0, 2)
        tail[i, :, 0] = words[complete:]
    full.flush()
    tail.flush()
    for name in ("full", "tail"):
        (folder / f"{name}.tmp.npy").replace(folder / f"{name}.npy")


def load_train_tokens(inputs: Inputs, config: dict, output: Path) -> TrainTokens:
    plan = json_at(TRAIN_PLAN)
    folder = output / "cache"
    manifest = folder / "identity.json"
    identity = _identity(inputs, config)
    paths = [folder / name for name in ("full.npy", "tail.npy")]
    if not manifest.is_file():
        _write_cache(inputs, config, folder, plan["fit"] + plan["holdout"])
        write_json(manifest, identity | {"cache_sha256": {p.name: sha256_file(p) for p in paths}})
    saved = json_at(manifest)
    if saved != identity | {"cache_sha256": {p.name: sha256_file(p) for p in paths}}:
        raise ValueError("Train feature cache provenance differs")
    values = [torch.from_numpy(np.load(p, mmap_mode="c", allow_pickle=False).view("=i2"))
              .view(torch.bfloat16).to("cuda") for p in paths]
    return TrainTokens(values[0], values[1], len(plan["fit"]))
