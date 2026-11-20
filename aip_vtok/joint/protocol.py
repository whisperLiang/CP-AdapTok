"""Frozen assets, native split isolation, and experiment ownership."""

import fcntl
import hashlib
import random
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

from aip_vtok.c1.artifacts import FITTED, frozen_inputs, load_frozen
from aip_vtok.c1_followup.io import CANDIDATES, tensor_digest
from aip_vtok.c1_followup.quantization import ORIGINAL
from aip_vtok.diagnostics.diag0.io import command, json_at, metric_weight_hashes
from aip_vtok.runtime import ROOT, file_hash, resolve, write_json
from aip_vtok.sparse_decoder.protocol import MILESTONES, PROVENANCE
from aip_vtok.sparse_rfsq.calibration import load_scan
from aip_vtok.training.data import read_window
from aip_vtok.training.determinism import configure_determinism
from cp_adaptok.large_scale_data import load_manifests

from .bitstream import GOP
from .codec import JointCodec
from .decoder import migrate_sld
from .scan import BUDGETS, REF_BUDGET, sp1_order

BASE = "c5cd06dfc8cb6afd19c6da668d7664eb3f32c2ea"
BRANCH = "experiment/aip-vtok-joint-v1"
OUTPUT = resolve("results/aip_vtok/joint_v1")
CHECKPOINTS = resolve("checkpoints/aip_vtok/joint_v1")
CONFIG = "configs/aip_vtok/joint_v1.yaml"
SLD_500 = resolve("checkpoints/aip_vtok/sparse_decoder/E4/seed1/step_0500.pt")
PILOT_MANIFEST = resolve("results/aip_vtok/sparse_decoder/pilot_checkpoint_verification.json")
MODEL_VERSION = "AIP-VTok-Joint-V1"


def setup(seed: int, threads: int) -> None:
    configure_determinism()
    torch.set_num_threads(threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def source_hash() -> str:
    targets = list((ROOT / "aip_vtok/joint").glob("*.py"))
    targets.extend((ROOT / "scripts/aip_vtok").glob("*joint_v1.py"))
    content = "".join(f"{p.relative_to(ROOT)}\t{file_hash(p)}\n" for p in sorted(targets))
    return hashlib.sha256(content.encode()).hexdigest()


def verify_base() -> None:
    if command("git", "merge-base", "HEAD", BASE) != BASE:
        raise RuntimeError("Joint V1 requires the frozen sparse-decoder baseline")
    prefixes = ("aip_vtok/joint/", "results/aip_vtok/joint_v1/", "tests/aip_vtok/test_joint_")
    allowed = {CONFIG, "docs/aip_vtok_joint_v1.md", *(f"scripts/aip_vtok/{s}_joint_v1.py"
               for s in ("run", "evaluate", "benchmark"))}
    paths = command("git", "diff", "--name-only", "--ignore-submodules=none", BASE).splitlines()
    if foreign := [p for p in paths if p not in allowed and not p.startswith(prefixes)]:
        raise RuntimeError(f"frozen baseline files changed: {foreign}")


def verify_files(provenance: dict) -> dict:
    frozen = provenance["files"] | {
        str(SLD_500.relative_to(ROOT)): json_at(PILOT_MANIFEST)["models"]["E4"]["snapshots"]["500"]["sha256"]
    }
    for name, sha in frozen.items():
        if file_hash(name) != sha:
            raise RuntimeError(f"frozen SHA256 mismatch: {name}")
    if metric_weight_hashes() != provenance["metrics_sha256"]:
        raise RuntimeError("pretrained LPIPS or DISTS weights differ from sealed provenance")
    return frozen


def check_frame_isolation(splits: dict) -> None:
    owners: dict[str, str] = {}
    for split, rows in splits.items():
        for row in rows:
            for item in row["frame_hashes"]:
                if owners.setdefault(item["sha256"], split) != split:
                    raise ValueError("manifest frame SHA256 belongs to multiple splits")


class AuditedData:
    def __init__(self, directory: str, manifest_hashes: dict):
        splits, hashes = load_manifests(directory)
        if hashes != manifest_hashes:
            raise ValueError("dataset manifest SHA256 differs from the frozen protocol")
        check_frame_isolation(splits)
        self.hashes = hashes
        self._rows = {split: {row["content_id"]: row for row in rows if row["resolution"] == [1920, 1080]}
                      for split, rows in splits.items()}
        self.counts = dict.fromkeys(("train", "validation", "test"), 0)
        self.test_claim = None

    def catalog(self, split: str) -> dict:
        if split == "test" and self.test_claim is None:
            raise PermissionError("Test remains sealed until formal checkpoint selection")
        return self._rows[split]

    def read(self, split: str, video: str, start: int, length: int) -> torch.Tensor:
        row = self.catalog(split)[video]
        roles = {"train": {"train"}, "validation": {"validation", "historical-development"}, "test": {"test"}}
        if row["role"] not in roles[split]:
            raise ValueError("frame role conflicts with the audited split")
        frames = read_window(row, start, length)
        self.counts[split] += len(frames)
        return frames


@dataclass
class Experiment:
    config: dict
    legacy_config: dict
    legacy_codec: object
    legacy_identity: dict
    codec: JointCodec
    data: AuditedData
    identity: dict
    provenance: dict
    migration: dict

    def verify_integrity(self) -> None:
        verify_base()
        verify_files(self.provenance)
        if source_hash() != self.identity["source_sha256"]:
            raise RuntimeError("Joint source files changed during the run")
        frozen = {"backbone": self.codec.backbone, "projector": self.codec.projector}
        for name, module in frozen.items():
            if any(p.requires_grad or p.grad is not None for p in module.parameters()):
                raise RuntimeError(f"{name} received trainable parameters or gradients")
            if tensor_digest(module) != self.provenance["frozen_tensors_sha256"][name]:
                raise RuntimeError(f"{name} tensor SHA256 changed")


def initial_coordinates() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    layouts = json_at(CANDIDATES)["arms"]["A2"]
    fits = torch.load(FITTED, weights_only=True, map_location="cpu")["fits"]
    orders, levels, scales = [], [], []
    for kind in ("I", "P"):
        layout = layouts[kind]
        fit = fits[f"{kind}_aligned"]
        order = fit["energy"].argsort(descending=True, stable=True)
        stage, = layout["stages"]
        if order[:24].tolist() != layout["indices"] or layout["space"] != "aligned" or stage["mu"] != 0:
            raise ValueError("A2 does not match its sealed aligned rank and linear scale")
        orders.append(order)
        levels.append(stage["levels"] + [list(ORIGINAL)] * 4)
        scales.append(torch.cat((torch.tensor(stage["scales"]), fit["scales"][4:8])))
    return torch.stack(orders), torch.tensor(levels), torch.stack(scales)


def open_experiment(config_path: str | Path, seed: int) -> Experiment:
    verify_base()
    config = yaml.safe_load(resolve(config_path).read_text())
    if (config["version"], config["geometry"], config["gop"], tuple(config["budgets"]),
        config["reference_budgets"], config["precision"], config["batch"], config["optimizer"]) != (
        MODEL_VERSION, [1080, 1920], GOP, BUDGETS, {"I": REF_BUDGET[True], "P": REF_BUDGET[False]},
        "bf16", 1, "AdamW",
    ):
        raise ValueError("configuration violates the Joint V1 codec protocol")
    setup(seed, config["cpu_threads"])
    legacy_config, legacy_identity = frozen_inputs()
    provenance = json_at(PROVENANCE)
    if legacy_identity != provenance["codec"]:
        raise RuntimeError("legacy codec differs from the sealed sparse-decoder assets")
    frozen = verify_files(provenance)
    data = AuditedData(legacy_config["manifest_dir"], legacy_identity["manifests"])
    scan = load_scan()
    if scan["winner"] != "SP-1" or scan["sigma"] != 1.5:
        raise RuntimeError("requires Train-frozen SP-1 Gaussian configuration")
    legacy = load_frozen(legacy_config)
    orders, levels, scales = initial_coordinates()
    codec = JointCodec(legacy.backbone, legacy.projector, orders, levels, scales,
                       sp1_order(scan["keys"]["SP-1"]), config["scan_temperature"])
    artifact = torch.load(SLD_500, weights_only=True, map_location="cpu")
    if artifact["step"] != MILESTONES[-1] or artifact["identity"]["arm"] != "E4":
        raise RuntimeError("legacy initialization requires the sealed E4 step 500 checkpoint")
    migration = migrate_sld(codec.sld, artifact["model"], orders)
    migration.update(source=str(SLD_500.relative_to(ROOT)), step=artifact["step"], sha256=file_hash(SLD_500))
    codec.to(device=config["device"], memory_format=torch.channels_last)
    setup(seed, config["cpu_threads"])
    identity = {"version": MODEL_VERSION, "base_commit": BASE, "seed": seed,
                "source_sha256": source_hash(), "config_sha256": file_hash(config_path),
                "dataset_manifest_sha256": data.hashes, "frozen_file_sha256": frozen,
                "metric_weights_sha256": provenance["metrics_sha256"], "torch": str(torch.__version__),
                "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(codec.device)}
    run = Experiment(config, legacy_config, legacy, legacy_identity, codec, data, identity, provenance, migration)
    run.verify_integrity()
    write_json(OUTPUT / "migration_report.json", migration)
    return run


@contextmanager
def exclusive_run():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
