"""Frozen C0 inputs and audited inference-only model loading."""

import argparse
import fcntl
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import torch

from hla_tok import HLATok
from hla_tok.evaluation.data import AuditedFrames
from hla_tok.evaluation.io import environment, write_csv, write_json
from hla_tok.evaluation.metrics import metric_hashes, sha256_file
from hla_tok.training.v2 import state_digest
from scripts.hla_tok.v2.protocol import COHORT, ROOT, frozen_data, json_at, setup
from scripts.hla_tok.v3.downstream import model_from_checkpoint
from scripts.hla_tok.v3.protocol import PROTOCOL as V3_PROTOCOL
from scripts.hla_tok.v3.protocol import read_config

CONFIG = ROOT / "configs/hla_tok/continuous/protocol.yaml"
CONFIG_SHA256 = "72517a17fa778ac4f558c99bada80e4c9942fe7684e933266443d76d4a4be3cd"
RESULTS = ROOT / "results/hla_tok/continuous"
V3 = ROOT.parent.parent / "hla-tok-v3" / ROOT.name
METRICS = ("global_psnr", "lpips", "dists")


def run_arguments(description: str | None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--v3", type=Path, default=V3)
    parser.add_argument("--output", type=Path, default=RESULTS)
    return parser


@contextmanager
def locked_output(output: Path) -> Iterator[None]:
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".continuous.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def require_digest(path: Path, expected: str) -> str:
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"SHA-256 mismatch for {path}")
    return observed


def model_digest(model: torch.nn.Module) -> str:
    state = model.state_dict()
    return state_digest(state, state.keys())


def verify_model_state(model: torch.nn.Module, expected: str) -> str:
    observed = model_digest(model)
    if observed != expected:
        raise ValueError("state_dict differs from the published checkpoint")
    return observed


def load_model(v3: Path, arm: str, config: dict, device: str = "cuda") -> tuple[HLATok, dict]:
    stage = f"v3/quantized/seed1/{arm}"
    relative = f"checkpoints/hla_tok/{stage}/step_10000.pt"
    record, = (row for row in json_at(ROOT / "results/hla_tok/v3/checkpoint_manifest.json")
               if row["path"] == relative)
    path = v3 / relative
    digest = require_digest(path, config["checkpoint_sha256"][arm])
    model, state = model_from_checkpoint(path)
    if record["sha256"] != digest or state["identity"] != record["identity"] or state["step"] != 10_000:
        raise ValueError("checkpoint identity differs from the published Seed 1 endpoint")
    weights = state["model"]
    state_sha = state_digest(weights, weights.keys())
    if not isinstance(model, HLATok):
        raise ValueError("expected a continuous HLA-Tok checkpoint")
    model.eval().requires_grad_(False).to(device=device, memory_format=torch.channels_last)
    verify_model_state(model, state_sha)
    return model, record | {"verified_path": str(path), "state_dict_sha256": state_sha,
                            "budget_updates": state["counters"]["budget_updates"]}


@dataclass(frozen=True)
class Inputs:
    config: dict
    reader: AuditedFrames
    cohort: dict[str, int]
    metric_weights: dict[str, str]
    model: HLATok
    checkpoint: dict

    def provenance(self) -> dict:
        return {"protocol_sha256": CONFIG_SHA256, "cohort_sha256": self.config["cohort_sha256"],
                "data_manifest_sha256": self.reader.hashes, "metric_weight_sha256": self.metric_weights,
                "checkpoint_sha256": self.checkpoint["sha256"],
                "model_state_sha256": self.checkpoint["state_dict_sha256"],
                "cohort": self.cohort, "training_steps": 0}

    def finish(self, path: Path) -> None:
        digest = verify_model_state(self.model, self.checkpoint["state_dict_sha256"])
        require_digest(Path(self.checkpoint["verified_path"]), self.checkpoint["sha256"])
        if self.reader.counts["test"]:
            raise RuntimeError("Test pixel isolation failed")
        write_json(path, self.provenance() | {"state_dict_after_sha256": digest,
                                             "pixel_reads": dict(self.reader.counts),
                                             "test_pixel_reads": self.reader.counts["test"]})


def open_inputs(v3: Path) -> Inputs:
    require_digest(CONFIG, CONFIG_SHA256)
    config = read_config(CONFIG)
    require_digest(COHORT, config["cohort_sha256"])
    require_digest(V3_PROTOCOL, config["v3_protocol_sha256"])
    setup()
    if torch.cuda.device_count() != 1 or "H100" not in torch.cuda.get_device_name() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("HLA-Tok-C requires one CUDA BF16-capable NVIDIA H100")
    reader, cohort, identity = frozen_data()
    if metric_hashes() != identity["metric_weights"]:
        raise ValueError("metric weights differ from V3")
    model, checkpoint = load_model(v3, "C0", config)
    return Inputs(config, reader, cohort, identity["metric_weights"], model, checkpoint)


def replay_result(inputs: Inputs, v3: Path, output: Path) -> dict:
    reference = v3 / "results/hla_tok/v3/quantized/seed1/C0/full/step_10000/metrics.json"
    expected, = json_at(reference)
    replay = output / "native_replay"
    observed, = json_at(replay / "metrics.json")
    reads = json_at(replay / "pixel_reads.json")
    limits = inputs.config["reproduction"]["tolerances"]
    comparisons = [
        {"metric": key, "v3_reference": expected[key], "measured": observed[key],
         "absolute_delta": abs(expected[key] - observed[key]), "tolerance": limits[key]}
        for key in METRICS
    ]
    ok = (reads == {"train": 0, "validation": 174, "test": 0}
          and observed["frames"] == expected["frames"] == 174
          and all(row["absolute_delta"] <= row["tolerance"] for row in comparisons))
    return inputs.provenance() | {
        "status": "PASS" if ok else "FAIL", "frames": observed["frames"],
        "native_resolution_hw": [1080, 1920], "measurement": "V3 native evaluator",
        "reference_metrics_sha256": sha256_file(reference),
        "compute_dtype": "bf16", "metric_dtype": "fp32", "history": "stateless",
        "state_initialization": None, "state_updates": "none",
        "pixel_reads": reads, "test_pixel_reads": reads["test"], "metrics": comparisons,
    }


def record_replay(v3: Path, output: Path) -> None:
    inputs = open_inputs(v3)
    write_json(output / "checkpoint_manifest.json", inputs.checkpoint)
    write_json(output / "environment_manifest.json", environment() | {
        "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "python_dependencies": {name: version(name) for name in (
            "torch", "torchvision", "numpy", "Pillow", "PyYAML", "lpips", "DISTS-pytorch", "pytorch-msssim")},
        "metric_weights": inputs.metric_weights,
        "metric_source_sha256": {
            path: sha256_file(ROOT / path) for path in (
                "scripts/hla_tok/v3/quality.py", "hla_tok/evaluation/metrics.py", "cp_adaptok/metrics.py")
        },
    })
    result = replay_result(inputs, v3, output)
    write_json(output / "reproduction.json", result)
    write_csv(output / "reproduction.csv", result["metrics"])
    if result["status"] != "PASS":
        raise RuntimeError("V3 native replay failed its frozen reproduction tolerances")
    print(f"Phase 0 {result['status']} with {result['frames']} native Validation frames", flush=True)
