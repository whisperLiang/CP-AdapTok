"""Read and verify the frozen experiment without writing to its directories."""

import fcntl
import json
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass

import torch

from aip_vtok.c1.artifacts import FITTED, LOCKED, frozen_inputs, load_frozen, timestamp
from aip_vtok.c1.metrics import metric_suite
from aip_vtok.c1_followup.io import CANDIDATES, ReadAudit, create_record, tensor_digest
from aip_vtok.diagnostics.diag0.io import FIX_WEIGHTS, PROJECTOR, command, json_at, metric_weight_hashes
from aip_vtok.runtime import ROOT, file_hash, resolve, write_json
from aip_vtok.sparse_rfsq.calibration import Inputs, load_inputs, load_scan
from aip_vtok.sparse_rfsq.protocol import CALIBRATION, PROTOCOL, S1_GATE, S2_GATE, SCAN_LOCK, SEED, UPDATES
from aip_vtok.sparse_rfsq.protocol import OUTPUT as OLD
from aip_vtok.sparse_rfsq.protocol import WEIGHTS as OLD_WEIGHTS
from aip_vtok.sparse_rfsq.receiver import RecoveryNet
from aip_vtok.training.determinism import configure_determinism

BASE = "606216e768c277aa59b4dde6d3bd84247a02f905"
BRANCH = "experiment/aip-vtok-sparse-decoder"
OUTPUT = resolve("results/aip_vtok/sparse_decoder")
WEIGHTS = resolve("checkpoints/aip_vtok/sparse_decoder")
PROVENANCE = OUTPUT / "provenance.json"
HISTORICAL_DENSE = resolve("results/aip_vtok/c1_followup/validation.json")
W1 = OLD_WEIGHTS / "W1/seed1.pt"
TRAIN_PLAN = OLD / "training_plan.json"
SIGMA = 1.5
MILESTONES = (0, 100, 300, 500)


def verify_tree():
    if command("git", "branch", "--show-current") != BRANCH or command("git", "merge-base", "HEAD", BASE) != BASE:
        raise RuntimeError(f"requires {BRANCH} descended from {BASE}")
    prefixes = ("aip_vtok/sparse_decoder/", "results/aip_vtok/sparse_decoder/", "checkpoints/aip_vtok/sparse_decoder/")
    files = {"scripts/aip_vtok/run_sparse_decoder.py", "tests/aip_vtok/test_sparse_decoder.py",
             "docs/aip_vtok_sparse_decoder_report.md"}
    changed = command("git", "diff", "--name-only", "--ignore-submodules=none", BASE).splitlines()
    foreign = [p for p in changed if not p.startswith(prefixes) and p not in files]
    if foreign:
        raise RuntimeError(f"frozen base files differ from {BASE}: {foreign}")


def input_hashes(config):
    paths = [
        LOCKED, FITTED, CANDIDATES, CANDIDATES.with_name("calibration.json"), CANDIDATES.with_name("protocol.json"),
        PROTOCOL, SCAN_LOCK, CALIBRATION.with_suffix(".json"), S1_GATE, S2_GATE, OLD / "W1_training.json",
        OLD / "s1.json", OLD / "s2.json", TRAIN_PLAN, W1, HISTORICAL_DENSE,
        resolve(config["intra_checkpoint"]), resolve(config["ld_checkpoint"]),
        resolve(config["checkpoints"]) / "alignment.pt", PROJECTOR, FIX_WEIGHTS,
    ]
    paths.extend(sorted(resolve(config["manifest_dir"]).glob("cpat_*.json")))
    return {str(path.relative_to(ROOT)): file_hash(path) for path in paths}


def load_w1(lock, config):
    from aip_vtok.c1.training import LR

    record = json_at(OLD / "W1_training.json")
    gate = json_at(S2_GATE)
    digest = file_hash(W1)
    if record["checkpoint_sha256"] != digest or gate["weights"]["W1"] != digest:
        raise RuntimeError("W1 checkpoint SHA256 differs from its sealed training record")
    state = torch.load(W1, weights_only=True, map_location="cpu")
    identity = record["identity"]
    frozen = json_at(PROTOCOL)["pilot"]
    loss = config["stage1"]["loss"] | {
        "latent_anchor": frozen["loss"]["latent_anchor"], "reference": frozen["loss"]["reference"],
    }
    expected = {"variant": "W1", "seed": SEED, "updates": UPDATES, "rollout_frames": 4,
                "scan_lock_sha256": file_hash(SCAN_LOCK), "lr": LR, "loss": loss}
    if (record["steps"] != UPDATES or state["step"] != UPDATES or state["identity"] != identity
            or json.loads(json.dumps(state["history"])) != record["updates"]
            or state["plan"] != json_at(TRAIN_PLAN)["windows"]
            or any(identity[key] != value for key, value in expected.items())
            or loss != frozen["loss"] or lock["winner"] != gate["winner"]):
        raise RuntimeError("W1 disagrees with the frozen codec, loss, scan, or training record")
    model = RecoveryNet().cuda().to(memory_format=torch.channels_last)
    model.load_state_dict(state["model"])
    return model.eval().requires_grad_(False), identity, state["optimizer"]["param_groups"][0]


@dataclass
class Run:
    config: dict
    codec: object
    suite: object
    reader: ReadAudit
    inputs: Inputs
    lock: dict
    w1: RecoveryNet
    provenance: dict

    def verify_integrity(self):
        verify_tree()
        modules = {"backbone": self.codec.backbone, "projector": self.codec.projector, "W1": self.w1}
        tensors = {key: tensor_digest(model) for key, model in modules.items()}
        hashes_match = (input_hashes(self.config) == self.provenance["files"]
                        and metric_weight_hashes() == self.provenance["metrics_sha256"]
                        and tensors == self.provenance["frozen_tensors_sha256"])
        frozen_grads = [f"{name}.{key}" for name, model in modules.items() for key, p in model.named_parameters()
                        if p.requires_grad or p.grad is not None]
        if not hashes_match or frozen_grads or self.reader.counts["test"] or self.reader.denied:
            raise RuntimeError(f"frozen input or split integrity failed, frozen gradients={frozen_grads}")
        return {"passed": True, "frozen_tensors_sha256": tensors,
                "images_read": dict(self.reader.counts), "denied_reads": self.reader.denied}


def open_run(*, metrics=True):
    verify_tree()
    configure_determinism()
    config, identity = frozen_inputs()
    files = input_hashes(config)
    metric_hashes = metric_weight_hashes()
    prior = json_at(PROTOCOL)
    if identity != prior["identity"] or metric_hashes != prior["metric_weights"]:
        raise RuntimeError("codec or LPIPS/DISTS asset differs from the sealed S2 provenance")
    if (prior["c1_scan_sha256"] != file_hash(LOCKED)
            or prior["c1_followup_candidates_sha256"] != file_hash(CANDIDATES)):
        raise RuntimeError("C1 scan or Follow-up candidates have a different SHA256")
    lock = load_scan()
    s1 = json_at(S1_GATE)
    if (not s1["released"] or s1["scan_lock_sha256"] != file_hash(SCAN_LOCK)
            or s1["s1_sha256"] != file_hash(OLD / "s1.json")
            or lock["winner"] != "SP-1" or lock["sigma"] != SIGMA):
        raise RuntimeError("requires the released Train-only SP-1 scan and sigma=1.5")
    inputs = load_inputs(config, identity)
    if inputs.legacy_sigma != SIGMA or config["geometry"] != [1080, 1920] or config["gop"] != 32:
        raise RuntimeError("legacy or native geometry protocol differs")
    w1, w1_identity, optimizer = load_w1(lock, config)
    codec = load_frozen(config)
    tensors = {"backbone": tensor_digest(codec.backbone), "projector": tensor_digest(codec.projector)}
    if tensors != json_at(S2_GATE)["integrity"]["tensors_before"]:
        raise RuntimeError("loaded DCVC or Projector tensor hash differs from S2")
    provenance = {"base_commit": BASE, "branch": BRANCH, "codec": identity, "files": files,
                  "metrics_sha256": metric_hashes, "frozen_tensors_sha256": tensors | {"W1": tensor_digest(w1)},
                  "scan": "SP-1", "sigma": SIGMA, "w1": w1_identity,
                  "w1_optimizer": {k: v for k, v in optimizer.items() if k != "params"},
                  "reference": prior["reference_budgets"], "validation": prior["validation"]}
    # Normalize optimizer beta tuples to their JSON representation for replay.
    provenance["w1_optimizer"]["betas"] = list(provenance["w1_optimizer"]["betas"])
    create_record(PROVENANCE, provenance)
    return Run(config, codec, metric_suite() if metrics else None,
               ReadAudit(config, manifest_hashes=identity["manifests"]), inputs, lock, w1, provenance)


def gpu_snapshot():
    processes = command("nvidia-smi", "-i", "0", "--query-compute-apps=pid", "--format=csv,noheader")
    other = [p for p in processes.splitlines() if p.strip() != str(os.getpid())]
    if other or not torch.cuda.is_available():
        raise RuntimeError(f"requires available CUDA GPU 0 without competing compute processes: {other}")
    return command("nvidia-smi", "-i", "0", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader")


@contextmanager
def stage_audit(stage, *, metrics=True):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / ".run.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        directory = OUTPUT / "audits"
        path = directory / f"{stage}_{len(list(directory.glob(f'{stage}_*.json'))) + 1:02}.json"
        entry = {"command": sys.orig_argv, "started": timestamp(), "torch": str(torch.__version__),
                 "cuda": torch.version.cuda}
        run = None
        try:
            entry["gpu"] = gpu_snapshot()
            run = open_run(metrics=metrics)
            yield run
            entry["integrity"] = run.verify_integrity()
            entry["exit_code"] = 0
        except BaseException as exc:
            entry.update(exit_code=1, error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            if run is not None:
                entry["images_read"] = dict(run.reader.counts)
                entry["denied_reads"] = run.reader.denied
            entry["finished"] = timestamp()
            write_json(path, entry)
