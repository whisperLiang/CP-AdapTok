"""Evaluate each Compact setting through a real packet and independent synthesis."""

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from hla_tok.compression.profiles import PROFILES_BY_LABEL
from hla_tok.evaluation.io import write_csv, write_json
from hla_tok.evaluation.metrics import sha256_file
from hla_tok.evaluation.reconstruction import save_native
from hla_tok.streaming.compact_wire import deserialize
from hla_tok.streaming.continuous import ContinuousReceiver, encode_continuous
from hla_tok.streaming.continuous_wire import wire_rate
from scripts.hla_tok.compact.assets import load_asset
from scripts.hla_tok.compact.inputs import CONFIG, compact_orders
from scripts.hla_tok.continuous.inputs import Inputs
from scripts.hla_tok.continuous.matrix import compute, groups, validation_frames
from scripts.hla_tok.continuous.quality import NativeQuality, summarize
from scripts.hla_tok.v2.protocol import ROOT, csv_at, json_at

KEYS = ("asset", "profile", "projection", "order", "budget", "compute_dtype")
LEDGER = ("payload_bytes", "header_bytes", "side_information_bytes", "padding_bytes", "total_bytes")


@dataclass(frozen=True, order=True)
class Case:
    asset: str
    order: str
    budget: int


def case_groups(cases: Iterable[Case]) -> dict[tuple[str, str], tuple[int, ...]]:
    groups: dict[tuple[str, str], set[int]] = defaultdict(set)
    for case in cases:
        groups[(case.asset, case.order)].add(case.budget)
    return {key: tuple(sorted(budgets)) for key, budgets in groups.items()}


def _read_rows(path: Path) -> list[dict]:
    cast = dict.fromkeys(("frame", "budget", *LEDGER), int) | dict.fromkeys(("mse", "psnr", "lpips", "dists"), float)
    return [{key: cast.get(key, str)(value) for key, value in row.items()} for row in csv_at(path)]


def _matrix_binding(inputs: Inputs, config: dict, cases: list[Case], assets: dict) -> dict:
    production = [*sorted((ROOT / "hla_tok/compression").glob("*.py")),
                  *(ROOT / "hla_tok/streaming" / name for name in ("compact.py", "compact_wire.py"))]
    sources = [*production, Path(__file__), ROOT / "scripts/hla_tok/continuous/quality.py", ROOT / "cp_adaptok/metrics.py"]
    return inputs.provenance() | {
        "protocol": sha256_file(CONFIG), "compute_dtype": config["compute_dtype"], "torch": str(torch.__version__),
        "gpu": torch.cuda.get_device_name(), "cases": [vars(case) for case in cases],
        "implementation": {str(p.relative_to(ROOT)): sha256_file(p) for p in sources},
        "assets": {key: assets[key]["sha256"] for key in sorted({c.asset for c in cases})},
    }


class Matrix:
    def __init__(self, inputs: Inputs, config: dict, assets: dict, cases: list[Case], output: Path):
        self.inputs = inputs
        self.config = config
        self.cases = cases
        self.output = output
        self.quality = NativeQuality()
        self.receiver = ContinuousReceiver.from_model(inputs.model)
        self.orders = compact_orders()
        self.groups = case_groups(cases)
        self.profiles = {key: assets[key] for key, _ in self.groups}
        self.codecs = {key: self._codec(key, assets) for key in self.groups}

    def _codec(self, key: tuple[str, str], assets: dict):
        asset, order = key
        spec = assets[asset]
        codec, _ = load_asset(self.output / "profiles" / spec["filename"], spec, PROFILES_BY_LABEL[spec["profile"]],
                              self.orders[order], self.inputs.checkpoint["state_dict_sha256"])
        return codec

    def score(self, video: str, index: int, source: Tensor) -> list[dict]:
        precision = self.config["compute_dtype"]
        with compute(precision):
            full = encode_continuous(self.inputs.model, source)
            ordered = {order: self.orders[order].apply(full) for order in {key[1] for key in self.groups}}
        self.quality.prepare(source)
        rows = []
        for key, budgets in self.groups.items():
            rows.extend(self._score_cases(key, budgets, ordered[key[1]], video, index))
        return rows

    def _score_cases(self, key: tuple[str, str], budgets: tuple[int, ...], ordered: Tensor,
                     video: str, index: int) -> list[dict]:
        asset, order = key
        codec = self.codecs[key]
        precision = self.config["compute_dtype"]
        tags = {"asset": asset, "profile": codec.profile.label, "projection": self.profiles[asset]["projection"],
                "order": order, "compute_dtype": precision, "video": video, "frame": index}
        with compute(precision):
            u = codec.transform.encode(ordered, codec.ranks)
            encoded = codec.frame(codec.encode_codes(u[:, :budgets[-1]]), index)
        rows = []
        for budget in budgets:
            raw = encoded.serialize_range(0, budget)
            packet = deserialize(raw)
            if packet.identity != codec.identity or packet.count != budget or packet.offset != 0:
                raise RuntimeError("Compact packet does not satisfy the requested prefix")
            with compute(precision):
                image = codec.decode_payload(self.receiver, packet.payload)
            rows.append(tags | {"budget": budget} | self.quality.score(image) | packet.ledger())
        return rows


def _check_rows(rows: list[dict], cases: list[Case], video: str, count: int) -> None:
    expected = {(c.asset, c.order, c.budget, frame) for c in cases for frame in range(count)}
    observed = {(r["asset"], r["order"], r["budget"], r["frame"]) for r in rows}
    if observed != expected or len(rows) != len(expected) or {r["video"] for r in rows} != {video}:
        raise ValueError(f"incomplete or duplicate matrix rows for {video}")


@torch.inference_mode()
def evaluate(inputs: Inputs, config: dict, assets: dict, output: Path, name: str,
             cases: Iterable[Case]) -> list[dict]:
    cases = sorted(set(cases))
    folder = output / "matrix" / name
    identity = folder / "identity.json"
    binding = _matrix_binding(inputs, config, cases, assets)
    if identity.exists() and json_at(identity) != binding:
        raise ValueError(f"matrix {name} has a different evaluation identity")
    write_json(identity, binding)
    all_rows = []
    scorer = None
    for position, (video, count) in enumerate(inputs.cohort.items()):
        path = folder / f"video_{position}.csv"
        if path.exists():
            rows = _read_rows(path)
        else:
            if scorer is None:
                scorer = Matrix(inputs, config, assets, cases, output)
            rows = []
            for _, index, source in validation_frames(inputs, ((video, i) for i in range(count))):
                rows.extend(scorer.score(video, index, source))
                if (index + 1) % 10 == 0:
                    print(f"{name}: {video} {index + 1}/{count}", flush=True)
            _check_rows(rows, cases, video, count)
            write_csv(path, rows)
        _check_rows(rows, cases, video, count)
        all_rows.extend(rows)
        print(f"{name}: {video} {count}/{count}", flush=True)
    return all_rows


def summarize_matrix(rows: list[dict], *, by_video: bool = False) -> list[dict]:
    keys = (*KEYS, "video") if by_video else KEYS
    summary = []
    for fields, items in groups(rows, keys):
        sums = {name: sum(row[name] for row in items) for name in LEDGER}
        rates = wire_rate(sums["payload_bytes"], sums["total_bytes"], len(items))
        summary.append(fields | summarize(items) | rates | sums | {"actual_tokens_per_frame": fields["budget"]})
    return summary


@torch.inference_mode()
def sample(inputs: Inputs, config: dict, output: Path, assets: dict, asset: str, order: str,
           budget: int, label: str) -> None:
    spec = assets[asset]
    case = config["sample"]
    _, index, source = next(validation_frames(inputs, [(case["video"], case["frame"])]))
    decoder = ContinuousReceiver.from_model(inputs.model)
    codec, _ = load_asset(output / "profiles" / spec["filename"], spec, PROFILES_BY_LABEL[spec["profile"]],
                          compact_orders()[order], inputs.checkpoint["state_dict_sha256"])
    with compute(config["compute_dtype"]):
        frame = codec.encode(inputs.model, source, budget, index)
        packet = frame.serialize_range(0, budget)
        save_native(output / "samples", label, codec.decode(decoder, packet))
        save_native(output / "samples", "source", source)
