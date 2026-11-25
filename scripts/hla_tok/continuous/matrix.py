"""Read a frozen source frame once and score independent native reconstructions."""

from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from hla_tok.evaluation.benchmark import transfer
from hla_tok.evaluation.io import write_csv, write_json
from hla_tok.evaluation.metrics import sha256_file
from hla_tok.evaluation.reconstruction import save_native
from hla_tok.streaming.continuous import ContinuousReceiver, RankOrder, encode_continuous, truncate
from scripts.hla_tok.continuous.inputs import Inputs
from scripts.hla_tok.continuous.quality import NativeQuality, summarize
from scripts.hla_tok.v2.protocol import ROOT, csv_at, json_at


@dataclass(frozen=True)
class Setting:
    order: RankOrder
    budget: int
    wire_dtype: str = "none"

    def fields(self) -> dict:
        return {"order": self.order.name.name, "budget": self.budget, "wire_dtype": self.wire_dtype}


Decode = Callable[[Setting, Tensor, int], tuple[Tensor, dict]]


def validation_frames(inputs: Inputs, selections: Iterable[tuple[str, int]]) -> Iterator[tuple[str, int, Tensor]]:
    for video, index in selections:
        yield video, index, transfer(inputs.reader.read("validation", video, index)[None])


def representative(cohort: dict[str, int]) -> list[tuple[str, int]]:
    return [(video, index) for video, count in cohort.items() for index in (0, count // 2, count - 1)]


def compute(precision: str):
    if precision not in {"fp32", "bf16"}:
        raise ValueError("expected fp32 or bf16 computation")
    return torch.autocast("cuda", enabled=precision == "bf16", dtype=torch.bfloat16)


def _decode_raw(inputs: Inputs) -> Decode:
    receiver = ContinuousReceiver.from_model(inputs.model)

    def decode(setting: Setting, values: Tensor, frame_id: int) -> tuple[Tensor, dict]:
        return receiver.decode(values, setting.order), {}

    return decode


def _sample_label(config: dict, setting: Setting, video: str, index: int) -> str | None:
    sample = config["sample"]
    if (video, index) != (sample["video"], sample["frame"]) or setting.budget not in sample["budgets"]:
        return None
    return f"{setting.order.name.name}_{setting.wire_dtype}_K{setting.budget}"


def _score_frame(inputs: Inputs, quality: NativeQuality, settings: list[Setting], decode: Decode,
                 video: str, index: int, source: Tensor, samples: Path | None) -> list[dict]:
    precision = inputs.config["quality_compute_dtype"]
    with compute(precision):
        full = encode_continuous(inputs.model, source)
        ordered = {order: order.apply(full) for order in dict.fromkeys(case.order for case in settings)}
    quality.prepare(source)
    rows = []
    for case in settings:
        with compute(precision):
            image, details = decode(case, truncate(ordered[case.order], case.budget), index)
        rows.append(case.fields() | {"compute_dtype": precision, "history": "stateless", "video": video,
                                     "frame": index} | quality.score(image) | details)
        label = _sample_label(inputs.config, case, video, index) if samples is not None else None
        if samples is not None and label is not None:
            save_native(samples, label, image)
            if not (samples / "source.png").exists():
                save_native(samples, "source", source)
    return rows


def _bind_matrix(inputs: Inputs, settings: list[Setting], name: str) -> dict:
    sources = ["models/local.py", "models/codec.py", "models/spatial.py", "progressive/geometry.py",
               "streaming/continuous.py"]
    if any(case.wire_dtype != "none" for case in settings):
        sources.append("streaming/continuous_wire.py")
    files = [ROOT / "hla_tok" / path for path in sources]
    files.extend(Path(__file__).with_name(name) for name in ("matrix.py", "quality.py"))
    files.append(ROOT / "cp_adaptok/metrics.py")
    if any(case.wire_dtype != "none" for case in settings):
        files.extend(Path(__file__).with_name(name) for name in ("receiver.py", "wire_eval.py"))
    binding = inputs.provenance() | {
        "name": name, "implementation": {str(path.relative_to(ROOT)): sha256_file(path) for path in files},
        "torch": str(torch.__version__), "gpu": torch.cuda.get_device_name(), "history": "stateless",
        "compute_dtype": inputs.config["quality_compute_dtype"],
        "settings": [case.fields() | {"native_ranks": list(case.order.ranks)} for case in settings],
    }
    return binding


def _restore_rows(path: Path) -> list[dict]:
    integers = {"budget", "frame", "payload_bytes", "header_bytes", "side_information_bytes", "total_bytes"}
    floats = {"mse", "psnr", "lpips", "dists"}
    cast = {field: int for field in integers} | dict.fromkeys(floats, float)
    return [{key: cast.get(key, str)(value) for key, value in row.items()} for row in csv_at(path)]


def _verify_matrix_rows(rows: list[dict], video: str, count: int, settings: list[Setting]) -> None:
    expected = {(case.order.name.name, case.wire_dtype, case.budget, index)
                for case in settings for index in range(count)}
    observed = {(row["order"], row["wire_dtype"], row["budget"], row["frame"]) for row in rows}
    if observed != expected or len(rows) != len(expected) or {row["video"] for row in rows} != {video}:
        raise ValueError(f"invalid native matrix coverage for {video}")


@torch.inference_mode()
def evaluate_matrix(inputs: Inputs, quality: NativeQuality, settings: list[Setting], output: Path, name: str,
                    decode: Decode | None = None, *, samples: Path | None = None) -> list[dict]:
    """Resume only complete per-video tables with the same inputs, settings and revision."""
    folder = output / "matrix" / name
    binding = _bind_matrix(inputs, settings, name)
    manifest = folder / "identity.json"
    if manifest.exists() and json_at(manifest) != binding:
        raise ValueError(f"matrix identity differs in {folder}")
    write_json(manifest, binding)
    decoder = _decode_raw(inputs) if decode is None else decode
    all_rows = []
    for position, (video, count) in enumerate(inputs.cohort.items()):
        path = folder / f"video_{position}.csv"
        if path.exists():
            rows = _restore_rows(path)
        else:
            rows = []
            frames = validation_frames(inputs, ((video, i) for i in range(count)))
            for _, index, source in frames:
                rows.extend(_score_frame(inputs, quality, settings, decoder, video, index, source, samples))
                if (index + 1) % 16 == 0:
                    print(f"{name} {video} {index + 1}/{count}", flush=True)
            write_csv(path, rows)
        _verify_matrix_rows(rows, video, count, settings)
        all_rows.extend(rows)
        print(f"{name} {video} {count}/{count}", flush=True)
    return all_rows


def groups(rows: list[dict], fields: tuple[str, ...]) -> Iterator[tuple[dict, list[dict]]]:
    parts = defaultdict(list)
    for row in rows:
        parts[tuple(row[field] for field in fields)].append(row)
    for key, data in parts.items():
        yield dict(zip(fields, key, strict=True)), data


def summarize_matrix(rows: list[dict], *, by_video: bool = False) -> list[dict]:
    keys = ("order", "budget", "wire_dtype", "compute_dtype", "history")
    return [fields | summarize(data) for fields, data in groups(rows, (*keys, "video") if by_video else keys)]
