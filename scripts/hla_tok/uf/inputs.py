"""Use the exact archived 174-frame and Train-only holdout plans."""

from pathlib import Path

from hla_tok.evaluation.data import AuditedFrames
from scripts.hla_tok.temporal.inputs import Clip
from scripts.hla_tok.v2.protocol import ROOT, frozen_data, json_at

OUTPUT = ROOT / "results/hla_tok/uf"
ARCHIVE = ROOT / "results/hla_tok/temporal"
BASELINE = "502eaf64f6fe8693dc90ddf6164c87799f1aebfe"


def _check(reader: AuditedFrames, rows: list[dict]) -> list[Clip]:
    clips = [Clip(row["split"], row["video"], row["start"], row["length"]) for row in rows]
    if any(clip.manifest(reader) != row for clip, row in zip(clips, rows, strict=True)):
        raise ValueError("archived clip identity differs from the frozen manifest")
    return clips


def inputs() -> tuple[AuditedFrames, dict[str, list[Clip]]]:
    reader, cohort, _ = frozen_data()
    archive = json_at(ARCHIVE / "sequence_manifest.json")
    if archive["dataset_sha256"] != reader.hashes:
        raise ValueError("frozen temporal manifest hashes changed")
    primary = _check(reader, archive["validation"])
    if {clip.video: clip.length for clip in primary} != cohort:
        raise ValueError("Validation cohort no longer matches the 174-frame archive")
    fit, holdout = (_check(reader, [row for row in archive["training"][name] if row["length"] == 32])
                    for name in ("fit", "holdout"))
    if any(clip.split != "train" for clip in (*fit, *holdout)):
        raise PermissionError("fitting and calibration must stay inside Train")
    if {clip.video for clip in fit} & {clip.video for clip in holdout}:
        raise ValueError("Calibration overlaps fitting sources")
    return reader, {"validation": primary, "long": _check(reader, archive["long_validation"]),
                    "fit": fit, "calibration": holdout}


def checkpoint_directory(argument: Path | None) -> Path:
    return argument if argument is not None else ROOT / "checkpoints/hla_tok/uf/official"
