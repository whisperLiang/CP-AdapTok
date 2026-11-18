"""Post-hoc transmitted-code diagnostics; these statistics do not select gates.

Empirical ID entropy describes code frequencies, not semantic information bits.
Use complete native evaluations only, on each model's real K2048 history.
"""

import math
from collections import Counter

from adaptok_hr.codec.bitstream import parse
from adaptok_hr.runtime import file_hash, write_json

from .run import ROOT

LEVELS = {"base": (0, 135), "medium": (135, 645), "fine": (645, 2048)}


def distribution(counts):
    total = counts.total()
    if not total:
        raise ValueError("missing transmitted code IDs")
    probabilities = [value / total for value in counts.values()]
    return {"transmitted_ids": total, "unique_ids": len(counts),
            "maximum_fraction": max(probabilities),
            "empirical_id_entropy_bits": -sum(p * math.log2(p) for p in probabilities)}


def code_usage(directory):
    paths = sorted((directory / "same_context/K2048").rglob("*.bin"))
    counts = {level: Counter() for level in LEVELS}
    within = {level: [] for level in LEVELS}
    for path in paths:
        ids = parse(path.read_bytes()).ids
        if len(ids) != 2048:
            raise ValueError("diagnostics require actual K2048 packets")
        for level, (start, stop) in LEVELS.items():
            frame = Counter(ids[start:stop])
            counts[level].update(frame)
            within[level].append(distribution(frame))
    result = {}
    for level in LEVELS:
        frames = within[level]
        result[level] = {**distribution(counts[level]), "frames": len(frames),
                         "mean_frame_unique_ids": sum(r["unique_ids"] for r in frames) / len(frames),
                         "mean_frame_maximum_fraction": sum(r["maximum_fraction"] for r in frames) / len(frames)}
    return result


def main():
    results = {}
    for path in sorted(ROOT.glob("S*/validation_*_64/summary.json")):
        key = f"{path.parents[1].name}/{path.parent.name.split('_')[1]}"
        results[key] = {"summary_sha256": file_hash(path), "levels": code_usage(path.parent)}
    write_json(ROOT / "code_usage.json", {"scope": __doc__, "results": results})
    print({"completed_native_evaluations": len(results)})


if __name__ == "__main__":
    main()
