"""Targeted behavioral mutations in disposable packages."""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hla_tok.evaluation.io import write_json

CASES = (
    ("rank_reordering", "progressive/geometry.py", "local.transpose(1, 2).flatten(1, 2)",
     "local.flatten(1, 2)", "test_codec.py::test_rank_major_counts_and_density"),
    ("future_leak", "evaluation/causality.py", "for frame in frames:",
     "for frame in [*frames[1:], frames[-1]]:", "test_codec.py::test_future_frames_cannot_change_previous_ids_or_reconstruction"),
    ("ignore_received_ids", "models/quantizer.py", "return self.embedding(ids)",
     "return self.embedding(torch.zeros_like(ids))", "test_codec.py::test_receiver_has_only_prefix_and_public_geometry"),
    ("weight_only_bypass", "models/codec.py", "values = self.quantizer.lookup(ids)",
     "values = torch.zeros(*ids.shape, 32, device=ids.device)", "test_codec.py::test_receiver_has_only_prefix_and_public_geometry"),
    ("drop_padding_bits", "streaming/fixed13.py", '"actual_bpp": byte_count * 8 / pixels',
     '"actual_bpp": tokens * TOKEN_BITS / pixels', "test_fixed13.py::test_rate_counts_written_padding_and_only_real_source_frames"),
    ("read_sealed_alias", "evaluation/data.py", 'raise PermissionError("manifest entry aliases sealed Test content")',
     "return path.read_bytes()", "test_data_boundary.py::test_sealed_test_alias_is_rejected_before_decode"),
)


def mutate(root, output, item):
    name, rel, before, after, test = item
    with TemporaryDirectory(prefix="hla-mutation-") as tmp:
        directory = Path(tmp)
        shutil.copytree(root / "hla_tok", directory / "hla_tok", ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(root / "tests/hla_tok", directory / "tests", ignore=shutil.ignore_patterns("__pycache__"))
        target = directory / "hla_tok" / rel
        text = target.read_text()
        if text.count(before) != 1:
            raise ValueError(f"mutation target is ambiguous: {name}")
        target.write_text(text.replace(before, after))
        env = os.environ | {"PYTHONPATH": os.pathsep.join((str(directory), str(root))), "CUDA_VISIBLE_DEVICES": ""}
        process = subprocess.run([sys.executable, "-m", "pytest", "-q", "--tb=short", f"tests/{test}"],
                                 cwd=directory, env=env, capture_output=True, text=True)
        log = process.stdout + process.stderr
        (output / f"{name}.txt").write_text(log)
        return {"mutation": name, "killed": process.returncode == 1 and "1 failed" in log and "ERROR collecting" not in log}


def main():
    root = Path(__file__).resolve().parents[2]
    output = root / "results/hla_tok/phase0/mutations"
    output.mkdir(parents=True, exist_ok=True)
    rows = [mutate(root, output, item) for item in CASES]
    write_json(output / "summary.json", rows)
    if not all(row["killed"] for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
