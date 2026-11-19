"""Behavior mutations in disposable experiment copies; never edit a live run."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MUTATIONS = [
    (
        "fsq_wrong_zero_offset",
        "tokenizer/fsq.py",
        ".long() + self.half_width",
        ".long() + 0",
        "test_fsq_bijection.py",
    ),
    (
        "first_residual_leaks_gradient",
        "tokenizer/residual_tokenizer.py",
        "latent - reconstructed0.detach()",
        "latent - reconstructed0",
        "test_tier_map_and_prefix.py",
    ),
    (
        "second_residual_not_second_order",
        "tokenizer/residual_tokenizer.py",
        "latent - reconstructed1.detach()",
        "residual1",
        "test_tier_map_and_prefix.py",
    ),
    (
        "tier_priority_reversed",
        "tokenizer/tier_map.py",
        "[groups[tier] for tier in (3, 2, 1)]",
        "[groups[tier] for tier in (1, 2, 3)]",
        "test_tier_map_and_prefix.py",
    ),
    (
        "valid_id_zero_shifted",
        "decoder/latent_canvas.py",
        "self.tokenizer.fsq.ids_to_codes(ids)",
        "self.tokenizer.fsq.ids_to_codes(ids + 1)",
        "test_canvas.py",
    ),
    (
        "canvas_reprojects_old_offsets",
        "decoder/latent_canvas.py",
        "self.count += count",
        "self.count += 0",
        "test_canvas.py",
    ),
    (
        "double_commit_allowed",
        "decoder/causal_state.py",
        "if self.committed:",
        "if False:",
        "test_causal_state_and_canvas.py",
    ),
    (
        "state_storage_aliases_encoder",
        "decoder/causal_state.py",
        "tensor.detach().clone()",
        "tensor.detach()",
        "test_causal_state_and_canvas.py",
    ),
    (
        "psnr_only_false_pass",
        "evaluation/gates.py",
        'r["lpips_gain"] > 0',
        'r["lpips_gain"] > -1',
        "test_smoke_gate.py",
    ),
]


def run_mutation(spec, directory):
    name, filename, before, after, test = spec
    source = directory / "aip_vtok" / filename
    original = source.read_text()
    if before not in original:
        raise ValueError(f"mutation target missing: {name}")
    changed = original.replace(before, after, 1)
    compile(changed, str(source), "exec")
    source.write_text(changed)
    env = dict(os.environ, PYTHONPATH=str(directory), PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="4")
    process = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", f"tests/aip_vtok/{test}"],
        cwd=directory,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    output = process.stdout + process.stderr
    return {
        "name": name,
        "killed": process.returncode == 1 and "FAILED" in output,
        "returncode": process.returncode,
        "output": output,
    }


def main():
    results = []
    for spec in MUTATIONS:
        with tempfile.TemporaryDirectory(prefix="aip-vtok-mutation-") as temporary:
            directory = Path(temporary)
            for folder in ("aip_vtok", "tests/aip_vtok"):
                shutil.copytree(
                    ROOT / folder, directory / folder, ignore=shutil.ignore_patterns("__pycache__")
                )
            for folder in ("cp_adaptok", "third_party"):
                (directory / folder).symlink_to(ROOT / folder, target_is_directory=True)
            (directory / "pyproject.toml").symlink_to(ROOT / "pyproject.toml")
            results.append(run_mutation(spec, directory))
    output = ROOT / "results/aip_vtok/tests/mutations.json"
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({"killed": sum(r["killed"] for r in results), "total": len(results)}))
    if not all(r["killed"] for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
