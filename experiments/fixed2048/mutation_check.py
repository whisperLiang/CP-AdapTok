"""Local mutation checks in disposable copies; never mutate a running experiment."""

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
        "continuous_uses_codebook",
        "codec.py",
        "return self.vq.project_out(self.vq.project_features(features))",
        "return self.vq(features).embeddings",
        "test_codec.py",
    ),
    (
        "full_reference_zeroed",
        "codec.py",
        "feature = output.feature",
        "feature = torch.zeros_like(output.feature)",
        "test_codec.py",
    ),
    ("backbone_train_mode", "codec.py", "self.intra.eval()", "self.intra.train(mode)", "test_codec.py"),
    (
        "historical_reference_2048",
        "codec.py",
        ".embeddings[:, :128]",
        ".embeddings[:, :2048]",
        "test_codec.py",
    ),
    (
        "resume_duplicates_step",
        "checkpoint.py",
        'return artifact["step"], copy.deepcopy',
        'return artifact["step"] + 1, copy.deepcopy',
        "test_train.py",
    ),
    (
        "grid_wrong_axis",
        "checkpoint.py",
        "size=target.grid,",
        "size=target.grid[::-1],",
        "test_checkpoint.py",
    ),
    (
        "raw_latent_scaled_in_r0",
        "codec.py",
        "latent = encoded.latent\n",
        "latent = encoded.latent * 2\n",
        "test_codec.py",
    ),
]


def run_mutation(spec, directory):
    name, filename, before, after, test = spec
    source = directory / "experiments/fixed2048" / filename
    text = source.read_text()
    # Python formatting can change quote style; match the current source explicitly.
    if before not in text:
        raise ValueError(f"mutation target no longer exists: {name}")
    source.write_text(text.replace(before, after, 1))
    env = dict(os.environ, PYTHONPATH=str(directory), PYTHONDONTWRITEBYTECODE="1")
    process = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", f"experiments/fixed2048/tests/{test}"],
        cwd=directory,
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
    )
    return {
        "mutation": name,
        "killed": process.returncode == 1,
        "exit_code": process.returncode,
        "output": process.stdout + process.stderr,
    }


def main():
    rows = []
    for spec in MUTATIONS:
        with tempfile.TemporaryDirectory(prefix="fixed2048-mutation-") as temporary:
            directory = Path(temporary)
            shutil.copytree(
                ROOT / "experiments", directory / "experiments", ignore=shutil.ignore_patterns("__pycache__")
            )
            (directory / "cp_adaptok").symlink_to(ROOT / "cp_adaptok", target_is_directory=True)
            (directory / "third_party").symlink_to(ROOT / "third_party", target_is_directory=True)
            (directory / "pyproject.toml").symlink_to(ROOT / "pyproject.toml")
            rows.append(run_mutation(spec, directory))
    output = ROOT / "results/fixed2048/mutation_check.json"
    output.write_text(json.dumps(rows, indent=2) + "\n")
    if not all(row["killed"] for row in rows):
        raise RuntimeError("a mutation survived or the test failed to execute; inspect raw output")
    print(f"{len(rows)}/{len(rows)} mutations killed")


if __name__ == "__main__":
    main()
