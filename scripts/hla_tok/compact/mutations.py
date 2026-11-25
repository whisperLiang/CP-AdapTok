"""Exercise narrowly targeted behavior faults in an isolated copy of the codec."""

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from hla_tok.evaluation.io import write_json
from scripts.hla_tok.compact.inputs import OUTPUT
from scripts.hla_tok.continuous.inputs import run_arguments
from scripts.hla_tok.v2.protocol import ROOT


@dataclass(frozen=True)
class Mutation:
    name: str
    source: str
    old: str
    new: str
    test: str


C = "hla_tok/compression/"
S = "hla_tok/streaming/"
E = "scripts/hla_tok/compact/"
WIRE = "test_compact_wire.py"
CODEC = "test_codec_behavior.py"
TRANSFORM = "test_transform.py"
MUTATIONS = (
    Mutation("wrong_native_rank", C + "transform.py", "weights.index_select(0, ranks)",
             "weights.index_select(0, ranks.flip(0))", TRANSFORM + "::test_rank_aware_projection_uses_original_rank_for_an_arbitrary_subset"),
    Mutation("wrong_static_scale", C + "scalar.py", "codes.float() * self.scale", "codes.float() + self.scale",
             TRANSFORM + "::test_scalar_scale_is_static_over_a_prefix"),
    Mutation("wrong_sign_bit", C + "packing.py", "((unsigned ^ sign) - sign)", "(unsigned - sign)",
             WIRE + "::test_signed_words_have_canonical_wire_bytes"),
    Mutation("int4_stored_as_int8", C + "packing.py", "if bits == 8:", "if bits in (4, 8):",
             WIRE + "::test_signed_words_have_canonical_wire_bytes"),
    Mutation("wrong_int6_cross_byte", C + "packing.py", "output[..., 1] = (b >> 2)",
             "output[..., 1] = (b >> 3)", WIRE + "::test_signed_words_have_canonical_wire_bytes"),
    Mutation("metadata_bytes_omitted", S + "compact_wire.py", '"total_bytes": HEADER_BYTES + payload_bytes',
             '"total_bytes": payload_bytes', WIRE + "::test_physical_packet_sizes_and_identity"),
    Mutation("recalibrate_per_budget", C + "scalar.py", "_centers(self.scales, ranks)",
             "_centers(self.scales, ranks) * values.shape[1]", TRANSFORM + "::test_scalar_scale_is_static_over_a_prefix"),
    Mutation("wrong_prefix", S + "compact.py", "projected[:, :budget]", "projected[:, -budget:]",
             CODEC + "::test_prefix_excludes_untransmitted_token_information"),
    Mutation("resend_old_tokens", S + "compact_wire.py", "self.payload[start * stride:stop * stride]",
             "self.payload[:count * stride]", WIRE + "::test_incremental_bytes_missing_and_out_of_order"),
    Mutation("unused_token_leak", S + "compact.py", "return self.encode_codes(projected[:, :budget])",
             "return self.encode_codes(projected[:, :budget] + full.mean())",
             CODEC + "::test_prefix_excludes_untransmitted_token_information"),
    Mutation("future_frame_read", S + "compact.py", "enumerate(frames)", "enumerate(list(frames))",
             CODEC + "::test_stream_yields_before_reading_future_frames"),
    Mutation("frozen_parameter_update_ignored", E + "adapter.py",
             "return verify_model_state(model, expected_sha256)", "return expected_sha256",
             "test_training_contract.py::test_frozen_c0_attestation_detects_updates"),
    Mutation("test_reads", E + "inputs.py", 'reader.catalog("train")', 'reader.catalog("test")',
             "test_calibration_plan.py::test_train_plan_reads_only_train_catalog"),
)


def _runner(root: Path, test: str) -> subprocess.CompletedProcess:
    env = os.environ | {"CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"}
    return subprocess.run([sys.executable, "-m", "pytest", "-q", f"tests/hla_tok/compact/{test}"],
                          cwd=root, env=env, text=True, capture_output=True, check=False)


def _case(root: Path, log: Path, mutant: Mutation) -> dict:
    source = root / mutant.source
    original = source.read_text()
    if original.count(mutant.old) < 1:
        raise ValueError(f"mutation anchor absent: {mutant.name}")
    source.write_text(original.replace(mutant.old, mutant.new, 1))
    try:
        result = _runner(root, mutant.test)
    finally:
        source.write_text(original)
    (log / f"{mutant.name}.txt").write_text(result.stdout + result.stderr)
    killed = result.returncode == 1 and "FAILED tests/hla_tok/compact/" in result.stdout
    return {"mutation": mutant.name, "test": mutant.test, "exit_code": result.returncode, "killed": killed}


def run(output: Path) -> None:
    logs = output / "logs" / "targeted_mutations"
    logs.mkdir(parents=True, exist_ok=True)
    ignore = shutil.ignore_patterns("__pycache__", ".pytest_cache")
    with TemporaryDirectory(prefix="hla-compact-mutation-") as name:
        root = Path(name)
        for folder in ("hla_tok", "scripts/hla_tok", "tests/hla_tok/compact"):
            shutil.copytree(ROOT / folder, root / folder, ignore=ignore)
        for folder in ("configs", "cp_adaptok", "pyproject.toml"):
            (root / folder).symlink_to(ROOT / folder)
        unchanged = _runner(root, "")
        (logs / "baseline.txt").write_text(unchanged.stdout + unchanged.stderr)
        if unchanged.returncode:
            raise RuntimeError("directed mutation baseline failed")
        evidence = [_case(root, logs, mutant) for mutant in MUTATIONS]
    complete = all(r["killed"] for r in evidence)
    write_json(output / "targeted_mutations.json", {"status": "PASS" if complete else "FAIL",
                                                   "mutations": evidence, "killed": sum(r["killed"] for r in evidence)})
    if not complete:
        raise RuntimeError("one or more targeted faults survived")


def main() -> None:
    parser = run_arguments(__doc__)
    parser.set_defaults(output=OUTPUT)
    run(parser.parse_args().output)


if __name__ == "__main__":
    main()
