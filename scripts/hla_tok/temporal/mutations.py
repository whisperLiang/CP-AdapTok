"""Run focused behavior mutations in disposable source copies."""

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from hla_tok.evaluation.io import write_json
from scripts.hla_tok.temporal.inputs import RESULTS
from scripts.hla_tok.v2.protocol import ROOT

STREAM = "hla_tok/streaming/temporal.py"
STATE = "hla_tok/temporal/state.py"
WIRE = "hla_tok/streaming/temporal_wire.py"
TEST = "tests/hla_tok/temporal/"


@dataclass(frozen=True)
class Mutation:
    path: str
    anchors: tuple[tuple[str, str], ...]
    node: str


def mutations() -> dict[str, Mutation]:
    stream_test = TEST + "test_stream.py::"
    contract = TEST + "test_contract.py::"
    return {
        "read_future_frame": Mutation(STREAM, (("for frame_id, (target, budget) in enumerate(frames):",
                                                "for frame_id, (target, budget) in enumerate(list(frames)):"),),
                                      stream_test + "test_stream_does_not_fetch_or_depend_on_a_future_frame"),
        "use_source_instead_of_closed_reference": Mutation(STREAM, (
            ("prediction = predict(profile, reference, header.intra)",
             "prediction = flat[profile.scan[:profile.base_components]] if not header.intra else 0"),),
            stream_test + "test_intra_and_predictive_closed_loop"),
        "enhancement_contaminates_committed_base": Mutation(STREAM, (
            ("self.reference = active.commit_base() if complete else None",
             "self.reference = active.commit_base() if complete else None\n"
             "        if complete and active.prefix > self.profile.base_records:\n"
             "            from dataclasses import replace\n"
             "            self.reference = replace(self.reference, digest=bytes(32))"),),
            stream_test + "test_independent_history_over_70_frames_is_enhancement_independent"),
        "skip_reference_digest": Mutation(STREAM, (("if header.reference_digest != prior.digest:", "if False:"),),
                                          stream_test + "test_invalid_binding_frame_id_and_history_digest"),
        "remove_tail_mask": Mutation("hla_tok/temporal/layout.py", (
            ("self.mask = self._padded(np.ones(MAX_K, dtype=bool))",
             "self.mask = np.ones((len(NATIVE_RANKS), REGIONS), dtype=bool)"),),
            contract + "test_native_layout_covers_each_real_token_once"),
        "incorrect_negative_rounding": Mutation("hla_tok/temporal/quantization.py", (
            ("return quotient + ((double > step) | ((double == step) & ((quotient & 1) != 0)))",
             "return quotient + (double >= step)"),), contract + "test_integer_rounding_ties_and_clipping"),
        "unchecked_integer_overflow": Mutation("hla_tok/temporal/quantization.py", (
            ("if values.min(initial=0) < LATTICE.min or values.max(initial=0) > LATTICE.max:", "if False:"),),
            contract + "test_integer_overflow_is_rejected"),
        "wrong_bitmap_bit_order": Mutation("hla_tok/temporal/sparse.py", (
            ('np.packbits(nonzero, axis=1, bitorder="little")', 'np.packbits(nonzero, axis=1, bitorder="big")'),),
            contract + "test_sparse_physical_mode_and_padding"),
        "invent_zero_block": Mutation("hla_tok/temporal/sparse.py", (
            ("mode[nonzero == 0] = Mode.ZERO", "mode[nonzero <= 2] = Mode.ZERO"),
            ("lengths[nonzero == 0] = 0", "lengths[nonzero <= 2] = 0")), contract + "test_sparse_physical_mode_and_padding"),
        "omit_control_bytes_from_ledger": Mutation(STREAM, (
            ("+ self.record_header_bytes + payload", "+ payload"),), stream_test + "test_record_sizes_bitmap_crc_and_payload_limit"),
        "budget_dependent_requantization": Mutation(STREAM, (
            ("flat = target.reshape(-1)", "flat = target.reshape(-1) + (budget % 10)"),),
            stream_test + "test_independent_budgets_share_wire_and_integer_prefix"),
        "budget_specific_header_field": Mutation(STREAM, (
            ("header = _encoding_header(frame_id, reference, profile, flat, force_intra)",
             "header = _encoding_header(frame_id, reference, profile, flat, force_intra)\n"
             "    from dataclasses import replace\n"
             "    header = replace(header, profile_id=profile.profile_id + budget % 9)"),),
            stream_test + "test_independent_budgets_share_wire_and_integer_prefix"),
        "duplicate_record_applied": Mutation(STREAM, (
            ("if not 0 <= index < self.profile.records or index in self.seen:",
             "if not 0 <= index < self.profile.records:"),),
            stream_test + "test_incremental_reordering_duplicate_overlap_and_crc"),
        "commit_incomplete_base": Mutation(STREAM, (
            ("if self.prefix < self.profile.base_records:", "if False:"),),
            stream_test + "test_incremental_reordering_duplicate_overlap_and_crc"),
        "pad_missing_tokens_as_valid": Mutation(STREAM, (("support = base.support.copy()", "support = np.ones_like(base.support)"),),
                                              stream_test + "test_intra_and_predictive_closed_loop"),
        "read_sealed_test_alias": Mutation("hla_tok/evaluation/data.py", (
            ('if any(path.is_relative_to(sealed) for sealed in self._sealed_paths) or item["sha256"] in self._sealed_hashes:',
             "if False:"),
            ("if not path.is_relative_to(root):", "if False:")), "tests/hla_tok/test_data_boundary.py::test_sealed_test_alias_is_rejected_before_decode"),
    }


def _pytest(root: Path, targets: list[str]) -> subprocess.CompletedProcess:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    return subprocess.run([sys.executable, "-m", "pytest", "-q", *targets], cwd=root, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=environment, check=False)


def run(output: Path) -> None:
    logs = output / "logs" / "targeted_mutations"
    logs.mkdir(parents=True, exist_ok=True)
    cases = mutations()
    with TemporaryDirectory(prefix="hlt-mutations-") as temp:
        root = Path(temp)
        for directory in ("hla_tok", "cp_adaptok", "tests/hla_tok/temporal"):
            shutil.copytree(ROOT / directory, root / directory, ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copy(ROOT / "pyproject.toml", root)
        shutil.copy(ROOT / "tests/hla_tok/test_data_boundary.py", root / "tests/hla_tok")
        baseline = _pytest(root, sorted({test.node for test in cases.values()}))
        (logs / "baseline.log").write_text(baseline.stdout)
        if baseline.returncode:
            raise RuntimeError("targeted mutation baseline failed")
        results = {}
        for name, test in cases.items():
            path = root / test.path
            original = current = path.read_text()
            for before, after in test.anchors:
                if current.count(before) != 1:
                    raise ValueError(f"mutation anchor is ambiguous for {name}")
                current = current.replace(before, after)
            path.write_text(current)
            try:
                replay = _pytest(root, [test.node])
            finally:
                path.write_text(original)
            (logs / f"{name}.log").write_text(replay.stdout)
            killed = replay.returncode == 1 and "FAILED " in replay.stdout
            results[name] = {"status": "KILLED" if killed else "SURVIVED", "test": test.node}
            print(name, results[name]["status"], flush=True)
    status = "PASS" if all(row["status"] == "KILLED" for row in results.values()) else "FAIL"
    write_json(output / "targeted_mutations.json", {"status": status, "results": results,
               "motion_bytes_mutation": "NOT_RUN: HLT v1 registers no motion records"})
    if status == "FAIL":
        raise RuntimeError("targeted behavioral mutation survived")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RESULTS)
    run(parser.parse_args().output)
