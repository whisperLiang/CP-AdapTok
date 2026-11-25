"""Inject focused codec regression faults in isolated Python test processes."""

from __future__ import annotations

import argparse
import importlib.abc
import importlib.machinery
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from hla_tok.evaluation.io import write_json

ROOT = Path(__file__).resolve().parents[3]
TESTS = "tests/hla_tok/uf_v2/"


@dataclass(frozen=True)
class Fault:
    name: str
    module: str
    original: str
    replacement: str
    test: str
    gpu: bool = False


RECORDS = "hla_tok.uf_v2.records"
EVALUATE = "scripts.hla_tok.uf_v2.evaluate"
NATIVE = "hla_tok.uf_v2.decoder"
BYTES = "test_v2_measurement.py::test_actual_byte_ledger_includes_base_native_payload_and_record_headers"
STREAM = "test_v2_stream.py::test_fragments_decode_once_and_preserve_base"
NATIVE_STATE = "test_v2_native.py::test_native_d1_caches_current_features_and_never_updates_reference_with_display"
FAULTS = [
    Fault("repeat Base commit", RECORDS, 'if self.frame is not None:\n            raise WireError("base was already',
          'if self.frame is not None and False:\n            raise WireError("base was already',
          "test_v2_stream.py::test_base_cannot_commit_twice"),
    Fault("replay accepted enhancement", RECORDS, "del self.buffer[:consumed]", "pass", STREAM),
    Fault("accept truncated enhancement", RECORDS, "if self.frame is None or self.buffer:",
          "if self.frame is None:", "test_v2_stream.py::test_finish_refuses_truncated_enhancement_and_zero_path_does_not_run_head"),
    Fault("skip refinement dependency", RECORDS,
          "bands = self.geometry.groups.grid[0] if self.bank.static else 0\n        consumed",
          "bands = self.geometry.groups.grid[0] if self.bank.static else 0\n"
          "        if len(buffer) >= 5 and buffer[4] == 1:\n"
          "            self.levels[int.from_bytes(buffer[:4], 'little')] = 1\n        consumed",
          "test_v2_stream.py::test_bad_record_does_not_allow_render[dependency]"),
    Fault("accept a different prefix family", "hla_tok.uf.wire", "magic != MAGIC or tag != self.tag",
          "magic != MAGIC", "test_v2_stream.py::test_incomplete_base_and_model_tag"),
    Fault("ignore the native Base and hyperprior", EVALUATE,
          '"native_joint_y_z_bytes": family.base_end - APP_BASE_OVERHEAD', '"native_joint_y_z_bytes": 0', BYTES),
    Fault("ignore the I-frame bytes", EVALUATE,
          '"status": "PASS", "bytes": len(prefix), "base_bytes": family.base_end,',
          '"status": "PASS", "bytes": len(prefix) - (family.base_end if prefix[21] & 1 else 0), '
          '"base_bytes": family.base_end,', BYTES),
    Fault("ignore record metadata", EVALUATE, '"enhancement_headers": records * RECORD_OVERHEAD',
          '"enhancement_headers": 0', BYTES),
    Fault("report estimate instead of physical length", EVALUATE,
          '"status": "PASS", "bytes": len(prefix), "base_bytes": family.base_end,',
          '"status": "PASS", "bytes": receiver.symbol_count, "base_bytes": family.base_end,', BYTES),
    Fault("rewrite the selected prefix", "hla_tok.uf.wire", "return self.raw[:self.ends[index]], index",
          'return b"X" + self.raw[1:self.ends[index]], index', BYTES),
    Fault("incorrect integer CDF remainder order", "hla_tok.uf_v2.static_entropy",
          "np.argsort(-remain", "np.argsort(remain", "test_v2_static_entropy.py::test_integer_table_exact_largest_remainder"),
    Fault("allow early Test access", "hla_tok.evaluation.data",
          'raise PermissionError("Test remains sealed")', 'return {}',
          "test_v2_measurement.py::test_read_guard_denies_test_and_preserves_zero_count"),
    Fault("exclude full-max generation from end-to-end clock", EVALUATE,
          "start = perf_counter()\n                    out = encode(codec, rgb, index, encode_cap)",
          "out = encode(codec, rgb, index, encode_cap)\n                    start = perf_counter()",
          "test_v2_measurement.py::test_end_to_end_clock_includes_full_max_generation"),
    Fault("skip full-max family selection", EVALUATE, "encode_cap = cap if demand else MAX_APPLICATION_BYTES",
          "encode_cap = cap", "test_v2_measurement.py::test_end_to_end_clock_includes_full_max_generation"),
    Fault("use next frame context for an earlier display", NATIVE,
          "base.context)","(None if intra else self._video().proxy.ufv2_display_snapshot()[2]))",
          NATIVE_STATE, True),
    Fault("replace Base with enhancement output", NATIVE,
          "return restore_rgb(pixels.float(), *base.info.dimensions)",
          "return base.rgb.copy_(restore_rgb(pixels.float(), *base.info.dimensions))", NATIVE_STATE, True),
]


class Loader(importlib.machinery.SourceFileLoader):
    def __init__(self, module: str, path: str, fault: Fault):
        super().__init__(module, path)
        self.fault = fault

    def get_code(self, fullname):
        del fullname
        source = self.get_data(self.path).decode()
        if source.count(self.fault.original) != 1:
            raise RuntimeError(f"mutation anchor differs for {self.fault.name}")
        new = source.replace(self.fault.original, self.fault.replacement)
        return compile(new, self.path, "exec")


class Finder(importlib.abc.MetaPathFinder):
    def __init__(self, fault: Fault):
        self.fault = fault

    def find_spec(self, fullname, path, target=None):
        if fullname != self.fault.module:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is None or spec.origin is None:
            raise ImportError(f'cannot load fault target {fullname}')
        spec.loader = Loader(fullname, spec.origin, self.fault)
        return spec


def run_child(fault: Fault):
    import pytest

    sys.meta_path.insert(0, Finder(fault))
    return pytest.main(["-q", "--tb=short", TESTS + fault.test])


def execute(fault: Fault, output: Path) -> dict:
    command = [sys.executable, "-m", "scripts.hla_tok.uf_v2.mutations", "--inject", fault.name]
    result = subprocess.run(command, cwd=ROOT, check=False, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    killed = result.returncode == 1 and " failed" in result.stdout and "ERROR" not in result.stdout
    filename = f"case-{FAULTS.index(fault) + 1}.log"
    (output / filename).write_text(result.stdout)
    return asdict(fault) | {"result": "KILLED" if killed else "SURVIVED" if result.returncode == 0 else "ERROR",
                            "returncode": result.returncode, "log": filename}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--inject")
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    if args.inject:
        return run_child(next(fault for fault in FAULTS if fault.name == args.inject))
    args.output.mkdir(parents=True, exist_ok=True)
    rows = [execute(fault, args.output) for fault in FAULTS if fault.gpu == args.gpu]
    write_json(args.output / "summary.json", {"cases": rows, "counts": {
        status: sum(row["result"] == status for row in rows) for status in ("KILLED", "SURVIVED", "ERROR")}})
    return 0


if __name__ == "__main__":
    sys.exit(main())
