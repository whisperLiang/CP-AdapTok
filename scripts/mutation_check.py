"""Behavior mutations in isolated temporary copies; never edit the live checkout."""

import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MUTATIONS = [
    ("rANS termination", "streaming/entropy_codec.py", "decode", "termination", "test_packet_roundtrip.py"),
    ("CRC validation", "streaming/bitstream.py", "_checked_body", "crc", "test_packet_roundtrip.py"),
    ("contiguous packets", "streaming/packetizer.py", "append", "order", "test_packet_roundtrip.py"),
    ("softmax normalization", "decoder/prefix_cache.py", "append", "denominator", "test_prefix.py"),
    ("missing token mask", "decoder/token_grid_projector.py", "forward", "mask", "test_prefix.py"),
    ("nearest VQ index", "tokenizer/vector_quantizer.py", "forward", "nearest", "test_vq.py"),
    ("single base commit", "streaming/receiver.py", "decode_base", "commit", "test_reference_state.py"),
    ("reference fingerprint", "streaming/receiver.py", "start_frame", "reference", "test_streaming.py"),
    ("display ACK reference", "streaming/session.py", "_display_ack", "ack", "test_adaptive_stream.py"),
    ("deadline send guard", "streaming/transport.py", "send", "expiry", "test_adaptive_stream.py"),
    (
        "retain better base",
        "streaming/sender.py",
        "allocation_points",
        "retention",
        "test_adaptive_stream.py",
    ),
    (
        "charge unsent bytes",
        "streaming/sender.py",
        "allocation_points",
        "remaining",
        "test_adaptive_stream.py",
    ),
]


class Mutator(ast.NodeTransformer):
    def __init__(self, function, kind):
        self.function, self.kind, self.changed = function, kind, False

    def visit_FunctionDef(self, node):
        if node.name != self.function:
            return node
        for child in ast.walk(node):
            if self.changed and self.kind != "expiry":
                break
            self.mutate(child)
        return node

    def mutate(self, child):
        predicates = {
            "termination": "position !=",
            "crc": "CRC.unpack",
            "order": "first_token_index !=",
            "commit": "self.context.committed",
            "reference": "reference_fingerprint !=",
            "ack": "DisplayAck.from_bytes(wire) != expected",
            "expiry": "time.perf_counter()",
        }
        if isinstance(child, ast.If) and self.kind in predicates:
            if predicates[self.kind] in ast.unparse(child.test):
                child.test = ast.Constant(False)
                self.changed = True
        elif self.kind == "denominator" and isinstance(child, ast.Return):
            if isinstance(child.value, ast.BinOp) and isinstance(child.value.op, ast.Div):
                child.value.op = ast.Add()
                self.changed = True
        elif self.kind == "nearest" and isinstance(child, ast.Attribute) and child.attr == "argmin":
            child.attr = "argmax"
            self.changed = True
        elif self.kind == "mask" and isinstance(child, ast.Assign):
            if isinstance(child.value, ast.Call) and isinstance(child.value.func, ast.Attribute):
                if child.value.func.attr == "masked_fill":
                    child.value = child.value.func.value
                    self.changed = True
        elif self.kind == "retention" and isinstance(child, ast.Compare):
            if ast.unparse(child) == "k >= delivered_tokens":
                child.ops[0] = ast.Gt()
                self.changed = True
        elif self.kind == "remaining" and isinstance(child, ast.BinOp):
            if isinstance(child.op, ast.Sub) and ast.unparse(child.right) == "delivered_bytes":
                child.op = ast.Add()
                self.changed = True


def check_mutation(spec):
    name, file, function, kind, test = spec
    with tempfile.TemporaryDirectory(prefix="cp_adaptok_mutation_") as temporary:
        directory = Path(temporary)
        shutil.copytree(
            ROOT / "cp_adaptok", directory / "cp_adaptok", ignore=shutil.ignore_patterns("__pycache__")
        )
        (directory / "third_party").symlink_to(ROOT / "third_party", target_is_directory=True)
        target = directory / "cp_adaptok" / file
        transformer = Mutator(function, kind)
        tree = transformer.visit(ast.parse(target.read_text()))
        if not transformer.changed:
            raise RuntimeError(f"mutation target no longer exists: {name}")
        target.write_text(ast.unparse(ast.fix_missing_locations(tree)) + "\n")
        env = dict(os.environ, OMP_NUM_THREADS="2", PYTHONPATH=str(directory))
        result = subprocess.run(
            [sys.executable, "-m", "pytest", f"cp_adaptok/tests/{test}", "-q"],
            cwd=directory,
            env=env,
            capture_output=True,
            text=True,
            timeout=45,
        )
        log = ROOT / "results/mutations" / f"{kind}.txt"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(result.stdout + result.stderr)
        # A collection failure is not a killed behavior mutation.
        killed = result.returncode == 1 and " failed" in result.stdout
        return {"name": name, "killed": killed, "exit_code": result.returncode, "log": str(log)}


def main():
    results = [check_mutation(spec) for spec in MUTATIONS]
    (ROOT / "results/mutations/summary.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    return 0 if all(item["killed"] for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
