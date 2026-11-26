"""Run one directed implementation mutation at a time in a scratch checkout."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
RATE = "test_baseline.py"
WIRE = "test_bitstream.py"
PRECISION = "test_precision.py"
STREAMING = "test_pht_streaming.py"
NATIVE = "test_native.py"
INCREMENTAL = "test_codec.py::test_mask_and_precision_incremental_decode_refresh_and_reference_loss"
CAUSAL = "test_codec.py::test_one_encoder_call_and_chunk_causality_with_shared_backing_storage"


@dataclass(frozen=True, slots=True)
class Mutation:
    label: str
    file: str
    old: str
    new: str
    witness: str
    gpu: bool = False


MUTATIONS = (
    Mutation("repeat_frame", "pht_tok/codec.py", "torch.cat((frames, padding))",
             "torch.cat((frames[:1].expand_as(frames), padding))",
             "test_codec.py::test_assemble_preserves_input_order_and_pads_only_the_tail", True),
    Mutation("single_frame_rate_denominator", "experiments/pht_tok/baseline.py",
             "bits / (height * width * frames)", "bits / (height * width)", RATE),
    Mutation("reset_each_reference", "pht_tok/official_adapter.py",
             '        self.commit_reference(self.snapshot_reference(intra=True))',
             '        self._warm_anchor = self.snapshot_reference(intra=True)\n'
             '        self.commit_reference(self._warm_anchor)', NATIVE, True),
    Mutation("read_next_frame", "pht_tok/codec.py", "    padding = frames[-1:]",
             "    if len(frames) == CHUNK_FRAMES:\n"
             "        frames = frames.as_strided(frames.size(), frames.stride(),"
             " frames.storage_offset() + frames.stride(0))\n"
             "    padding = frames[-1:]", CAUSAL, True),
    Mutation("rewrite_prefix_header", "pht_tok/bitstream.py", "return self.data[:self.prefix_boundaries[count - 1]]",
             "return b'X' + self.data[1:self.prefix_boundaries[count - 1]]", WIRE),
    Mutation("need_hidden_full_stream", "pht_tok/codec.py", "            for payload in self.pending:",
             "            if not self.parser.complete:\n"
             "                return self.stage\n"
             "            for payload in self.pending:", INCREMENTAL, True),
    Mutation("omit_hyperprior_from_ledger", "pht_tok/bitstream.py",
             '"payload_bytes": list(self.sizes)', '"payload_bytes": list(self.sizes[1:])', WIRE),
    Mutation("use_unreceived_precision", "pht_tok/codec.py",
             "peer.entropy.precision.preview(self._parents, self.stage)",
             "peer.entropy.precision.preview(self._parents, min(3, self.stage + 1))", INCREMENTAL, True),
    Mutation("invalid_cdf_normalization", "pht_tok/precision.py", "CDF_TOTAL - count",
             "CDF_TOTAL + count", PRECISION),
    Mutation("clip_escape_symbols", "pht_tok/entropy.py",
             "return representative(lows, width).astype(np.int8)",
             "return np.clip(representative(lows, width), -8, 8).astype(np.int8)", PRECISION),
    Mutation("repeat_encoder", "pht_tok/codec.py", "        analysis = self.adapter.encode_chunk(assemble(frames), qp)",
             "        analysis = self.adapter.encode_chunk(assemble(frames), qp)\n"
             "        analysis = self.adapter.encode_chunk(assemble(frames), qp)", CAUSAL, True),
    Mutation("decode_old_layers_again", "pht_tok/codec.py", "        self.pending.clear()\n        return self.stage",
             "        return self.stage", INCREMENTAL, True),
    Mutation("pretend_full_synthesis_on_prefix", "pht_tok/codec.py",
             "out = peer.adapter.reconstruct(self.stage,",
             "out = peer.adapter.reconstruct(GROUPS,", INCREMENTAL, True),
    Mutation("allow_unconfirmed_reference", "pht_tok/codec.py",
             '            raise MissingReference("required reference chunk or version is missing")',
             "            pass", INCREMENTAL, True),
    Mutation("include_tail_in_duration", "experiments/pht_tok/baseline.py",
             "bits, duration = 8 * len(raw), frames / fps", "bits, duration = 8 * len(raw), (frames + padding) / fps",
             RATE),
    Mutation("count_late_as_on_time", "pht_tok/streaming.py",
             "if packet.arrival is not None and packet.arrival <= deadline",
             "if packet.arrival is not None", STREAMING),
    Mutation("look_at_future_bandwidth", "pht_tok/streaming.py",
             "        if item.at > now:\n            break\n", "", STREAMING),
)


def apply(mutation: Mutation, scratch: Path) -> None:
    path = scratch / mutation.file
    code = path.read_text()
    if code.count(mutation.old) != 1:
        raise ValueError(f"mutation target is ambiguous: {mutation.label}")
    code = code.replace(mutation.old, mutation.new)
    if mutation.label == "reset_each_reference":
        code = code.replace("self.reference.tensors, self.reference.intra", "self._warm_anchor.tensors, True")
    path.write_text(code)


def execute(mutation: Mutation) -> dict:
    with TemporaryDirectory(prefix="pht-mutation-") as folder:
        scratch = Path(folder)
        shutil.copytree(ROOT / "pht_tok", scratch / "pht_tok", ignore=shutil.ignore_patterns("__pycache__"))
        (scratch / "experiments").mkdir()
        shutil.copy(ROOT / "experiments/__init__.py", scratch / "experiments/__init__.py")
        shutil.copytree(ROOT / "experiments/pht_tok", scratch / "experiments/pht_tok",
                        ignore=shutil.ignore_patterns("__pycache__"))
        apply(mutation, scratch)
        env = dict(os.environ)
        env["PYTHONPATH"] = f"{scratch}:{ROOT}:{env.get('PYTHONPATH', '')}"
        test = str(ROOT / "tests/pht_tok" / mutation.witness)
        run = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", test], cwd=scratch, env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120)
        killed = run.returncode == 1 and "failed" in run.stdout
        return {"name": mutation.label, "witness": mutation.witness, "gpu": mutation.gpu, "killed": killed,
                "returncode": run.returncode, "result": "\n".join(run.stdout.splitlines()[-12:])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for mutation in MUTATIONS:
        if mutation.gpu != args.gpu:
            continue
        result = execute(mutation)
        rows.append(result)
        print(result["name"], result["killed"], flush=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")
    if not all(row["killed"] for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
