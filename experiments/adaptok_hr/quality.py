"""Explicit substitutes for unavailable crapper/dryer/mutator executables."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from difflib import SequenceMatcher
from pathlib import Path

from adaptok_hr.runtime import write_json
from scripts.quality_audit import analyze_file

ROOT = Path(__file__).resolve().parents[2]
MUTATIONS = [
    ("13bit", "codec/bitstream.py", "(pending << 13) | code", "(pending << 12) | code", "test_bitstream.py::test_13bit_roundtrip_including_endpoints"),
    ("edge_mask", "models/window_attention.py", "value=False", "value=True", "test_model.py::test_window_padding_and_actual_local_computation"),
    ("position_order", "models/tokenizer.py", "current = int(distances.argmax())", "current = 0", "test_model.py::test_native_positions_and_distributed_prefix"),
    ("future", "models/model.py", "video.unbind(2)", "video.flip(2).unbind(2)", "test_model.py::test_past_is_independent_of_future_and_gop_reset"),
    ("prefix", "models/model.py", "self.encode_features(frame, state).ids[:, :budget]", "self.encode_features(frame, state).ids[:, -budget:]", "test_model.py::test_nested_prefix_and_only_transmitted_values"),
    ("continuous", "models/vq.py", "ste = discrete.detach() + (z - z.detach())", "ste = z", "test_model.py::test_vq_discrete_forward_gradients_and_range"),
    ("gop", "models/temporal_state.py", "index % gop == 0", "index == 0", "test_model.py::test_past_is_independent_of_future_and_gop_reset"),
    ("history", "codec/codec.py", " or packet.previous != state.chain", "", "test_stream.py::test_65_frame_closed_loop_gop_missing_stream_and_truncated_history"),
    ("frozen", "models/scorer.py", "requires_grad_(False)", "requires_grad_(True)", "test_training.py::test_frozen_tokenizer_scorer_stage_changes_only_scorer"),
    ("missing_mask", "models/tokenizer.py", "mask[:, :, indices] = 1", "mask[:, :, indices] = 0", "test_model.py::test_scatter_places_received_values_and_marks_absence"),
    ("aux_commit", "training/train.py", "return total, main.state, row", "return total, alternative.state, row", "test_training.py::test_auxiliary_budget_does_not_change_committed_reference"),
    ("resume_optimizer", "training/checkpoint.py", "trainer.optimizer.load_state_dict(state[\"optimizer\"])", "pass  # dropped optimizer history", "test_training.py::test_optimizer_resume_matches_uninterrupted_and_sampling"),
    ("global_psnr", "evaluation/evaluate.py", "\"psnr_global\": psnr(mse)", "\"psnr_global\": sum(r[\"psnr\"] for r in rows) / len(rows)", "test_evaluation.py::test_psnr_global_and_frame_mean_stay_distinct"),
]


def similarity_pairs(functions):
    pairs = []
    for i, first in enumerate(functions):
        for second in functions[i + 1:]:
            matcher = SequenceMatcher(None, first[2], second[2], autojunk=False)
            if matcher.quick_ratio() >= .8:
                ratio = matcher.ratio()
                if ratio >= .8:
                    pairs.append({"a": first[:2], "b": second[:2], "similarity": ratio})
    return pairs


def audit(coverage, output):
    measured = json.loads(Path(coverage).read_text())
    rows, functions = [], []
    for file in sorted((ROOT / "adaptok_hr").rglob("*.py")):
        if "tests" not in file.parts:
            blocks, bodies = analyze_file(file, measured)
            rows.extend(blocks)
            functions.extend(bodies)
    rows = list({(r["file"], r["function"], r["line"]): r for r in rows}.values())
    result = {"tools": "radon+coverage CRAP, AST SequenceMatcher; named crapper/dryer unavailable",
              "functions": rows, "crap_over_10": [r for r in rows if r["crap"] >= 10],
              "duplicates_over_08": similarity_pairs(functions)}
    write_json(output, result)
    print(json.dumps({"crap_over_10": len(result["crap_over_10"]),
                      "duplicates_over_08": len(result["duplicates_over_08"])}))


def run_mutation(case):
    name, relative, original, replacement, test = case
    with tempfile.TemporaryDirectory(prefix="adaptok_hr_mutation_") as directory:
        temporary = Path(directory)
        shutil.copytree(ROOT / "adaptok_hr", temporary / "adaptok_hr", ignore=shutil.ignore_patterns("__pycache__"))
        path = temporary / "adaptok_hr" / relative
        source = path.read_text()
        if source.count(original) != 1:
            raise ValueError(f"ambiguous mutation target: {name}")
        path.write_text(source.replace(original, replacement, 1))
        env = {**os.environ, "PYTHONPATH": str(temporary) + os.pathsep + str(ROOT), "OMP_NUM_THREADS": "4"}
        result = subprocess.run([sys.executable, "-m", "pytest", "-q", "adaptok_hr/tests/" + test,
                                 "--junitxml=" + str(temporary / "result.xml")],
                                cwd=temporary, env=env, capture_output=True, text=True, timeout=60)
        xml = ET.parse(temporary / "result.xml").getroot()
        failures, errors = list(xml.iter("failure")), list(xml.iter("error"))
        killed = result.returncode == 1 and bool(failures) and not errors
        return {"mutation": name, "target": relative, "test": test, "killed": killed,
                "returncode": result.returncode, "assertion_failures": len(failures), "collection_errors": len(errors),
                "output_tail": (result.stdout + result.stderr)[-2000:]}


def mutate(output):
    results = []
    for case in MUTATIONS:
        result = run_mutation(case)
        results.append(result)
        print(json.dumps({k: result[k] for k in ("mutation", "killed", "collection_errors")}), flush=True)
    write_json(output, {"tool": "isolated actual behavioral mutations, substitute for unavailable mutator",
                        "all_killed": all(r["killed"] for r in results), "results": results})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("audit", "mutate"))
    parser.add_argument("--coverage", default="results/adaptok_hr/coverage.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.mode == "audit":
        audit(args.coverage, args.output)
    else:
        mutate(args.output)


if __name__ == "__main__":
    main()
