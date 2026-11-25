"""HLA-Tok-C evaluator. Each stage opens only the assets needed for inference."""

from pathlib import Path

from scripts.hla_tok.continuous.dynamic import dynamic_phase
from scripts.hla_tok.continuous.inputs import locked_output, open_inputs, run_arguments
from scripts.hla_tok.continuous.phase1 import native_phase, ordering_phase, replay
from scripts.hla_tok.v2.protocol import json_at


def _require_pass(output: Path, *names: str) -> None:
    for name in names:
        path = output / f"{name}.json"
        if not path.is_file() or json_at(path)["status"] != "PASS":
            raise RuntimeError(f"required preceding audit is absent or failing: {path}")


def _phase2(v3: Path, output: Path, stage: str) -> None:
    _require_pass(output, "reproduction", "prefix_audit", "causality_audit", "dynamic_state_audit")
    inputs = open_inputs(v3)
    if stage == "wire-checks":
        from scripts.hla_tok.continuous.wire_checks import run_wire_checks

        run_wire_checks(inputs, output)
    else:
        from scripts.hla_tok.continuous.wire_eval import run_wire_evaluation

        _require_pass(output, "wire_prefix_audit", "wire_roundtrip_audit")
        run_wire_evaluation(inputs, v3, output)
    inputs.finish(output / f"{stage}_run.json")


def run(v3: Path, output: Path, stage: str) -> None:
    if stage == "phase0":
        replay(v3, output)
    elif stage == "native":
        native_phase(v3, output)
    elif stage == "ordering":
        _require_pass(output, "reproduction")
        ordering_phase(v3, output)
    elif stage == "dynamic":
        _require_pass(output, "prefix_audit", "causality_audit", "closed_state_audit")
        inputs = open_inputs(v3)
        dynamic_phase(inputs, output)
        inputs.finish(output / "dynamic_run.json")
    else:
        _phase2(v3, output, stage)


def main() -> None:
    parser = run_arguments(__doc__)
    parser.add_argument("stage", choices=("phase0", "native", "ordering", "dynamic", "wire-checks", "wire"))
    args = parser.parse_args()
    with locked_output(args.output):
        run(args.v3.resolve(), args.output.resolve(), args.stage)


if __name__ == "__main__":
    main()
