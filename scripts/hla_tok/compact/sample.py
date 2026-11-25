"""Write the frozen full-resolution Compact sample at matched byte limits."""

from hla_tok.compression.profiles import PROFILES_BY_LABEL
from hla_tok.evaluation.io import write_json
from hla_tok.streaming.compact_wire import max_budget
from hla_tok.streaming.continuous import MAX_K
from scripts.hla_tok.compact.inputs import OUTPUT, compact_inputs
from scripts.hla_tok.compact.matrix import sample
from scripts.hla_tok.continuous.inputs import locked_output, run_arguments
from scripts.hla_tok.v2.protocol import json_at


def main() -> None:
    parser = run_arguments(__doc__)
    parser.set_defaults(output=OUTPUT)
    args = parser.parse_args()
    with locked_output(args.output):
        inputs, config = compact_inputs(args.v3, args.output)
        baseline = json_at(args.output / "no_training_selection.json")
        tuned = json_at(args.output / "training_selection.json")
        assets = json_at(args.output / "compact_profile_manifest.json")
        key = baseline["main_asset"]
        if tuned["adopted"]:
            key = tuned["asset"]
            assets[key] = tuned["spec"]
        profile = PROFILES_BY_LABEL[assets[key]["profile"]]
        k5 = max_budget(config["frame_wire_byte_caps"][0], profile)
        eq = max_budget(config["equal_payload"]["cap_bytes"], profile, payload_only=True)
        cases = [("Compact_K16320", MAX_K), ("Compact_5Mbps", k5), ("Compact_EqualPayload", eq)]
        for label, budget in cases:
            sample(inputs, config, args.output, assets, key, baseline["order"], budget, label)
        write_json(args.output / "samples/compact_cases.json", {"asset": key, "order": baseline["order"],
                                                              "cases": dict(cases)})
        inputs.finish(args.output / "sampling_inputs.json")


if __name__ == "__main__":
    main()
