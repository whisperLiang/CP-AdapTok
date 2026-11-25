"""Directed behavior mutations in the existing disposable-package runner."""

import argparse
from pathlib import Path

from hla_tok.evaluation.io import write_json
from scripts.hla_tok.continuous.inputs import RESULTS
from scripts.hla_tok.mutation_check import CASES as BASE_CASES
from scripts.hla_tok.mutation_check import mutate
from scripts.hla_tok.v2.protocol import ROOT

W = "streaming/continuous_wire.py"
C = "streaming/continuous.py"
WT = "continuous/test_wire_behavior.py::"
CT = "continuous/test_continuous_codec.py::"
DT = "continuous/test_dynamic_behavior.py::"

CASES = (
    ("wrong_rank_order", C, "return cls(Order.O1, NATIVE_RANKS[::-1], device)",
     "return cls(Order.O1, NATIVE_RANKS, device)", CT + "test_rank_major_vector_mapping_and_inverse"),
    ("wrong_token_offset", W, "TOKEN_DIM, frame_id, token_offset, count,",
     "TOKEN_DIM, frame_id, 0, count,", WT + "test_enhancement_contains_only_the_unpublished_suffix_and_preserves_prefix"),
    ("header_dropped_from_total_bpp", W, '"total_bpp": 8 * total_bytes / pixels',
     '"total_bpp": 8 * payload_bytes / pixels', WT + "test_bpp_and_rate_charge_real_header_for_each_packet"),
    ("bf16_sent_as_float32", W, '(3, torch.bfloat16, torch.int16, "<i2")',
     '(3, torch.float32, torch.int32, "<i4")',
     WT + "test_wire_roundtrip_bits_dimensions_and_actual_lengths[bf16-dtype2-2]"),
    ("resend_prefix_in_extension", W, "truncate(full, end)[:, start:]",
     "truncate(full, end)", WT + "test_enhancement_contains_only_the_unpublished_suffix_and_preserves_prefix"),
    ("read_future_before_output", C, "for source, budget in frames:",
     "for source, budget in list(frames):", DT + "test_current_frame_can_be_emitted_before_future_input_is_read"),
    ("budget_history_bleeds_into_next_frame", C,
     "    for source, budget in frames:\n"
     "        prefix = truncate(order.apply(encode_continuous(model, source)), budget)\n"
     "        yield prefix, receiver.decode(prefix, order)",
     "    feedback = None\n"
     "    for source, budget in frames:\n"
     "        driven = source if feedback is None else (source + feedback) / 2\n"
     "        prefix = truncate(order.apply(encode_continuous(model, driven)), budget)\n"
     "        feedback = receiver.decode(prefix, order)\n"
     "        yield prefix, feedback",
     DT + "test_budget_switches_do_not_carry_reconstruction_into_next_frame"),
    ("unreceived_tail_enters_recovery", C, "return values[:, :budget]", "return values",
     CT + "test_receiver_uses_only_the_received_prefix"),
    ("changed_model_weights_ignored", "training/v2.py",
     "digest.update(state[name].detach().cpu().contiguous().numpy().tobytes())",
     "digest.update(b'weights ignored')", "continuous/test_inputs.py::test_state_integrity_rejects_mutation"),
    BASE_CASES[-1],
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RESULTS)
    args = parser.parse_args()
    logs = args.output / "mutations"
    logs.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in CASES:
        row = mutate(ROOT, logs, case)
        print(f'{row["mutation"]}: {"PASS" if row["killed"] else "FAIL"}', flush=True)
        rows.append(row)
    passed = all(row["killed"] for row in rows)
    write_json(args.output / "mutation_audit.json", {"status": "PASS" if passed else "FAIL", "cases": rows})
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
