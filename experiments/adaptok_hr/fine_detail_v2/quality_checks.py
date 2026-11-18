"""Run the existing 24 behavioral mutations plus targeted Fine-path corruptions."""

import json

from adaptok_hr.runtime import write_json
from experiments.adaptok_hr.fine_recovery.quality import MUTATIONS, NEW_MUTATIONS
from experiments.adaptok_hr.quality import run_mutation

from .run import ROOT

TEST = "test_fine_detail_v2.py::"
CASES = [
    ("local_neighborhood", "models/tokenizer.py", "residual = values - F.avg_pool2d(padded, 3, 1)",
     "residual = values", TEST + "test_spatial_residual_uses_neighbors_and_does_not_change_coarse_tokens"),
    ("suffix_position_order", "training/fine_detail.py", "[low - offset:high - offset]", "[:high - low]",
     TEST + "test_suffix_objective_targets_only_sent_extra_ids_and_commits_main"),
    ("suffix_auxiliary_weight", "training/fine_detail.py", "(weight / len(outputs))", "weight",
     TEST + "test_suffix_objective_targets_only_sent_extra_ids_and_commits_main"),
    ("suffix_target_residual", "training/fine_detail.py", "rgb.float() - coarse.float()", "rgb.float()",
     TEST + "test_suffix_targets_use_transmission_order_and_valid_native_pixels"),
    ("suffix_valid_geometry", "training/fine_detail.py", "valid = counts > 0", "valid = counts >= 0",
     TEST + "test_suffix_targets_use_transmission_order_and_valid_native_pixels"),
    ("suffix_rms", "training/fine_detail.py", "valid.sum(-1).clamp_min(1).sqrt()", "valid.sum(-1).clamp_min(1)",
     TEST + "test_angular_loss_ignores_padding_flat_regions_and_matches_hand_value"),
    ("flat_fine_direction", "models/tokenizer.py", "unit[:, 0] = 1", "unit[:, 1] = 1",
     TEST + "test_spatial_residual_uses_neighbors_and_does_not_change_coarse_tokens"),
]

DECODER_CASES = [
    ("fine_mask_promotes_bf16_reference", "models/decoder.py", "fine_map[:, -1:].to(x.dtype)", "fine_map[:, -1:]",
     "test_fine_detail_decoder.py::test_zero_residual_keeps_bf16_reference_and_p_frame_bitwise"),
    ("fine_residual_missing_mask", "models/decoder.py", "self.output(x) * mask", "self.output(x)",
     "test_fine_detail_decoder.py::test_residual_decoder_start_and_training_preserve_absent_fine"),
    ("fine_residual_nonzero_start", "models/decoder.py", "nn.init.zeros_(self.output.weight)",
     "nn.init.ones_(self.output.weight)",
     "test_fine_detail_decoder.py::test_residual_decoder_start_and_training_preserve_absent_fine"),
    ("high_frequency_weight", "training/losses.py", 'loss + self.weights["detail"] * terms["detail"]',
     'loss + terms["detail"]', "test_fine_detail_decoder.py::test_highpass_loss_matches_reflected_rgb_and_nested_weights"),
]


def main():
    results = []
    for case in (*MUTATIONS, *NEW_MUTATIONS, *CASES, *DECODER_CASES):
        result = run_mutation(case)
        results.append(result)
        write_json(ROOT / "mutations.json", {"all_killed": all(r["killed"] for r in results), "results": results})
        print(json.dumps({"case": case[0], "killed": result["killed"]}), flush=True)
    if not all(r["killed"] for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
