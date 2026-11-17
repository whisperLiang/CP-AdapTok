"""Targeted behavioral mutations run against isolated checkout copies."""

import argparse
import json

from adaptok_hr.runtime import write_json
from experiments.adaptok_hr.quality import MUTATIONS, run_mutation

NEW_MUTATIONS = [
    ("active_loss_mask", "models/vq.py", 'count = budget if self.loss_scope == "active_prefix" else self.ids.shape[1]',
     "count = self.ids.shape[1]", "test_fine_recovery.py::test_active_vq_loss_and_gradient_are_exact_prefix"),
    ("commitment_beta", "models/vq.py", "codebook + self.beta * commitment", "codebook + commitment",
     "test_fine_recovery.py::test_active_vq_loss_and_gradient_are_exact_prefix"),
    ("ste_gradient", "models/vq.py", "ste = discrete.detach() + (z - z.detach())", "ste = discrete",
     "test_fine_recovery.py::test_reconstruction_ste_reaches_only_active_tokens_and_never_codebook"),
    ("level_lookup", "models/vq.py", "yield book, slice(offset, end)", "yield self.codebook, slice(offset, end)",
     "test_fine_recovery.py::test_per_level_lookup_uses_position_not_id_and_hashes_every_book"),
    ("fine_identity", "models/vq.py", 'zip(("Base", "Medium", "Fine"), self.books)',
     'zip(("Base", "Medium", "Fine"), self.books[:2])',
     "test_fine_recovery.py::test_per_level_lookup_uses_position_not_id_and_hashes_every_book"),
    ("auxiliary_vq_weight", "training/train.py", '(value + encoded.loss_terms(auxiliary)[key]) * .5',
     'value + encoded.loss_terms(auxiliary)[key]',
     "test_fine_recovery.py::test_nested_vq_matches_reconstruction_weights_and_has_no_tail_gradient"),
    ("budget_rng_resume", "training/checkpoint.py", 'state["budget_rng"] = trainer.budget_rng.get_state()',
     'state["budget_rng"] = torch.Generator().manual_seed(7).get_state()',
     "test_fine_recovery.py::test_independent_budget_generator_survives_resume"),
    ("untransmitted_decoder", "models/model.py", "self.decoder(encoded.quantized[:, :budget], reference)",
     "self.decoder(encoded.quantized, reference)",
     "test_fine_recovery.py::test_training_frame_decoder_never_observes_unsent_candidates"),
    ("initialization_rng", "models/vq.py", "with torch.random.fork_rng(devices=[]):", "with torch.no_grad():",
     "test_fine_recovery.py::test_codebook_count_does_not_change_encoder_decoder_initialization"),
    ("inference_config_identity", "training/checkpoint.py",
     'if state["version"] != 1 or state["config_identity"] != stable_hash(state["config"]):',
     "if False:", "test_fine_recovery.py::test_inference_loader_rejects_tampered_configuration_identity"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/adaptok_hr_fine_recovery/mutations.json")
    args = parser.parse_args()
    results = []
    for case in (*MUTATIONS, *NEW_MUTATIONS):
        result = run_mutation(case)
        results.append(result)
        write_json(args.output, {"all_killed": all(r["killed"] for r in results), "results": results})
        print(json.dumps({k: result[k] for k in ("mutation", "killed", "collection_errors")}), flush=True)
    if not all(r["killed"] for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
