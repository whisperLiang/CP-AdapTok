import copy

from aip_vtok.evaluation.gate1 import assess_gate1


def valid_seed(seed):
    budgets = {
        str(k): {
            "frames": 174,
            "psnr_global_mse": 22 + i,
            "lpips": 0.5 - i * 0.05,
            "dists": 0.4 - i * 0.05,
            "chroma_retention": 0.8,
        }
        for i, k in enumerate((510, 1024, 2048, 4096, 8192))
    }
    return {
        "seed": seed,
        "training_steps": 1000,
        "summary": {"paired": budgets, "independent": copy.deepcopy(budgets)},
        "health_summary": {
            str(p): {"minimum_unique_ids": 10, "minimum_dimension_entropy_bits": 0.5} for p in range(3)
        },
        "intervention_summary": {
            f"pass{p}_{kind}": {"ci95_low": 0.001} for p in (1, 2) for kind in ("shuffle", "cross_video")
        },
        "maximum_independent_rgb_error": 0,
        "independent_ids_all_equal": True,
    }


LIMITS = {
    "core_psnr_gain_db": 0.5,
    "minimum_high_budget_psnr_db": 24,
    "minimum_chroma_retention": 0.75,
    "minimum_dimension_entropy_bits": 0.05,
}


def test_gate1_requires_both_seeds_and_all_modalities():
    one, two = valid_seed(1), valid_seed(2)
    assert not assess_gate1([one], LIMITS)["passed"]
    assert assess_gate1([one, two], LIMITS)["passed"]
    two["summary"]["paired"]["2048"]["lpips"] = 0.9
    assert not assess_gate1([one, two], LIMITS)["passed"]


def test_gate1_rejects_chroma_collapse_or_noncausal_reference_drift():
    one, two = valid_seed(1), valid_seed(2)
    two["summary"]["independent"]["8192"]["chroma_retention"] = 0.5
    assert not assess_gate1([one, two], LIMITS)["passed"]
    one, two = valid_seed(1), valid_seed(2)
    one["health_summary"]["2"]["minimum_dimension_entropy_bits"] = 0
    assert not assess_gate1([one, two], LIMITS)["passed"]
    one, two = valid_seed(1), valid_seed(2)
    two["maximum_independent_rgb_error"] = 1e-7
    assert not assess_gate1([one, two], LIMITS)["passed"]
