from aip_vtok.evaluation.gates import assess_smoke


def test_smoke_rejects_flat_or_perceptually_reversed_budgets():
    initial = {str(k): {"psnr_global_mse": 10, "lpips": 1, "dists": 1} for k in (510, 2048, 8192)}
    final = {str(k): {"psnr_global_mse": 20, "lpips": 0.5, "dists": 0.5} for k in (510, 2048, 8192)}
    assert not assess_smoke(initial, final, 1.0, 0.5)["passed"]
    final["2048"] = {"psnr_global_mse": 21, "lpips": 0.4, "dists": 0.4}
    final["8192"] = {"psnr_global_mse": 22, "lpips": 0.3, "dists": 0.3}
    assert assess_smoke(initial, final, 1.0, 0.5)["passed"]
    final["8192"]["lpips"] = 0.41
    assert not assess_smoke(initial, final, 1.0, 0.5)["passed"]
