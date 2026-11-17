import pytest

from experiments.fixed2048.seal import seal_experiment


def test_test_seal_is_immutable_and_contains_the_complete_protocol(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"trained checkpoint")
    (tmp_path / "cpat_test.json").write_text("{}")
    config = {"manifest_dir": str(tmp_path), "final_test": {"frames": 64}}
    plan = {"R1": {"checkpoint": str(checkpoint), "paths": ["R1", "R2"], "references": ["full2048"]}}
    output = tmp_path / "seal.json"
    result = seal_experiment(config, plan, output)
    digest = result["models"]["R1"]["sha256"]
    assert result["protocols"][digest]["paths"] == ["R1", "R2"]
    assert result["no_training_or_selection_after_seal"]
    with pytest.raises(FileExistsError, match="immutable"):
        seal_experiment(config, plan, output)
