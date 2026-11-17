import pytest

from cp_adaptok.runtime import read_config
from experiments.fixed2048.train import require_training_config


def test_training_cannot_restart_after_final_seal(tmp_path):
    config = read_config("configs/fixed2048.yaml")
    seal = tmp_path / "seal.json"
    seal.write_text("{}")
    config["test_seal"] = str(seal)
    with pytest.raises(ValueError, match="sealed"):
        require_training_config(config)
