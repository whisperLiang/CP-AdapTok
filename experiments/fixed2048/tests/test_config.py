import json

import pytest

from experiments.fixed2048.cli import read_config


def test_saved_json_configuration_preserves_scientific_float_values(tmp_path):
    config = read_config("configs/fixed2048.yaml")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    restored = read_config(path)
    assert restored == config
    assert isinstance(restored["training"]["lr"], float)


def test_config_boundary_rejects_a_non_mapping_json(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("[1,2]")
    with pytest.raises(ValueError, match="mapping"):
        read_config(path)
