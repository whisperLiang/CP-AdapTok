from dataclasses import asdict

import pytest
import torch

from experiments.fixed2048.checkpoint import FORMAT, load_reconstructor
from experiments.fixed2048.tests.test_codec import make_model


def test_real_checkpoint_boundary_strict_loads_and_rejects_missing_weights(tmp_path, monkeypatch):
    import experiments.fixed2048.checkpoint as checkpoint

    torch.set_num_threads(2)
    original = make_model().eval()

    def factory(config):
        model = make_model()
        model.config = config
        return model

    monkeypatch.setattr(checkpoint, "CausalProgressiveCodec", factory)
    artifact = {
        "format_version": 1,
        "model_config": asdict(original.config),
        "model": original.state_dict(),
        "status": "trained",
        "step": 4000,
        "config": {"training": {}},
    }
    path = tmp_path / "source.pt"
    torch.save(artifact, path)
    model, identity = load_reconstructor(path, "cpu")
    assert identity["strict_load"] and identity["checkpoint_step"] == 4000
    for name, value in model.state_dict().items():
        assert torch.equal(value, original.state_dict()[name])
    artifact["experiment_format"] = FORMAT
    torch.save(artifact, path)
    loaded, _ = load_reconstructor(path, "cpu")
    assert loaded.config == model.config
    artifact["model"].pop("tokenizer.content")
    torch.save(artifact, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_reconstructor(path, "cpu")
    artifact["status"] = "random_weights_structure_only"
    torch.save(artifact, path)
    with pytest.raises(ValueError, match="trained"):
        load_reconstructor(path, "cpu")
