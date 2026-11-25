from hashlib import sha256

import pytest

from scripts.hla_tok.continuous.inputs import require_digest, verify_model_state


def test_checkpoint_sha256_rejects_changed_bytes(tmp_path):
    checkpoint = tmp_path / "C0.pt"
    checkpoint.write_bytes(b"published checkpoint")
    expected = sha256(checkpoint.read_bytes()).hexdigest()
    assert require_digest(checkpoint, expected) == expected
    checkpoint.write_bytes(b"changed checkpoint")
    with pytest.raises(ValueError, match="SHA-256"):
        require_digest(checkpoint, expected)


def test_state_integrity_rejects_mutation():
    import torch

    model = torch.nn.Linear(3, 2)
    from scripts.hla_tok.continuous.inputs import model_digest

    original = model_digest(model)
    assert verify_model_state(model, original) == original
    with torch.no_grad():
        model.weight[0, 0] += 1
    with pytest.raises(ValueError, match="state_dict"):
        verify_model_state(model, original)
