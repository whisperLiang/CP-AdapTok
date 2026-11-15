import pytest
import torch

from cp_adaptok.backbones.causal_state import CausalState, tensor_digest


def test_state_fingerprint_detects_inplace_changes_even_after_previous_read():
    feature = torch.randn(1, 256, 4, 8)
    state = CausalState("test", 0, feature, torch.randn_like(feature), "base")
    before = state.fingerprint()
    assert before == state.fingerprint()
    feature[0, 0, 0, 0] += 0.0001
    assert state.fingerprint() != before
    detached = state.detach()
    assert detached.fingerprint() == state.fingerprint()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA fingerprint equality")
def test_state_digest_is_exactly_equal_on_cpu_and_cuda():
    x = torch.randn(1, 256, 4, 8)
    assert tensor_digest(x) == tensor_digest(x.cuda())


def test_inference_tensor_mutations_are_not_hidden_by_cache():
    with torch.inference_mode():
        feature = torch.randn(1, 256, 4, 8)
        state = CausalState("test", 0, feature, None, "base")
        before = state.fingerprint()
        feature[0, 0, 0, 0] += 0.0001
        assert state.fingerprint() != before
