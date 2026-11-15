import pytest
import torch

from cp_adaptok.backbones.causal_state import CausalState, tensor_digest


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA compiler test")
def test_compiled_integer_fingerprint_is_bit_exact_for_native_state_sizes(monkeypatch):
    from cp_adaptok.backbones.causal_state import mix_fingerprint_words
    from cp_adaptok.runtime import ROOT

    if (ROOT / ".build/driver/libcuda.so").exists():
        monkeypatch.setenv("TRITON_LIBCUDA_PATH", str(ROOT / ".build/driver"))
    compiled = torch.compile(mix_fingerprint_words, fullgraph=True, dynamic=True)
    for size in [1, 19, 256 * 136 * 240, 128 * 136 * 240]:
        values = torch.randint(-(2**31), 2**31, (size,), device="cuda", dtype=torch.int32)
        positions = torch.arange(1, size + 1, device="cuda", dtype=torch.int64)
        torch.testing.assert_close(
            compiled(values, positions), mix_fingerprint_words(values, positions), rtol=0, atol=0
        )


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
