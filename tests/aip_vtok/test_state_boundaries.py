import pytest
import torch

from aip_vtok.decoder.causal_state import FrameContext


def test_reference_fingerprint_supports_bf16_without_losing_bits():
    feature = torch.ones(1, 192, 8, 8, dtype=torch.bfloat16)
    state = FrameContext(0, True, None, None).commit_reference_once(feature)
    original = state.fingerprint()
    state.feature[0, 0, 0, 0] = 1.125
    assert state.fingerprint() != original


def test_commit_rejects_wrong_frame_type_channels_and_does_not_lock_on_bad_input():
    context = FrameContext(0, True, None, None)
    with pytest.raises(ValueError, match="reference"):
        context.commit_reference_once(torch.zeros(1, 256, 8, 8))
    assert not context.committed
    context.commit_reference_once(torch.ones(1, 192, 8, 8))


def test_p_reference_requires_its_synchronized_memory():
    context = FrameContext(1, False, None, None)
    with pytest.raises(ValueError, match="memory"):
        context.commit_reference_once(torch.zeros(1, 256, 8, 8))
