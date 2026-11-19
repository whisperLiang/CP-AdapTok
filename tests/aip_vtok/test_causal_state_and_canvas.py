"""Gate 0 state contract; token-canvas gates are intentionally not claimed."""

import pytest
import torch

from aip_vtok.decoder.causal_state import FrameContext, ReferenceState


def test_reference_commits_once_and_owns_detached_storage():
    context = FrameContext(0, True, None, None)
    feature = torch.ones(1, 192, 8, 8, requires_grad=True)
    state = context.commit_reference_once(feature)
    before = state.fingerprint()
    with torch.no_grad():
        feature.zero_()
    assert state.feature.sum() > 0
    assert state.fingerprint() == before
    assert not state.feature.requires_grad
    with pytest.raises(RuntimeError, match="committed"):
        context.commit_reference_once(feature)


def test_initial_state_and_gop32_boundary():
    state = ReferenceState()
    assert state.is_intra(0, 32)
    context = FrameContext(31, False, None, torch.ones(1, 256, 8, 8))
    state = context.commit_reference_once(torch.ones(1, 256, 8, 8))
    assert state.is_intra(32, 32)
    assert not state.is_intra(32, 64)
    with pytest.raises(ValueError, match="consecutive"):
        state.is_intra(33, 32)


def test_committed_memory_also_owns_storage():
    memory = torch.ones(1, 256, 8, 8)
    state = FrameContext(1, False, None, memory).commit_reference_once(torch.ones_like(memory))
    memory.zero_()
    assert state.memory.sum() > 0
