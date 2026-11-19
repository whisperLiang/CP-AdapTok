import torch

from aip_vtok.training.trainer import RNGStreams


def test_four_rng_streams_restore_independently():
    streams = RNGStreams(1)
    for name in streams.names:
        torch.rand(5, generator=getattr(streams, name))
    snapshot = streams.state_dict()
    expected = {name: torch.rand(12, generator=getattr(streams, name)) for name in streams.names}
    restored = RNGStreams(999)
    restored.load_state_dict(snapshot)
    torch.rand(100, generator=restored.crop_rng)
    for name in streams.names[:-1]:
        assert torch.equal(torch.rand(12, generator=getattr(restored, name)), expected[name])
    restored.load_state_dict(snapshot)
    assert torch.equal(torch.rand(12, generator=restored.crop_rng), expected["crop_rng"])
