import torch

from aip_vtok.decoder.causal_state import FrameContext
from aip_vtok.joint.decoder import JointSLD, migrate_sld
from aip_vtok.sparse_decoder.models import SLD


def test_preserves_received_coefficients_and_migrates_heads():
    torch.manual_seed(1)
    old = SLD()
    for p in old.outputs.parameters():
        p.data.normal_(std=0.01)
    new = JointSLD()
    order = torch.stack((torch.randperm(128), torch.randperm(128)))
    report = migrate_sld(new, old.state_dict(), order)
    assert not report["shape_mismatches"]
    assert all(value["max_abs"] < 1e-5 for value in report["initialization_comparison"].values())
    mask = torch.rand(1, 8, 8, 12) > 0.7
    observed = mask.repeat_interleave(6, dim=1)
    values = torch.randn(1, 48, 8, 12) * observed
    result = new(values, mask.float(), FrameContext(0, True, None, None), 510, order[0])
    assert result.shape == (1, 128, 8, 12)
    assert torch.equal(result[:, :48][observed], values[observed])
    assert result[:, 48:].abs().max() > 0
