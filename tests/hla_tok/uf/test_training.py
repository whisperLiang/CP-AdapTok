import torch

from hla_tok.uf.anchor import AnchorModel
from hla_tok.uf.training import prefix_objective


def test_multirate_objective_updates_enhancement_and_rate_parameters():
    torch.manual_seed(7)
    target = torch.rand(2, 3, 128, 128)
    base = target * 0.5 + 0.2
    model = AnchorModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    prior = model.analysis.weight.detach().clone()
    loss, scalars = prefix_objective(model, target, base, [2400, 1200], None)
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss) and not torch.equal(model.analysis.weight, prior)
    assert model.log_scale.grad is not None and model.log_scale.grad.abs().sum() > 0
    assert model.quantizer.logit.grad is not None and model.quantizer.logit.grad.abs().sum() > 0
    assert scalars["estimated_bpp"].shape == (4, 2)
    assert (scalars["estimated_bpp"][1:] >= scalars["estimated_bpp"][:-1]).all()
