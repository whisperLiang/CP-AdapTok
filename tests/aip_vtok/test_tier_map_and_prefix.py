import torch

from aip_vtok.tokenizer.halton import halton_order
from aip_vtok.tokenizer.residual_tokenizer import ResidualTokenizer
from aip_vtok.tokenizer.tier_map import make_tier_map, scan_plan


def test_halton_is_deterministic_unique_and_covers_image_at_256():
    first = halton_order(17, 30)
    assert torch.equal(first, halton_order(17, 30))
    assert torch.equal(first.sort().values, torch.arange(510))
    ids = first[:256]
    for y in (ids // 30 < 9, ids // 30 >= 9):
        for x in (ids % 30 < 15, ids % 30 >= 15):
            assert 40 < (x & y).sum() < 90


def test_tier_ties_have_stable_scan_and_native_token_limit():
    residual = torch.ones(1, 128, 68, 120)
    tiers = make_tier_map(residual)
    assert tiers.shape == (34, 60) and set(tiers.flatten().tolist()) == {0, 1, 2, 3}
    plan = scan_plan(tiers)
    assert torch.equal(plan.pass_ids[:510], torch.zeros(510, dtype=torch.long))
    assert len(plan.positions) == 8192
    other = scan_plan(make_tier_map(residual))
    assert torch.equal(plan.positions, other.positions)
    assert torch.equal(plan.pass_ids, other.pass_ids)
    pass1_tiers = tiers.flatten()[plan.positions[plan.pass_ids == 1]]
    assert (pass1_tiers[:-1] >= pass1_tiers[1:]).all()


def test_encode_once_has_strict_id_prefix_and_quantized_residual_dependencies():
    torch.manual_seed(7)
    tokenizer = ResidualTokenizer()
    latent = torch.randn(1, 128, 8, 16)
    prediction = torch.zeros_like(latent)
    encoded = tokenizer.encode(latent, prediction)
    for k1, k2 in [(1, 7), (7, 15), (15, len(encoded.ids[0]))]:
        assert torch.equal(encoded.prefix_ids(k1), encoded.prefix_ids(k2)[:, :k1])
    rebuilt0 = tokenizer.synthesize_pass(
        encoded.codes[:, encoded.plan.pass_ids == 0],
        encoded.plan.positions[encoded.plan.pass_ids == 0],
        0,
        (8, 16),
    )
    rebuilt1 = tokenizer.synthesize_pass(
        encoded.codes[:, encoded.plan.pass_ids == 1],
        encoded.plan.positions[encoded.plan.pass_ids == 1],
        1,
        (8, 16),
    )
    torch.testing.assert_close(encoded.residual1, latent - (prediction + rebuilt0).detach())
    torch.testing.assert_close(encoded.residual2, latent - (prediction + rebuilt0 + rebuilt1).detach())
    assert not encoded.residual1.requires_grad and not encoded.residual2.requires_grad
