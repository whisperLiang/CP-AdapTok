import torch


def test_spatial_statistics_distinguish_constant_and_varying_features():
    from experiments.adaptok_hr.fine_recovery.diagnose import feature_statistics

    constant = torch.ones(1, 4, 16)
    identical = feature_statistics(constant)
    varying = feature_statistics(torch.eye(16)[None])
    assert identical["spatial_variance"] == 0
    assert abs(identical["mean_position_cosine"] - 1) < 1e-6
    assert varying["spatial_variance"] > 0
    assert abs(varying["mean_position_cosine"]) < 1e-6


def test_fine_interventions_do_not_read_unsent_tokens():
    from experiments.adaptok_hr.fine_recovery.diagnose import interventions

    z = torch.randn(1, 40, 16)
    discrete = torch.randn_like(z)
    second = torch.randn_like(z)
    actual = interventions(z, discrete, second, 32, 12)
    z[:, 32:] = 1e6
    discrete[:, 32:] = -1e6
    expected = interventions(z, discrete, second, 32, 12)
    assert actual["no_fine"].shape[1] == 12
    for name in actual:
        torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0)
    torch.testing.assert_close(actual["fine_continuous"][:, :12], discrete[:, :12])
