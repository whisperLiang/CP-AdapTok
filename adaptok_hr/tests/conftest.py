import pytest
import torch


@pytest.fixture(scope="session", autouse=True)
def cpu_threads():
    torch.set_num_threads(4)


@pytest.fixture
def small_config():
    from adaptok_hr.config import ModelConfig

    return ModelConfig(height=128, width=256, channels=(12, 12, 12, 12, 12, 12, 12),
                       decoder_channels=(12, 12, 12, 12, 12, 12), heads=3,
                       budgets=(4, 8, 16, 32))


@pytest.fixture
def small_model(small_config):
    from adaptok_hr.models.model import AdapTokHR

    torch.manual_seed(7)
    return AdapTokHR(small_config)
