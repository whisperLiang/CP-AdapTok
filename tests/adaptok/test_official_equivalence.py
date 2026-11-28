import os

import pytest
import torch


@pytest.mark.integration
@pytest.mark.gpu
def test_real_official_tile_and_group_masks():
    checkpoint = os.getenv("ADAPTOK_OFFICIAL_CHECKPOINT")
    video = os.getenv("ADAPTOK_OFFICIAL_VIDEO")
    if not checkpoint or not video:
        pytest.skip("set ADAPTOK_OFFICIAL_CHECKPOINT and ADAPTOK_OFFICIAL_VIDEO for the real GPU test")
    from baselines.adaptok.model_adapter import OfficialAdapter
    from baselines.adaptok.video_reader import read_input

    adapter = OfficialAdapter(checkpoint)
    frames, _ = read_input(video, frames=adapter.frame_num)
    tile = frames[:, :, :adapter.input_size, :adapter.input_size].permute(1, 0, 2, 3)[None].cuda()
    before = adapter.state_digest()
    with torch.inference_mode():
        direct = adapter.model.encode_eval(tile, with_latent_mask=False)
        reference = adapter.model.decode_eval(direct["encoded"], latent_mask=None).float().clamp(0, 1)
        encoded = adapter.encode(tile)
        actual = adapter.decode(encoded, adapter.total_tokens)
        torch.testing.assert_close(reference, actual, rtol=0, atol=0)
        torch.testing.assert_close(encoded["encoded"], adapter.encode(tile)["encoded"], rtol=0, atol=0)
        for k in [512, 1024, 2048]:
            mask = adapter.budget_mask(k, 1)
            assert mask.reshape(1, adapter.groups, -1).sum(-1).tolist() == [[k // adapter.groups] * adapter.groups]
            native = adapter.model.decode_eval(encoded["encoded"], latent_mask=mask).float().clamp(0, 1)
            torch.testing.assert_close(native, adapter.decode(encoded, k), rtol=0, atol=0)
    assert before == adapter.state_digest()
