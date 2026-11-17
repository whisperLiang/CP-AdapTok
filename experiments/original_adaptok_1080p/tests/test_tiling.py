import pytest
import torch

from experiments.original_adaptok_1080p.tiling import TileLayout, extract_batches, stitch


@pytest.mark.parametrize("stride,count,padded", [(128, 135, (1152, 1920)), (96, 220, (1088, 1952))])
def test_native_1080_geometry(stride, count, padded):
    plan = TileLayout(1080, 1920, 128, stride)
    assert len(plan.coordinates) == count
    assert plan.padded_shape == padded


def test_native_1080_identity_is_pixel_exact():
    clip = torch.randint(256, (16, 3, 1080, 1920), dtype=torch.uint8).float() / 255
    plan = TileLayout(1080, 1920)
    reconstructed = stitch(list(extract_batches(clip, plan, 4)), plan)
    assert reconstructed.shape == clip.shape
    assert torch.equal(clip, reconstructed)


@pytest.mark.parametrize("stride", [4, 3])
def test_timeline_and_coordinates_survive_stitching(stride):
    clip = torch.arange(5 * 3 * 7 * 11, dtype=torch.float32).reshape(5, 3, 7, 11)
    plan = TileLayout(7, 11, 4, stride)
    result = stitch(list(extract_batches(clip, plan, 2)), plan)
    torch.testing.assert_close(clip, result, atol=0.001, rtol=1e-6)


@pytest.mark.parametrize("stride", [0, 129])
def test_rejects_gaps_or_invalid_stride(stride):
    with pytest.raises(ValueError):
        TileLayout(1080, 1920, 128, stride)


def test_bottom_right_padding_reflects_original_pixels():
    clip = torch.arange(7 * 11).reshape(1, 1, 7, 11).float()
    batches = list(extract_batches(clip, TileLayout(7, 11, 4, 4), 4))
    assert batches[-1][0][-1] == (4, 8)
    last = batches[-1][1][-1]
    assert last[0, 0, -1, -1] == clip[0, 0, -2, -2]
