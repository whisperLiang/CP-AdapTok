import torch

from aip_vtok.training.losses import bt709_chroma, chroma_loss, crop_plan
from aip_vtok.training.trainer import RNGStreams


def test_signed_chroma_loss_detects_opposite_color_offsets():
    image = torch.zeros(1, 3, 64, 64) + 0.5
    red, blue = image.clone(), image.clone()
    red[:, 0] += 0.15
    blue[:, 2] += 0.15
    assert chroma_loss(image, image) == 0
    assert chroma_loss(red, blue) > 0.01
    assert torch.allclose(bt709_chroma(image), torch.zeros(1, 2, 64, 64), atol=1e-6)


def test_four_native_crops_are_nonoverlapping_and_independently_repeatable():
    rng = RNGStreams(1)
    snapshot = rng.state_dict()
    crops = crop_plan(1080, 1920, rng.crop_rng)
    rng.load_state_dict(snapshot)
    assert crops == crop_plan(1080, 1920, rng.crop_rng)
    assert len(crops) == 4
    for i, (y1, x1) in enumerate(crops):
        for y2, x2 in crops[i + 1 :]:
            assert y1 + 256 <= y2 or y2 + 256 <= y1 or x1 + 256 <= x2 or x2 + 256 <= x1
