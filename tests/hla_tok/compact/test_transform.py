import pytest
import torch

from hla_tok.compression.scalar import StaticScalar
from hla_tok.compression.transform import CompactTransform
from hla_tok.streaming.compact import native_ranks
from hla_tok.streaming.continuous import MAX_K, REGIONS, RankOrder


def test_rank_aware_projection_uses_original_rank_for_an_arbitrary_subset():
    center = torch.tensor([[0.0] * 32, [16.0] * 32, [-8.0] * 32])
    basis = torch.eye(32)[:, :4].repeat(3, 1, 1)
    basis[2] *= -1
    transform = CompactTransform.from_pca(center, basis)
    ranks = torch.tensor([2, 0, 2, 1])
    source = center[ranks][None].expand(2, -1, -1) + 2
    encoded = transform.encode(source, ranks)
    expected = torch.tensor([[-2.0] * 4, [2.0] * 4, [-2.0] * 4, [2.0] * 4])
    assert torch.equal(encoded, expected.expand(2, -1, -1))
    recovered = transform.decode(encoded[:, :3], ranks[:3])
    assert torch.equal(recovered[..., :4], source[:, :3, :4])
    assert torch.equal(recovered[..., 4:], center[ranks[:3]][None, :, 4:].expand(2, -1, -1))


@pytest.mark.parametrize("compute_dtype", [torch.float32, torch.bfloat16])
def test_full_pca_is_numerically_invertible(compute_dtype):
    gen = torch.Generator().manual_seed(78)
    basis = torch.linalg.qr(torch.randn(32, 32, generator=gen)).Q[None]
    center = torch.randn(1, 32, generator=gen)
    transform = CompactTransform.from_pca(center, basis)
    data = torch.randn((2, 19, 32), generator=gen).to(compute_dtype)
    ranks = torch.arange(19)
    with torch.autocast("cpu", enabled=compute_dtype == torch.bfloat16, dtype=torch.bfloat16):
        latent = transform.encode(data, ranks)
        recovered = transform.decode(latent, ranks)
    assert latent.dtype == compute_dtype
    assert (data.float() - recovered.float()).abs().max() < (
        1e-5 if compute_dtype == torch.float32 else 0.03)


def test_scalar_scale_is_static_over_a_prefix():
    scalar = StaticScalar(torch.tensor([[1.0, 0.5], [2.0, 0.5]]), bits=4)
    u = torch.tensor([[[7., -1.], [1., 4.5], [-16., -5.]]])
    ranks = torch.tensor([0, 0, 1])
    q = scalar.encode(u, ranks)
    assert torch.equal(q, torch.tensor([[[7, -2], [1, 7], [-8, -8]]], dtype=torch.int8))
    assert torch.equal(q[:, :1], scalar.encode(u[:, :1], ranks[:1]))
    assert torch.equal(scalar.decode(q, ranks)[0, 2], torch.tensor([-16., -4.]))


def test_reversed_order_keeps_partial_native_rank():
    order = RankOrder.reversed()
    ranks = native_ranks(order)
    assert len(ranks) == MAX_K
    assert ranks[:120].eq(120).all()
    assert ranks[120:120 + REGIONS].eq(119).all()
    assert ranks[-REGIONS:].eq(0).all()


@pytest.mark.gpu
def test_static_integer_payload_fingerprint_agrees_across_devices():
    from hla_tok.compression.packing import pack_signed

    values = torch.tensor([[[-9., -.5], [0.5, 7.5], [2., -3.]]])
    ranks = torch.tensor([0, 1, 0])
    scales = torch.tensor([[1., .5], [.25, 2.]])
    scalar = StaticScalar(scales, bits=6)
    cpu = scalar.encode(values, ranks)
    gpu = scalar.cuda().encode(values.cuda(), ranks.cuda()).cpu()
    assert torch.equal(cpu, gpu)
    assert pack_signed(cpu[0], 6) == pack_signed(gpu[0], 6)


def test_transform_rejects_inconsistent_frame_or_rank_ids():
    t = CompactTransform()
    with pytest.raises(ValueError, match="nonempty vectors"):
        t.encode(torch.zeros(2, 32), torch.arange(2))
    with pytest.raises(ValueError, match="int64 Rank ID"):
        t.encode(torch.zeros(1, 2, 32), torch.arange(2, dtype=torch.int32))
