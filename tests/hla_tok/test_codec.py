import pytest
import torch
from torch.nn import functional as F

from hla_tok import BUDGETS, Geometry, HLATok
from hla_tok.models.quantizer import VectorQuantizer
from hla_tok.progressive.geometry import merge_regions, rank_major, region_lengths, split_regions


@pytest.fixture(scope="module", autouse=True)
def keep_cpu_threads_small():
    threads = torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.set_num_threads(threads)


def test_native_padding_and_region_order():
    geometry = Geometry(1080, 1920)
    image = torch.arange(1080).view(1, 1, -1, 1).float().expand(1, 3, 1080, 1920)
    padded = geometry.pad(image)
    assert padded.shape == (1, 3, 1152, 1920)
    assert padded[0, 0, 1080, 0] == 1078
    assert padded[0, 0, -1, 0] == 1007
    assert torch.equal(geometry.crop(padded), image)
    sample = torch.arange(72 * 120).reshape(1, 1, 72, 120)
    regions = split_regions(sample, geometry)
    assert regions.shape == (1, 135, 64, 1)
    assert regions[0, 15, 0, 0] == 8 * 120
    assert regions[0, 1, 0, 0] == 8
    assert torch.equal(merge_regions(regions, geometry), sample)
    with pytest.raises(ValueError):
        Geometry(1080, 128)


def test_rank_major_counts_and_density():
    native = Geometry(1080, 1920)
    crop = Geometry(256, 256)
    local = torch.arange(12).reshape(1, 3, 4)
    assert rank_major(local).tolist() == [[0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]]
    assert region_lengths(4096, native.num_regions) == (31,) * 46 + (30,) * 89
    for budget in (*BUDGETS, 17, 69120):
        lengths = region_lengths(budget, native.num_regions)
        assert sum(lengths) == budget
        assert max(lengths) <= 512
        assert abs(crop.scale_budget(budget) / 4 - budget / 135) <= 0.125
    with pytest.raises(ValueError):
        region_lengths(69121, native.num_regions)


def test_blocked_vq_matches_brute_force_and_lowest_id_wins_ties():
    rng = torch.Generator().manual_seed(2)
    vq = VectorQuantizer(size=13, dim=4, vectors_per_block=9, codes_per_block=5)
    vectors = torch.randn(2, 3, 8, 4, generator=rng)
    with torch.no_grad():
        vq.embedding.weight[7].copy_(vq.embedding.weight[0])
        vectors[0, 0, 0] = vq.embedding.weight[0]
    distances = (vectors[..., None, :] - vq.embedding.weight).square().sum(-1)
    ids = vq(vectors)
    assert torch.equal(ids, distances.argmin(-1))
    assert ids[0, 0, 0] == 0
    assert torch.equal(vq.lookup(ids), vq.embedding.weight[distances.argmin(-1)])


@pytest.mark.parametrize("precision", [torch.float32, torch.bfloat16])
def test_complete_on_demand_and_appended_ids_match_cpu(precision):
    torch.manual_seed(3)
    model = HLATok().eval().to(memory_format=torch.channels_last)
    source = torch.rand(1, 3, 128, 256)
    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=precision == torch.bfloat16):
        whole = model.encode_frame(source)
        sender = model.prepare_frame(source)
        first = model.encode_prefix(sender, 61)
        assert torch.equal(first, whole[:, :61])
        assert torch.equal(model.extend_prefix(sender, 63), whole[:, :63])
        assert torch.equal(model.extend_prefix(sender, 179), whole[:, :179])
        assert torch.equal(model.extend_prefix(sender, 1024), whole)
        assert torch.equal(model.encode_prefix(sender, 1), whole[:, :1])
        assert torch.equal(model.encode_prefix(model.prepare_frame(source), 67), whole[:, :67])
        lengths = (0, 51)
        (left, right) = model.encode_regions(sender, lengths)
        assert left.shape == (1, 0)
        assert torch.equal(right, whole[:, 1:102:2])


def test_receiver_has_only_prefix_and_public_geometry():
    torch.manual_seed(4)
    sender = HLATok().eval()
    receiver = HLATok().eval()
    receiver.load_state_dict(sender.state_dict())
    source = torch.rand(1, 3, 128, 256)
    with torch.inference_mode():
        ids = sender.encode_frame(source, 129)
        original = ids.clone()
        ids[:, 17:] = (ids[:, 17:] + 1) % 8192
        # Perturb the receiver's unused encoder weights to catch accidental feature access.
        for parameter in receiver.encoder.parameters():
            parameter.fill_(float("nan"))
        a = receiver.decode_prefix(original[:, :17], Geometry(128, 256))
        b = receiver.decode_prefix(ids[:, :17], Geometry(128, 256))
        c = receiver.decode_prefix((ids[:, :17] + 1) % 8192, Geometry(128, 256))
        assert a.shape == source.shape and torch.isfinite(a).all()
        assert torch.equal(a, b)
        assert not torch.equal(a, c)


def test_future_frames_cannot_change_previous_ids_or_reconstruction():
    torch.manual_seed(5)
    model = HLATok().eval()
    frames = torch.rand(3, 1, 3, 128, 128)
    altered = torch.cat((frames[:2], 1 - frames[2:]))
    with torch.inference_mode():
        def run(sequence):
            return [
                (ids, model.decode_prefix(ids, Geometry(128, 128)))
                for ids in (model.encode_frame(frame, 31) for frame in sequence)
            ]

        base, changed = run(frames), run(altered)
    for lhs, rhs in zip(base[:2], changed[:2]):
        assert all(torch.equal(x, y) for x, y in zip(lhs, rhs))


def test_shared_training_prefix_reaches_each_module_and_uses_hard_vq_values():
    torch.manual_seed(6)
    model = HLATok()
    source = torch.rand(1, 3, 128, 256)
    received = model.encode_training(source, 81)
    assert torch.equal(received.values, model.quantizer.lookup(received.ids))
    outputs = [model.decode_training(received, count) for count in (23, 81)]
    loss = sum(F.l1_loss(image, source) for image in outputs)
    loss = loss + received.codebook_loss + 0.25 * received.commitment_loss
    loss.backward()
    for module in (model.encoder, model.query_encoder, model.quantizer, model.local_decoder, model.rgb_decoder):
        stats = [p.grad for p in module.parameters() if p.grad is not None]
        assert stats and all(torch.isfinite(g).all() for g in stats)
        assert any(g.count_nonzero() for g in stats)
