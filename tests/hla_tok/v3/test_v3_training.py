import pytest
import torch

from hla_tok import BUDGETS, Geometry, HLATok
from hla_tok.training.checkpoint import ResumeIdentity, read_checkpoint, save_checkpoint
from hla_tok.training.v3 import objective, prefix_weights, reconstruct_tokens, stratified_budget
from scripts.hla_tok.v3.protocol import CROP, NATIVE, PROTOCOL, SamplePlan, curriculum, read_config


def test_curricula_and_stratified_budgets_match_the_registered_protocol():
    config = read_config(PROTOCOL)
    expected = {"dense": (8000, 4000), "continuous": (2000, 10000, 8000),
                "quantized": (4000, 6000), "progressive": (8000, 12000)}
    for stage, lengths in expected.items():
        assert tuple(leg.updates for leg in curriculum(config, stage)) == lengths
    counts = config["progressive"]["proportions"]
    for offset in (0, 100, 19900):
        sample = [stratified_budget(1, update, tuple(counts)) for update in range(offset, offset + 100)]
        assert [sample.count(budget) for budget in BUDGETS] == counts
        other_seed = [stratified_budget(2, update, tuple(counts)) for update in range(offset, offset + 100)]
        assert sample != other_seed


@pytest.mark.parametrize("native_budget", (4096, 16320, 24512))
def test_progressive_encodes_once_and_decodes_physical_prefixes(native_budget):
    from unittest.mock import patch

    geometry = Geometry(128, 128)
    weights = prefix_weights(geometry, native_budget, 16320, 0.5)
    codec = HLATok("R2").eval()
    source = torch.rand(1, 3, 128, 128)
    with (patch.object(codec, "encode_training", wraps=codec.encode_training) as encoder,
          patch.object(codec, "_decode_features", wraps=codec._decode_features) as decoder):
        pixels, tokens, _ = reconstruct_tokens(codec, source, weights, quantize=True)
    assert encoder.call_count == 1
    assert tokens.ids.shape[1] == max(weights)
    assert list(pixels) == list(weights)
    assert sum(weights.values()) == 1.5
    assert decoder.call_count == len(weights) == (1 if native_budget == 16320 else 2)
    for call, budget in zip(decoder.call_args_list, weights, strict=True):
        assert torch.equal(call.args[0], tokens.values[:, :budget])


def test_two_progressive_reconstructions_charge_vq_once():
    geometry = Geometry(128, 128)
    weights = prefix_weights(geometry, 4096, 16320, 0.5)
    codec = HLATok("R2")
    source = torch.rand(1, 3, 128, 128)
    pixels, tokens, _ = reconstruct_tokens(codec, source, weights, quantize=True)

    def pixel_loss(image, predicted, crops):
        return {"rgb_mse": (predicted - image).square().mean()}

    terms, _ = objective(codec, source, pixel_loss, (), weights, quantize=True)
    expected = sum((pixels[budget] - source).square().mean() * weight for budget, weight in weights.items())
    assert torch.equal(terms["rgb_mse"], expected)
    assert torch.equal(terms["vq_codebook"], tokens.codebook_loss)
    assert torch.equal(terms["vq_commitment"], tokens.commitment_loss)
    sum(terms.values()).backward()
    assert codec.quantizer.embedding.weight.grad.count_nonzero()
    assert codec.query_encoder.output.weight.grad.count_nonzero()


def test_independent_receiver_owns_only_synthesis_and_decodes_actual_bytes():
    from copy import deepcopy

    from hla_tok.streaming.fixed13 import pack_tokens_13bit

    codec = HLATok("R2", rgb_variant="D1").eval()
    receiver = deepcopy(codec.receiver())
    geometry = Geometry(128, 256)
    ids = codec.encode_frame(torch.rand(1, 3, 128, 256), 65)
    payload = pack_tokens_13bit(ids[0].tolist())
    assert set(dict(receiver.named_children())) == {"quantizer", "local_decoder", "rgb_decoder"}
    assert torch.equal(receiver.decode_bytes(payload, 65, geometry), codec.decode_prefix(ids, geometry))
    assert not torch.equal(receiver.decode_prefix((ids + 1) % 8192, geometry), codec.decode_prefix(ids, geometry))
    with pytest.raises(ValueError):
        receiver.decode_bytes(payload[:-1], 65, geometry)
    with pytest.raises(ValueError):
        receiver.decode_bytes(b"", 0, geometry)


def test_resume_restores_schedule_samples_optimizer_and_parameters(tmp_path):
    from copy import deepcopy

    from hla_tok.models.spatial import DenseReconstruction, RGBDecoder, SpatialEncoder
    from scripts.hla_tok.v3.training import optimizer_for

    config = read_config(PROTOCOL)
    leg = curriculum(config, "dense")[0]
    model = DenseReconstruction(SpatialEncoder(), RGBDecoder())
    resumed = deepcopy(model)
    optimizer, scheduler = optimizer_for(model, leg, config)
    generator = torch.Generator().manual_seed(991)
    basename = tmp_path / "step_00001.pt"
    identity = ResumeIdentity("dense/test", "config", "dataset", "git", "code")

    def update(network, optim, schedule):
        frame = torch.rand(1, 3, 128, 128, generator=generator)
        optim.zero_grad()
        (network(frame) - frame).square().mean().backward()
        optim.step()
        schedule.step()
        return frame

    update(model, optimizer, scheduler)
    save_checkpoint(basename, identity, model, optimizer, scheduler, generator,
                    step=1, data_cursor=1, counters={"sample_chain": "abc", "seen_pixels": 128**2})
    expected = update(model, optimizer, scheduler)
    later_optimizer, later_scheduler = optimizer_for(resumed, leg, config)
    restored = read_checkpoint(basename, identity).restore(resumed, later_optimizer, later_scheduler, generator)
    assert restored["data_cursor"] == 1
    assert restored["counters"] == {"sample_chain": "abc", "seen_pixels": 128**2}
    assert torch.equal(expected, update(resumed, later_optimizer, later_scheduler))
    assert later_scheduler.get_last_lr() == scheduler.get_last_lr()
    assert all(torch.equal(tensor, resumed.state_dict()[name]) for name, tensor in model.state_dict().items())


def test_training_plan_reads_only_train_at_the_predeclared_native_coordinates():
    class Reader:
        def catalog(self, split):
            assert split == "train"
            return {"train-video": {"content_id": "train-video", "official_split": "train", "role": "train",
                                    "resolution": [1920, 1080], "resized": False,
                                    "frame_hashes": [{"sha256": "digest"}]}}

        def read(self, split, video, index):
            assert (split, video, index) == ("train", "train-video", 0)
            return original

    original = torch.arange(1920 * 1080, dtype=torch.int32).reshape(1, 1080, 1920).expand(3, -1, -1)
    config = read_config(PROTOCOL)
    plan = SamplePlan(Reader(), [CROP, NATIVE], 21, config)
    paired = SamplePlan(Reader(), [CROP, NATIVE], 21, config)
    assert plan.chains == paired.chains
    assert plan.chains != SamplePlan(Reader(), [CROP, NATIVE], 22, config).chains
    for raw, geometry, _, sample in plan.batches(0, len(plan.records)):
        left, top, width, height = sample["crop_xyhw"]
        expected = original[:, top:top + height, left:left + width]
        if sample["flip_horizontal"]:
            expected = expected.flip(-1)
        assert torch.equal(expected, raw)
        assert len(sample["perceptual_crops_yx"]) == (2 if geometry == NATIVE else 1)
    with pytest.raises(PermissionError):
        plan._load(plan.records[0] | {"split": "validation"})
