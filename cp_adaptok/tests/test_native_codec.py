"""Optional integration gates requiring the pinned native extensions and official assets."""

import sys
from types import ModuleType

import pytest
import torch

from cp_adaptok.backbones.native_baseline import NativeBaseline, NativeLatentTeacher
from cp_adaptok.data import read_video
from cp_adaptok.runtime import ROOT, MissingPrerequisites, preflight, seed_everything

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


def test_native_teacher_preflight_requires_decoder_latent_api(monkeypatch, tmp_path):
    import importlib.util

    module = ModuleType("inference_extensions_cuda")
    module.DMCIProxy = type("DMCIProxy", (), {})
    module.DMCLDProxy = type("DMCLDProxy", (), {})
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(importlib.util, "find_spec", lambda _: object())
    weight = tmp_path / "weights.pt"
    weight.touch()
    config = {"loss": {"latent": 1}, "paths": {"ld_checkpoint": str(weight), "intra_checkpoint": str(weight)}}
    with pytest.raises(MissingPrerequisites, match="decoded-latent"):
        preflight(config, "train")


@pytest.fixture
def native_assets():
    if not torch.cuda.is_available():
        pytest.skip("native CUDA integration requires a GPU")
    pytest.importorskip("MLCodec_extensions_cpp")
    pytest.importorskip("inference_extensions_cuda")
    paths = [ROOT / "checkpoints" / name for name in ("cvpr2026_video_ld.pth.tar", "cvpr2026_image.pth.tar")]
    source = ROOT / "data/pretrained_native/mcl_jcv01.mp4"
    if not all(path.exists() for path in [*paths, source]):
        pytest.skip("native integration needs official weights and prepared source data")
    seed_everything(20261115, "cuda:0")
    clip = read_video(dict(path=str(source), frames=2), 256, torch.Generator().manual_seed(47))
    yield paths, clip.to("cuda:0")
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def test_native_bitstreams_decode_i_p_and_independent_reset(native_assets):
    paths, clip = native_assets
    codec = NativeBaseline(*paths, "cuda:0")
    streams = []
    with pytest.raises(ValueError):
        codec.decoded_latent()
    for frame in clip:
        rec, counts, raw = codec.round_trip(frame[None])
        assert rec.shape == frame[None].shape and torch.isfinite(rec).all()
        assert len(raw) == counts["total_bytes"] > counts["payload_bytes"] > 0
        streams.append(raw)
        assert counts["sps_bytes"] + counts["frame_header_bytes"] + counts["payload_bytes"] == len(raw)
        snapshot = codec.decoded_latent()
        expected = snapshot.clone()
        snapshot.zero_()
        torch.testing.assert_close(codec.decoded_latent(), expected, rtol=0, atol=0)
    assert codec.frame_id == 2
    codec.reset()
    for frame, expected in zip(clip, streams):
        rec, _, raw = codec.round_trip(frame[None])
        assert torch.isfinite(rec).all() and raw == expected
    assert codec.frame_id == 2


def test_teacher_targets_are_exact_native_decoded_i_p_latents(native_assets):
    paths, clip = native_assets
    teacher = NativeLatentTeacher(*paths, "cuda:0")
    for channels, frame in zip((256, 128), clip):
        target = teacher.target(frame[None])
        assert target.shape == (1, channels, 16, 16)
        assert torch.isfinite(target).all() and not target.requires_grad
        assert teacher.last_provenance["native_bitstream"] is True
        assert teacher.last_provenance["bytes"] > 0
        assert teacher.last_provenance["latent_source"] == "native_decoder_quantized_y_hat"
        torch.testing.assert_close(target, teacher.native.decoded_latent().float(), rtol=0, atol=0)
    assert teacher.native.frame_id == 2


def test_native_teacher_uses_the_same_padding_domain_as_cp_encoder(native_assets):
    from torch.nn import functional as F

    paths, clip = native_assets
    teacher = NativeLatentTeacher(*paths, "cuda:0")
    rgb = clip[:1, :, :70, :90]
    target = teacher.target(rgb)
    assert target.shape == (1, 256, 8, 8)
    teacher.reset()
    teacher.native.round_trip(F.pad(rgb, (0, 38, 0, 58), mode="replicate"))
    torch.testing.assert_close(target, teacher.native.decoded_latent().float(), rtol=0, atol=0)


def test_native_teacher_preserves_the_callers_stream_and_waits_for_rgb(native_assets):
    paths, clip = native_assets
    caller = torch.cuda.current_stream()
    teacher = NativeLatentTeacher(*paths, "cuda:0")
    assert torch.cuda.current_stream() == caller
    producer = torch.cuda.Stream()
    producer.wait_stream(caller)
    with torch.cuda.stream(producer):
        torch.cuda._sleep(2_000_000)
        rgb = clip[:1].clone()
        actual = teacher.target(rgb)
    caller.wait_stream(producer)
    teacher.reset()
    expected = teacher.target(clip[:1])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
