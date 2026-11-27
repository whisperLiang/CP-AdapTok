from types import SimpleNamespace

import pytest
import torch

from baselines.dcvc_uf.frames import BaseFrame, FrameInfo
from baselines.hla_tok_uf_v2.evaluate import receive
from baselines.hla_tok_uf_v2.latent import Segmentation
from baselines.hla_tok_uf_v2.records import BatchReceiver, BatchRecords, Display
from baselines.hla_tok_uf_v2.wire import APP_BASE_OVERHEAD, RECORD_OVERHEAD, PrefixFamily
from common.inputs import inputs
from common.rate_distortion import at_quality


def test_read_guard_denies_test_and_preserves_zero_count():
    reader, _ = inputs()
    with pytest.raises(PermissionError, match="sealed"):
        reader.read("test", "any", 0)
    assert reader.counts["test"] == 0


@pytest.mark.parametrize("intra", [False, True])
def test_actual_byte_ledger_includes_base_native_payload_and_record_headers(monkeypatch, intra):
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: SimpleNamespace(synchronize=lambda: None))
    tag = b"ledger01"
    shape = (8, 8, 8)
    bank = BatchRecords((0.8, 0.5), Segmentation(8, 8))
    base = BaseFrame(FrameInfo(0, 1080, 1920, 0, intra, 1, False), b"physical native y+z")
    coarse = torch.ones((1, *shape), dtype=torch.int8)
    fine = torch.zeros_like(coarse)
    full = bank.family(base, tag, coarse, fine, 20000)
    pixels = torch.zeros((1, 3, 1080, 1920))
    codec = SimpleNamespace(receiver=lambda decoder: BatchReceiver(
        tag, lambda record: Display(pixels, bank.canvas(shape, intra), torch.ones(8, 1, 1), lambda delta: pixels)))
    decoder = SimpleNamespace(order=SimpleNamespace(desync=False))
    result, row = receive(codec, decoder, full, len(full.raw))
    assert result is not None
    assert row["bytes"] == row["full_max_bytes"] == len(full.raw)
    assert row["base_bytes"] == full.base_end
    assert row["base_header_bytes"] == APP_BASE_OVERHEAD
    assert row["native_joint_y_z_bytes"] == len(base.payload)
    assert row["enhancement_headers"] == RECORD_OVERHEAD
    assert row["bytes"] == sum(row[key] for key in ("native_joint_y_z_bytes", "base_header_bytes",
                                                    "enhancement_headers", "enhancement_payload"))
    for cap in (full.base_end, len(full.raw)):
        assert full.prefix(cap)[0] == full.raw[:cap]
    bad, overflow = receive(codec, decoder, PrefixFamily(full.raw, full.ends), full.base_end - 1)
    assert bad is None and overflow["status"] == "BASE_BUDGET_EXCEEDED" and decoder.order.desync


def test_equal_quality_interpolates_in_log_rate_and_never_extrapolates():
    assert at_quality([(25, 1), (27, 4)], "global_psnr", 26) == pytest.approx(2)
    assert at_quality([(25, 1), (27, 4)], "global_psnr", 28) is None


def test_end_to_end_clock_includes_full_max_generation(monkeypatch):
    from baselines.hla_tok_uf_v2 import evaluate

    clock = [0.0]
    caps = []

    def fake_encode(_codec, _rgb, _index, cap):
        caps.append(cap)
        clock[0] += 0.025
        return SimpleNamespace(family=object(), phase_ms={"encode_ms": 25})

    def fake_receive(*_args):
        clock[0] += 0.005
        return None, {"status": "PASS", "decode_ms": 5, "base_ready_ms": 2}

    monkeypatch.setattr(evaluate, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(evaluate, "source_frame", lambda *_: torch.zeros(1))
    monkeypatch.setattr(evaluate, "encode", fake_encode)
    monkeypatch.setattr(evaluate, "receive", fake_receive)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    codec = SimpleNamespace(reset=lambda: None, new_decoder=lambda: SimpleNamespace(reset=lambda: None))
    clip = SimpleNamespace(video="clip", length=5)
    rows = evaluate.timing(codec, None, None, [clip], [5], 1, 5, False)
    assert set(caps) == {evaluate.MAX_APPLICATION_BYTES}
    assert len(rows) == 5
    assert [row["total_ms"] for row in rows] == pytest.approx([30] * 5)


def test_reported_lpips_target_uses_the_archived_endpoint_without_extrapolation():
    from common.rate_distortion import archive_target

    original = [{"lpips": 0.2897786209466814}, {"lpips": 0.221}]
    query = archive_target("lpips", 0.28978, original)
    assert query == original[0]["lpips"]
    assert at_quality([(-query, 1.395), (-0.221, 2.4)], "lpips", query) == pytest.approx(1.395)
    assert archive_target("lpips", 0.28978, [{"lpips": .28}]) == 0.28978
