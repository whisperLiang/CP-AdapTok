import pytest
import torch

from aip_vtok.evaluation.fix import HORIZON_VIDEO
from aip_vtok.joint.evaluation import JOINT, aggregate, joint_rows, transmitted_tokens
from aip_vtok.joint.protocol import CONFIG, check_frame_isolation, open_experiment
from aip_vtok.joint.scan import BUDGETS
from aip_vtok.joint.trainer import History


def test_summary_distinguishes_global_and_frame_psnr_and_charges_reference_floor():
    base = dict(method=JOINT, history="CLOSED", budget=510, video=HORIZON_VIDEO, rate_basis="actual",
                token_bits=16, actual_bpp=0.1, total_bytes=100, peak_gpu_mib=1,
                total_encode_ms=1.0, total_decode_ms=2.0, end_to_end_ms=3.0, lpips=0.1, dists=0.2)
    report, = aggregate([base | {"frame": 50, "mse": 0.01, "psnr": 20.0},
                         base | {"frame": 51, "mse": 0.1, "psnr": 10.0}])
    assert report["global_psnr"] == pytest.approx(12.59637)
    assert report["mean_frame_psnr"] == 15
    assert report["late_lpips"] == 0.1
    assert report["total_bytes"] == 200
    assert transmitted_tokens(510, True) == 2048
    assert transmitted_tokens(510, False) == 510


def test_exact_duplicate_across_splits_is_fatal():
    row = {"frame_hashes": [{"sha256": "ab"}]}
    with pytest.raises(ValueError, match="multiple splits"):
        check_frame_isolation({"train": [row], "test": [row]})


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="native CUDA required")
def test_native_64_frame_closed_rollout(tmp_path):
    run = open_experiment(CONFIG, 1)
    with pytest.raises(PermissionError, match="sealed"):
        run.data.catalog("test")
    codec = run.codec.eval()
    profile = codec.profile()
    clip = run.data.read("validation", HORIZON_VIDEO, 0, 64)
    history = History.at(0)
    with torch.no_grad():
        for i, frame in enumerate(clip):
            rgb = frame[None].cuda().float() / 255
            if i in (0, 31, 32, 63):
                measurements = list(joint_rows(codec, profile, history, rgb, i, {}, None))
                assert [r["budget"] for r in measurements] == list(BUDGETS)
                assert measurements[0]["transmitted_tokens"] == (2048 if i % 32 == 0 else 510)
                assert measurements[0]["tx_reference_sha256"] == measurements[0]["rx_reference_sha256"]
            else:
                from aip_vtok.joint.bitstream import FRAME_HEADER

                frame = codec.encode_frame(rgb, history.tx, i, profile)
                a = codec.begin_frame(history.tx, frame.wire.data[:FRAME_HEADER.size], profile)
                b = codec.begin_frame(history.rx, frame.wire.data[:FRAME_HEADER.size], profile)
                a.decode_prefix(frame.wire.prefix(a.reference_budget))
                b.decode_prefix(frame.wire.prefix(b.reference_budget))
                history.submit(a.commit_reference(), b.commit_reference())
    assert history.rx.frame_id == 63
    assert torch.isfinite(history.rx.latent).all()
    run.verify_integrity()
    assert run.data.counts == {"train": 0, "validation": 64, "test": 0}


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="native CUDA required")
def test_one_native_frame_full_metrics_and_fixed_controls(tmp_path, monkeypatch):
    from aip_vtok.joint import evaluation

    run = open_experiment(CONFIG, 1)
    monkeypatch.setattr(evaluation, "cohort", lambda *args: {HORIZON_VIDEO: 1})
    rows = evaluation.evaluate(run, "validation", tmp_path, fixed=True, compare=True, archive=True)
    assert len(rows) == (8 + 3 * 5) * 2
    assert set(r["method"] for r in rows) == {"Joint V1", "B2", "W1", "SLD-500"}
    assert all(r["mse"] > 0 and r["lpips"] >= 0 for r in rows)
    assert len(list((tmp_path / "streams").rglob("*.aipj"))) == 2
    assert run.data.counts["test"] == 0
