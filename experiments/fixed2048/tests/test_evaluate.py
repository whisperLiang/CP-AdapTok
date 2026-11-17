import json

import pytest

from experiments.fixed2048.checkpoint import sha256
from experiments.fixed2048.evaluate import check_test_seal, stage_times


def test_test_split_is_closed_until_identities_and_config_are_sealed(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"trained")
    manifest = tmp_path / "cpat_test.json"
    manifest.write_text("{}")
    config = {"manifest_dir": str(tmp_path), "final_test": {"frames": 64}}
    with pytest.raises(ValueError, match="seal"):
        check_test_seal(config, checkpoint, None)
    seal = tmp_path / "seal.json"
    seal.write_text(
        json.dumps(
            {
                "checkpoint_sha256s": [sha256(checkpoint)],
                "test_config": config["final_test"],
                "test_manifest_sha256": sha256(manifest),
            }
        )
    )
    check_test_seal(config, checkpoint, seal)
    checkpoint.write_bytes(b"changed weights")
    with pytest.raises(ValueError, match="identity"):
        check_test_seal(config, checkpoint, seal)


def test_r0_latency_does_not_count_unrelated_paired_tokenization():
    from types import SimpleNamespace

    encoded = SimpleNamespace(timings={"encoder_ms": 12, "tokenizer_ms": 7})
    output = SimpleNamespace(timings={"dcvc_decoder_ms": 9})
    assert stage_times(encoded, output, "R0")["end_to_end_ms"] == 21
    assert stage_times(encoded, output, "R1")["end_to_end_ms"] == 28


def test_pair_measurement_is_independent_of_candidate_order(tmp_path):
    import torch

    from experiments.fixed2048.diagnostics import psnr
    from experiments.fixed2048.evaluate import evaluate_loop
    from experiments.fixed2048.tests.test_codec import make_model

    torch.set_num_threads(2)

    def suite(original, decoded):
        mse = torch.nn.functional.mse_loss(original, decoded).item()
        return {"mse": mse, "psnr": psnr(mse)}

    model = make_model().eval()
    rows, pairs = evaluate_loop(
        model,
        {"content_id": "validation:tiny", "path": "tiny"},
        torch.rand(2, 3, 64, 64),
        ["R2", "R0", "R1"],
        "paired",
        suite,
        tmp_path,
    )
    by_path = {row["path"]: row for row in rows if row["frame"] == 0}
    expected = by_path["R2"]["psnr"] - by_path["R1"]["psnr"]
    assert pairs[0]["r2_minus_r1_psnr"] == pytest.approx(expected)


def test_sealed_test_protocol_rejects_changed_frames_or_modes(tmp_path):
    from experiments.fixed2048.evaluate import validate_test_request

    plan = {
        "checkpoint_sha256s": ["digest"],
        "protocols": {"digest": {"paths": ["R1"], "references": ["full2048"]}},
    }
    config = {"final_test": {"videos": ["cows", "goat", "parkour"], "frames": 64}}
    validate_test_request(config, plan, "digest", ["R1"], ["full2048"], ["cows", "goat", "parkour"], 64)
    with pytest.raises(ValueError, match="protocol"):
        validate_test_request(config, plan, "digest", ["R1"], ["full2048"], ["cows"], 8)
    with pytest.raises(ValueError, match="protocol"):
        validate_test_request(config, plan, "digest", ["R2"], ["full2048"], ["cows", "goat", "parkour"], 64)


def test_original_reference_reproduces_historical_i_period_32(tmp_path):
    import torch

    from experiments.fixed2048.diagnostics import psnr
    from experiments.fixed2048.evaluate import evaluate_loop
    from experiments.fixed2048.tests.test_codec import make_model

    torch.set_num_threads(2)

    def suite(original, decoded):
        mse = torch.nn.functional.mse_loss(original, decoded).item()
        return {"mse": mse, "psnr": psnr(mse)}

    rows, _ = evaluate_loop(
        make_model().eval(),
        {"content_id": "validation:tiny", "path": "tiny"},
        torch.full((33, 3, 64, 64), 0.5),
        ["R1"],
        "original128",
        suite,
        tmp_path,
    )
    assert [r["frame"] for r in rows if r["frame_type"] == "I"] == [0, 32]
