import pytest
import torch

from hla_tok import BUDGETS, Geometry, HLATok
from hla_tok.streaming.fixed13 import pack_tokens_13bit, unpack_tokens_13bit


@pytest.mark.slow
def test_native_cpu_has_real_rgb_output_and_only_received_bytes():
    old_threads = torch.get_num_threads()
    torch.set_num_threads(4)
    try:
        torch.manual_seed(7)
        sender, receiver = HLATok().eval(), HLATok().eval()
        receiver.load_state_dict(sender.state_dict())
        geometry = Geometry(1080, 1920)
        frame = torch.rand(1, 3, 1080, 1920)
        with torch.inference_mode():
            ids = sender.encode_frame(frame, BUDGETS[0])
            wire = pack_tokens_13bit(ids[0].tolist())
            received = torch.tensor([unpack_tokens_13bit(wire, BUDGETS[0])])
            decoded = receiver.decode_prefix(received, geometry)
            assert decoded.shape == frame.shape
            assert torch.isfinite(decoded).all()
            assert 0 <= decoded.min() < decoded.max() <= 1
    finally:
        torch.set_num_threads(old_threads)


@pytest.mark.gpu
@pytest.mark.parametrize("amp", [False, True], ids=["fp32", "bf16"])
def test_native_gpu_strict_integer_prefixes_all_budgets(amp):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if amp and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA BF16 is unavailable")
    torch.manual_seed(8)
    model = HLATok().eval().to("cuda", memory_format=torch.channels_last)
    frame = torch.rand(1, 3, 1080, 1920, device="cuda")
    geometry = Geometry(1080, 1920)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        complete = model.encode_frame(frame)
        assert complete.shape == (1, geometry.capacity)
        ctx = model.prepare_frame(frame)
        for budget in BUDGETS:
            appended = model.extend_prefix(ctx, budget)
            fresh = model.encode_frame(frame, budget)
            assert torch.equal(appended, complete[:, :budget])
            assert torch.equal(fresh, complete[:, :budget])
        for budget in (4096, 16320, 65280):
            decoded = model.decode_prefix(complete[:, :budget], geometry)
            assert decoded.shape == frame.shape
            assert torch.isfinite(decoded).all()


@pytest.mark.gpu
def test_native_bf16_gradient_all_modules_and_training_ids():
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA BF16 is unavailable")
    torch.manual_seed(9)
    model = HLATok().to("cuda", memory_format=torch.channels_last)
    frame = torch.rand(1, 3, 1080, 1920, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        encoded = model.encode_training(frame, 16320)
        reconstructed = model.decode_training(encoded, 16320)
        loss = (reconstructed - frame).abs().mean() + encoded.codebook_loss + 0.25 * encoded.commitment_loss
    loss.backward()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        assert torch.equal(encoded.ids, model.encode_frame(frame, 16320))
    for module in (model.encoder, model.query_encoder, model.quantizer, model.local_decoder, model.rgb_decoder):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm() > 0
                   for p in module.parameters())
