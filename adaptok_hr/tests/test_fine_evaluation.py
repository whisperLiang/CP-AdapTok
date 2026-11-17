import torch


def test_independent_receiver_can_be_created_during_inference(small_model):
    from experiments.adaptok_hr.fine_recovery.common import independent_codec

    small_model.eval()
    with torch.inference_mode():
        receiver = independent_codec(small_model, amp_bf16=False)
        receiver.check_model()
    assert all(a is not b for a, b in zip(small_model.parameters(), receiver.model.parameters()))
