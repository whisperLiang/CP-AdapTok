"""An independent receiver owns ordinary immutable model tensors, even during inference."""

import copy

import torch

from adaptok_hr.codec.codec import Codec


def independent_codec(model, amp_bf16=True):
    with torch.inference_mode(False):
        receiver = copy.deepcopy(model).eval()
        return Codec(receiver, amp_bf16=amp_bf16)
