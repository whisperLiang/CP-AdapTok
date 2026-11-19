import torch
from torch import nn

from aip_vtok.backbones.latent_to_ref import LatentToRef
from aip_vtok.decoder.causal_state import FrameContext, ReferenceState
from aip_vtok.models.codec import AIPVTokCodec
from aip_vtok.training.prefix import prefix_canvas


class HistoryOnlyBackbone(nn.Module):
    def begin_frame(self, state, frame_id):
        return FrameContext(frame_id, state.is_intra(frame_id, 32), None, None)


@torch.no_grad()
def test_append_only_updates_new_values_without_assembling_full_reference_canvas(monkeypatch):
    codec = AIPVTokCodec(HistoryOnlyBackbone(), LatentToRef(16), reference_i=15, reference_p=8)
    prediction = torch.zeros(1, 128, 8, 16)
    tokens = codec.tokenizer.encode(torch.randn_like(prediction), prediction)
    expected = prefix_canvas(codec.tokenizer, tokens, prediction, 15, differentiable=False).latent()
    session = codec.begin_frame(0, tokens.tiers, ReferenceState())

    def forbidden_full_canvas_read():
        raise AssertionError("append must not assemble an entire latent canvas")

    monkeypatch.setattr(session.canvas, "latent", forbidden_full_canvas_read)
    for offset in range(0, tokens.ids.shape[1], 7):
        session.append(tokens.ids[:, offset : min(offset + 7, tokens.ids.shape[1])], offset)
    reference = session.commit_reference_once()
    assert torch.equal(reference.latent, expected)
    assert session.canvas.projected_tokens == tokens.ids.shape[1]
    assert session.rgb_decoder_calls == 0
