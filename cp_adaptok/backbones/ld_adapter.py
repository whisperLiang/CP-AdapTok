"""Pure access to pinned official LD modules; never calls its mutable DPB methods."""

import torch
from torch import nn

from third_party.DCVC.src.models.video_model_ld import DMC
from third_party.DCVC.src.utils.common import get_state_dict


class CausalLDBackboneAdapter(nn.Module):
    latent_channels = 128

    def __init__(self, checkpoint=None):
        super().__init__()
        self.model = DMC()
        if checkpoint:
            self.model.load_state_dict(get_state_dict(checkpoint), strict=True)

    def begin_frame(self, state):
        if state.feature is None:
            raise ValueError("P frame requires a decoded base reference")
        if state.memory is None:
            memory = self.model.feature_adaptor_i(state.feature)
        else:
            memory = self.model.feature_adaptor_m(state.memory, state.feature)
        return self.model.feature_extractor(memory), memory

    def quant(self, name, qp):
        parameter = getattr(self.model, name)
        index = torch.tensor([qp], device=parameter.device, dtype=torch.long)
        return self.model.index_select_dim0(parameter, index)

    def encode_latent(self, x_t, frozen_context):
        return self.model.encoder(x_t, frozen_context.context, self.quant("q_encoder", frozen_context.qp))

    def decode_latent(self, y_hat, frozen_context):
        feature = self.model.decoder(
            y_hat, frozen_context.context, self.quant("q_decoder", frozen_context.qp)
        )
        return self.model.recon_head(feature), feature

    @staticmethod
    def commit_base_state(feature, frozen_context, base_digest):
        return frozen_context.commit(feature, base_digest)
