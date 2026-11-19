"""An integer prefix crosses the encoder/receiver boundary before reference commit."""

from dataclasses import dataclass, field

import torch

from aip_vtok.c1.frames import Sample, capture
from aip_vtok.c1.recovery_net import commit_reference
from aip_vtok.c1.scan import REF_BUDGET
from aip_vtok.c1_followup.dense import Layout
from aip_vtok.decoder.causal_state import FrameContext, ReferenceState
from aip_vtok.diagnostics.diag0.closed import same_state

from .scan import Plan


@dataclass(frozen=True)
class FrameCode:
    layout: Layout
    plan: Plan
    rates: dict

    def encode(self, sample):
        ids, _ = self.layout.encode(sample)
        return self.plan.encode(ids)


@dataclass(frozen=True)
class LinkFrame:
    sample: Sample
    context: FrameContext
    payload: torch.Tensor


@dataclass
class CausalPair:
    tx: ReferenceState = field(default_factory=ReferenceState)
    rx: ReferenceState = field(default_factory=ReferenceState)

    @torch.no_grad()
    def commit(self, tx_context, tx_latent, tx_feature, rx_context, rx_latent, rx_feature):
        tx = commit_reference(tx_context, tx_latent, tx_feature)
        rx = commit_reference(rx_context, rx_latent, rx_feature)
        if not same_state(tx, rx):
            raise RuntimeError(f"sender and receiver references differ at frame {tx.frame_id}")
        self.tx, self.rx = tx, rx

    @torch.no_grad()
    def advance(self, codec, image, frame_id, cases, receiver):
        sample = capture(codec.backbone, self.tx, frame_id, image)
        context = codec.backbone.begin_frame(self.rx, frame_id)
        case = cases[sample.frame_type]
        payload = case.encode(sample)
        ref_payload = payload[:, :REF_BUDGET[context.intra]]
        tx = receiver(case, codec.backbone, sample.context, ref_payload)
        rx = receiver(case, codec.backbone, context, ref_payload)
        self.commit(sample.context, tx, codec.projector(tx, context.intra),
                    context, rx, codec.projector(rx, context.intra))
        return LinkFrame(sample, context, payload)
