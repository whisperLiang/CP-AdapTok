"""Receiver-only reconstruction with independently committed reference states."""

from dataclasses import dataclass, field

import torch

from aip_vtok.c1.frames import capture, frame_fields
from aip_vtok.c1.metrics import FrameMetrics
from aip_vtok.c1.recovery_net import commit_reference
from aip_vtok.decoder.causal_state import ReferenceState
from aip_vtok.diagnostics.diag0.closed import same_state

from .dense import chroma_planes, frame_key, quality_summary
from .gates import HISTORIES, METHODS


def recover(layout, backbone, payload, model, context):
    base = layout.decode(payload, backbone)
    if model is None:
        return base
    shape = (len(base), layout.groups, *base.shape[-2:])
    mask = torch.ones(shape, device=base.device, dtype=torch.bool)
    return model(base, mask, layout.indices, context)


def reconstruct_pair(codec, tx_context, rx_context, layout, payload, model):
    tx = recover(layout, codec.backbone, payload, model, tx_context)
    rx = recover(layout, codec.backbone, payload, model, rx_context)
    return tx, rx


@dataclass
class CausalPair:
    tx: ReferenceState = field(default_factory=ReferenceState)
    rx: ReferenceState = field(default_factory=ReferenceState)

    def advance(self, codec, image, frame_id, layouts, model=None):
        sample = capture(codec.backbone, self.tx, frame_id, image)
        tx_context = sample.context
        rx_context = codec.backbone.begin_frame(self.rx, frame_id)
        layout = layouts[sample.frame_type]
        payload, _ = layout.encode(sample)
        tx, rx = reconstruct_pair(codec, tx_context, rx_context, layout, payload, model)
        self.tx = commit_reference(tx_context, tx, codec.projector(tx, tx_context.intra))
        self.rx = commit_reference(rx_context, rx, codec.projector(rx, rx_context.intra))
        return sample, rx_context, payload, rx, same_state(self.tx, self.rx)


def scored_row(backbone, sample, rx, latent, sync, scorer, video, history, method):
    rgb, _ = backbone.decode_aligned(latent, rx)
    return {**frame_fields(sample, video, history, "validation"), "method": method, "state_sync": sync,
            **scorer(rgb, latent, sample.aligned), **chroma_planes(sample.rgb, rgb)}


def paired_rows(codec, result, layouts, model, scorer, video):
    sample, rx, payload, quantized, sync = result
    backbone = codec.backbone
    kind = sample.frame_type
    history = HISTORIES[0]
    continuous, quant = (layouts[name][kind] for name in ("continuous", "quantized"))
    cont_payload, _ = continuous.encode(sample)
    b0_cont = recover(continuous, backbone, cont_payload, None, rx)
    yield scored_row(backbone, sample, rx, b0_cont, sync, scorer, video, history, METHODS[0])
    yield scored_row(backbone, sample, rx, quantized, sync, scorer, video, history, METHODS[1])
    tx_latent, rx_latent = reconstruct_pair(codec, sample.context, rx, quant, payload, model)
    yield scored_row(backbone, sample, rx, rx_latent, sync and torch.equal(tx_latent, rx_latent),
                     scorer, video, history, METHODS[2])


@torch.no_grad()
def dense_rollout(codec, clip, video, layouts, model, suite):
    quantized, continuous, trained = CausalPair(), CausalPair(), CausalPair()
    for frame, image in enumerate(clip):
        reference = quantized.advance(codec, image, frame, layouts["quantized"])
        scorer = FrameMetrics(suite, reference[0].rgb)
        paired = tuple(paired_rows(codec, reference, layouts, model, scorer, video))
        yield from paired
        yield paired[1] | {"history": HISTORIES[1]}
        for link, method, variants, recovery in (
            (continuous, METHODS[0], layouts["continuous"], None),
            (trained, METHODS[2], layouts["quantized"], model),
        ):
            sample, rx, _, latent, sync = link.advance(codec, image, frame, variants, recovery)
            yield scored_row(codec.backbone, sample, rx, latent, sync, scorer, video, HISTORIES[1], method)


def summarize_history(rows, expected):
    table = {}
    for method in METHODS:
        group = [row for row in rows if row["method"] == method]
        actual = {frame_key(row)[:2] for row in group}
        if actual != expected or len(group) != len(expected):
            raise RuntimeError("dense RecoveryNet gate requires every unique frame of the Validation cohort")
        table[method] = quality_summary(group)["all"]
    table["state_sync"] = all(row["state_sync"] for row in rows)
    return table


def summarize_rollout(rows, cohort):
    expected = {(video, index) for video, count in cohort.items() for index in range(count)}
    return {history: summarize_history([row for row in rows if row["history"] == history], expected)
            for history in HISTORIES}
