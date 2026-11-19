"""Matched frame scoring and audited reference/payload identifiers."""

import csv
import hashlib
from itertools import product

import torch

from aip_vtok.c1.dense_eval import validation_videos
from aip_vtok.c1.scan import REF_BUDGET
from aip_vtok.evaluation.fix import HORIZON_VIDEO, rms
from aip_vtok.sparse_rfsq.evaluation import CLOSED, FIXED, FrameScorer, check_finite
from aip_vtok.sparse_rfsq.protocol import COHORT_FRAMES, HORIZON

__all__ = ["CLOSED", "FIXED", "FrameScorer"]


def tensor_sha256(tensor):
    return hashlib.sha256(tensor.detach().contiguous().view(torch.uint8).cpu().numpy()).hexdigest()


def validation(run):
    run.reader.validation_locked = True
    catalog = run.reader.open("validation")
    counts = validation_videos(run.config, catalog)
    if sum(counts.values()) != COHORT_FRAMES or counts[HORIZON_VIDEO] != HORIZON:
        raise RuntimeError("Validation requires the sealed 174 frames with the 64-frame horizon")
    return catalog, counts


def frame_fields(catalog, video, frame, pair, reference_arm):
    index = frame.context.frame_id
    reference_budget = REF_BUDGET[frame.context.intra]
    if pair.tx.frame_id != index or pair.rx.frame_id != index or not frame.context.committed:
        raise RuntimeError("display did not get a singly committed synchronized frame context")
    fields = {"video": video, "frame": index, "frame_type": frame.sample.frame_type,
              "input_rgb_sha256": catalog.rows[video]["frame_hashes"][index]["sha256"],
              "aligned_shape": list(frame.sample.aligned.shape),
              "aligned_sha256": tensor_sha256(frame.sample.aligned),
              "reference_arm": reference_arm,
              "reference_input_frame": None if frame.context.intra else index - 1,
              "reference_state_id": f"{video}/{reference_arm}/{index - 1}" if not frame.context.intra else "I-reset",
              "committed_state_id": f"{video}/{reference_arm}/{index}",
              "reference_budget": reference_budget, "state_sync": True,
              "reference_payload_sha256": tensor_sha256(frame.payload[:, :reference_budget]),
              "reference_latent_rms": rms(pair.rx.latent), "projected_feature_rms": rms(pair.rx.feature)}
    if video == HORIZON_VIDEO and index in (31, 63):
        fields.update(tx_reference_sha256=pair.tx.fingerprint(), rx_reference_sha256=pair.rx.fingerprint())
    return fields


def display_rows(scorer, backbone, frame, case, payload, receiver, fields, budgets, visualize=None):
    cache = scorer.context_cache(frame)
    for k in budgets:
        prefix = payload[:, :k]
        latent = receiver(case, backbone, frame.context, prefix)
        rate = case.rates[k]
        row = fields | {"budget": k, "symbols_received": prefix.shape[1], "prefix_ids_match": True,
                        "payload_sha256": tensor_sha256(prefix)} | rate
        if fields["history"] == CLOSED:
            combined = case.rates[max(k, fields["reference_budget"])]
            row.update(union_payload_bits=combined["payload_bits"], union_header_bytes=combined["header_bytes"],
                       union_entropy_payload_bits=combined["entropy_payload_bits"])
        row.update(scorer.measure(frame, latent, cache[k]))
        check_finite(row.values())
        if visualize is not None:
            visualize(frame, latent, row)
        yield row


def validate_rows(rows, counts, arms):
    frames = [(video, index) for video, count in counts.items() for index in range(count)]
    expected = {(*arm, *frame) for arm, frame in product(arms, frames)}
    keys = ("arm", "history", "budget", "video", "frame")
    actual = {tuple(row[k] for k in keys) for row in rows}
    if len(rows) != len(expected) or actual != expected:
        raise RuntimeError("missing, extra, or duplicate evaluation rows")
    if not all(r["state_sync"] and r["prefix_ids_match"] for r in rows):
        raise RuntimeError("reference synchronization or integer prefix failed")


def scopes(rows):
    horizon = [r for r in rows if r["video"] == HORIZON_VIDEO]
    return {"validation": rows, "horizon": horizon, "late": [r for r in horizon if 48 <= r["frame"] < HORIZON]}


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
