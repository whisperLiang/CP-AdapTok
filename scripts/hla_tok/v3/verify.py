"""Functional acceptance for real R2 weights on the frozen native Validation."""

from copy import deepcopy

import torch

from hla_tok import BUDGETS
from hla_tok.evaluation.benchmark import transfer
from hla_tok.evaluation.io import write_json
from hla_tok.models.anchors import anchor_digest, anchor_table
from hla_tok.models.local import RANK_CHUNK
from hla_tok.streaming.fixed13 import pack_tokens_13bit, payload_ledger, unpack_tokens_13bit
from scripts.hla_tok.v3.benchmark import native_references
from scripts.hla_tok.v3.protocol import NATIVE


def _exact(left, right, label):
    if not torch.equal(left, right):
        raise RuntimeError(f"strict prefix or receiver test failed: {label}")


def _count_set():
    regions = NATIVE.num_regions
    anchors = (regions, 2 * regions, regions * RANK_CHUNK, 2 * regions * RANK_CHUNK,
               3 * regions * RANK_CHUNK, 4 * regions * RANK_CHUNK, NATIVE.capacity - 4)
    return sorted(set(BUDGETS) | {anchor + offset for anchor in anchors for offset in range(-3, 4)} | {1})


def _prefixes(model, source, receiver, counts):
    context = model.prepare_frame(source)
    complete = model.encode_frame(source)
    for count in counts:
        prefix = complete[:, :count]
        _exact(prefix, model.extend_prefix(context, count), f"extend K{count}")
        _exact(prefix, model.encode_frame(source, count), f"fresh K{count}")
        wire = pack_tokens_13bit(prefix[0].cpu().tolist())
        recovered = torch.tensor([unpack_tokens_13bit(wire, count)], device=prefix.device)
        _exact(prefix, recovered, f"wire K{count}")
        if len(wire) != (13 * count + 7) // 8:
            raise RuntimeError(f"unexpected Fixed13 length at K{count}")
        if count in BUDGETS:
            expected = model.decode_prefix(prefix, NATIVE)
            actual = receiver.decode_bytes(wire, count, NATIVE)
            _exact(expected, actual, f"standalone K{count}")
            if actual.shape != source.shape or not actual.isfinite().all():
                raise RuntimeError("independent Receiver did not produce finite native RGB")
    return complete


def _receiver_boundaries(model, receiver, source, future, full, working_budget):
    original = full[:, :working_budget].clone()
    other = full.clone()
    other[:, working_budget:] = (other[:, working_budget:] + 1) % 8192
    image = receiver.decode_prefix(original, NATIVE)
    _exact(image, receiver.decode_prefix(other[:, :working_budget], NATIVE), "suffix backing storage")
    for frame in (future, 1.0 - future, future):
        model.encode_frame(frame, working_budget)
        _exact(original, model.encode_frame(source, working_budget), "future-frame independence")
        _exact(image, receiver.decode_prefix(original, NATIVE), "cross-frame Receiver independence")
    tail = pack_tokens_13bit([19])
    for payload, count in ((tail[:-1], 1), (bytes([tail[0], tail[1] | 0x80]), 1), (b"", 0),
                           (b"", NATIVE.capacity + 1), (tail, True)):
        try:
            receiver.decode_bytes(payload, count, NATIVE)
        except ValueError:
            continue
        raise RuntimeError("independent Receiver accepted an invalid Fixed13 stream")
    return payload_ledger(len(pack_tokens_13bit(original[0].cpu().tolist())), working_budget, 1,
                          NATIVE.height, NATIVE.width)


@torch.inference_mode()
def verify_native(model, reader, cohort, working_budget, output):
    model.eval()
    receiver = deepcopy(model.receiver()).eval()
    if set(dict(receiver.named_children())) != {"quantizer", "local_decoder", "rgb_decoder"}:
        raise RuntimeError("the independent Receiver includes analysis modules")
    expected_anchor = anchor_digest(anchor_table())
    if model.architecture_metadata()["anchor_table_sha256"] != expected_anchor:
        raise RuntimeError("received weights use an incompatible R2 anchor table")
    before = dict(reader.counts)
    counts = _count_set()
    rows = {}
    try:
        source, future = [transfer(reader.read("validation", *ref)[None]) for ref in native_references(cohort, 2)]
        for precision in ("fp32", "bf16"):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
                full = _prefixes(model, source, receiver, counts)
                rate = _receiver_boundaries(model, receiver, source, future, full, working_budget)
            rows[precision] = {"strict_budgets": counts, "nonpresets": len(set(counts) - set(BUDGETS)),
                               "rate": rate}
        result = {"status": "PASS", "precisions": rows, "anchor_sha256": expected_anchor,
                  "tests": {key: "PASS" for key in ("T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9",
                                                   "T10", "T11", "T12", "T13", "T14", "T19", "T20")}
                  | {"T15": "NOT_APPLICABLE", "T16": "NOT_APPLICABLE"}}
        write_json(output / "native_verify.json", result)
        return result
    finally:
        write_json(output / "pixel_reads.json", {key: reader.counts[key] - before[key] for key in before})
