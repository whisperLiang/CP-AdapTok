"""No-training channel, FSQ, space, and 2×2 scan attribution."""

from collections import defaultdict
from itertools import product

import torch

from aip_vtok.c1.scan import BUDGETS, DENSE, GROUPS, SHAPE
from aip_vtok.c1_followup.dense import quality_summary
from aip_vtok.diagnostics.diag0.io import json_at
from aip_vtok.runtime import file_hash, write_json
from aip_vtok.sparse_rfsq.calibration import codes
from aip_vtok.sparse_rfsq.evaluation import check_finite
from aip_vtok.sparse_rfsq.receiver import Receiver
from aip_vtok.sparse_rfsq.rollout import CausalPair

from .evaluation import (
    CLOSED,
    FIXED,
    FrameScorer,
    display_rows,
    frame_fields,
    scopes,
    tensor_sha256,
    validate_rows,
    validation,
    write_csv,
)
from .protocol import HISTORICAL_DENSE, OUTPUT, PROVENANCE, SIGMA

FACTORS = {"Q1": ("A1", "C1"), "Q2": ("A1", "SP-1"), "Q3": ("A2", "C1"), "Q4": ("A2", "SP-1")}
DENSE_ARMS = {"O0": None, "O1": None, "O2": DENSE}
METRICS = ("psnr_global_mse", "psnr_frame_mean", "lpips", "dists")
RESULTS = ("phase0_diagnostics.json", "phase0_factorial.json", "phase0_per_frame.csv",
           "phase0_k510.png", "phase0_report.md")
DONE = OUTPUT / "phase0_complete.json"


def factorial_cases(run):
    lock = json_at(HISTORICAL_DENSE.parent.parent / "c1/scan_lock.json")
    area = SHAPE[0] * SHAPE[1]
    slots = [(1, slot // area, *divmod(slot % area, SHAPE[1])) for slot in lock["slots"]]
    scans = {"C1": dict.fromkeys(("I", "P"), slots), "SP-1": run.lock["keys"]["SP-1"]}
    for arm in ("A1", "A2"):
        for kind, layout in run.inputs.layouts[arm].items():
            a1 = run.inputs.layouts["A1"][kind]
            if (len(layout.quantizer.stages) != 1 or layout.space != "aligned" or layout.channels != 128
                    or layout.indices.numel() != GROUPS * 6 or not torch.equal(layout.indices, a1.indices)):
                raise RuntimeError("2×2 requires the same aligned 24 channels and single-level FSQ")
    if any(len(keys) != BUDGETS[-1] or any(key[0] != 1 for key in keys)
           for scan in scans.values() for keys in scan.values()):
        raise RuntimeError("2×2 requires the two frozen single-level scans")
    return {q: codes(scans[scan], run.inputs.layouts[arm], run.inputs.entropy[arm])
            for q, (arm, scan) in FACTORS.items()}


def dense_rows(backbone, frame, scorer, layout, fields):
    sample = frame.sample
    ids, _ = layout.encode(sample)
    if len(ids) != 1 or ids[0].shape != (1, GROUPS, *SHAPE) or ids[0].numel() != DENSE:
        raise RuntimeError("dense FSQ diagnostic must decode one full level at every group and site")
    values = {
        "O0": sample.aligned,
        "O1": layout.restore(layout.select(sample), backbone),
        "O2": layout.decode(ids, backbone),
    }
    cache = scorer.context_cache(frame)
    for arm, latent in values.items():
        row = fields | {"arm": arm, "history": FIXED, "budget": DENSE_ARMS[arm], "prefix_ids_match": True,
                        "symbols_received": DENSE if arm == "O2" else 0}
        if arm == "O2":
            row["payload_sha256"] = tensor_sha256(ids[0])
        row.update(scorer.measure(frame, latent, cache["dense"]))
        check_finite(row.values())
        yield row


@torch.no_grad()
def diagnose_video(run, catalog, video, clip, cases):
    gaussian = Receiver(SIGMA)
    learned = Receiver(SIGMA, run.w1)
    pairs = {q: CausalPair() for q in FACTORS}
    backbone = run.codec.backbone
    for index, image in enumerate(clip):
        fixed = pairs["Q1"].advance(run.codec, image, index, cases["Q1"], gaussian)
        common = frame_fields(catalog, video, fixed, pairs["Q1"], "Q1")
        scorer = FrameScorer(run.codec, run.suite, fixed.sample.rgb)
        kind = fixed.sample.frame_type
        a2_sp1 = cases["Q4"][kind]
        sp1_packet = a2_sp1.encode(fixed.sample)
        yield from dense_rows(backbone, fixed, scorer, a2_sp1.layout, common)
        for arm, receiver in (("O3", gaussian), ("O4", learned)):
            fields = common | {"arm": arm, "history": FIXED, "quantizer": "A2", "scan": "SP-1"}
            yield from display_rows(scorer, backbone, fixed, a2_sp1, sp1_packet, receiver, fields, BUDGETS)
        for q, (quantizer, scan) in FACTORS.items():
            case = cases[q][kind]
            packet = {"Q1": fixed.payload, "Q4": sp1_packet}.get(q)
            if packet is None:
                packet = case.encode(fixed.sample)
            fields = {"arm": q, "quantizer": quantizer, "scan": scan}
            yield from display_rows(scorer, backbone, fixed, case, packet, gaussian,
                                    common | fields | {"history": FIXED}, (510,))
            closed = fixed if q == "Q1" else pairs[q].advance(run.codec, image, index, cases[q], gaussian)
            info = common if q == "Q1" else frame_fields(catalog, video, closed, pairs[q], q)
            yield from display_rows(scorer, backbone, closed, case, closed.payload, gaussian,
                                    info | fields | {"history": CLOSED}, (510,))
        del scorer


def points(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["arm"], row["history"], row["budget"]].append(row)
    return [{"arm": arm, "history": history, "budget": budget, **quality_summary(group)}
            for (arm, history, budget), group in grouped.items()]


def difference(after, before):
    return {key: after[key] - before[key] for key in METRICS}


def effects(points):
    q1, q2, q3, q4 = (points[q] for q in FACTORS)
    scan_a1 = difference(q2, q1)
    scan_a2 = difference(q4, q3)
    return {"scan_a1": scan_a1, "scan_a2": scan_a2, "quant_old": difference(q3, q1),
            "quant_sp1": difference(q4, q2), "interaction": difference(scan_a2, scan_a1),
            "Q4_minus_Q1": difference(q4, q1)}


def contrasts(fixed, closed):
    a, b = effects(fixed), effects(closed)
    return {FIXED: a, CLOSED: b, "closed_minus_fixed": {effect: difference(b[effect], a[effect]) for effect in a},
            "closed_minus_fixed_per_arm": {q: difference(closed[q], fixed[q]) for q in FACTORS}}


def factorial_summary(rows):
    result = {}
    for scope, selected in scopes(rows).items():
        scored = points(selected)
        index = {(p["history"], p["arm"]): p for p in scored}
        effect_sizes = {}
        for kind in ("all", "I", "P"):
            if kind not in scored[0]:
                continue
            fixed, closed = ({q: index[history, q][kind] for q in FACTORS} for history in (FIXED, CLOSED))
            effect_sizes[kind] = contrasts(fixed, closed)
        result[scope] = {"points": scored, "contrasts": effect_sizes}
    return result


def historical_comparison(rows):
    prior = json_at(HISTORICAL_DENSE)
    result = {}
    for arm, old in (("O1", "A0"), ("O2", "A2")):
        current = {(r["video"], r["frame"]): r for r in rows if r["arm"] == arm}
        sealed = {(r["video"], r["frame"]): r for r in prior["rows"] if r["arm"] == old}
        shared = current.keys() & sealed.keys()
        intra = [key for key in shared if current[key]["frame_type"] == "I"]
        result[arm] = {"old_arm": old, "same_frame_inventory": current.keys() == sealed.keys(),
                       "same_psnr_aggregation": True, "same_p_reference_protocol": False,
                       "old_reference": "dense A1 at all 32640 symbols",
                       "current_reference": "C1 A1 + old scan at I2048/P510",
                       "old": quality_summary(list(sealed.values())),
                       "max_intra_frame_mse_abs_difference": max(
                           (abs(current[key]["mse"] - sealed[key]["mse"]) for key in intra), default=None)}
    return result


def diagnostic_summary(rows):
    scored = points(rows)
    index = {(p["arm"], p["budget"]): p for p in scored}
    chain = [("O0", None), ("O1", None), ("O2", DENSE), ("O3", 8192), ("O4", 8192)]
    changes = {}
    for name, before, after in zip(("channel", "quantization", "spatial", "learned"), chain, chain[1:]):
        changes[name] = {kind: difference(index[after][kind], index[before][kind]) for kind in ("all", "I", "P")}
    return {"points": scored, "changes": changes, "comparison_to_a8474ee": historical_comparison(rows)}


def require_phase0():
    record = json_at(DONE)
    if (not record["integrity"]["passed"] or record["integrity"]["images_read"]["train"]
            or record["integrity"]["images_read"]["test"] or record["provenance_sha256"] != file_hash(PROVENANCE)
            or record["outputs"] != {name: file_hash(OUTPUT / name) for name in RESULTS}):
        raise RuntimeError("Phase 0 completion evidence differs from the locked measurements")
    return record


def run_phase0(run):
    from .phase0_report import publish

    if DONE.is_file():
        require_phase0()
        print("Phase 0 already verified.", flush=True)
        return
    catalog, counts = validation(run)
    if counts != json_at(HISTORICAL_DENSE)["cohort"]:
        raise RuntimeError("Phase 0 differs from the recorded Validation cohort")
    cases = factorial_cases(run)
    rows = []
    for video, count in counts.items():
        rows.extend(diagnose_video(run, catalog, video, run.reader.read(catalog, video, 0, count), cases))
        print(f"Phase 0 · {video} · {count} frames · {len(rows)} cumulative measurements", flush=True)
    diagnostic = [r for r in rows if r["arm"].startswith("O")]
    factorial = [r for r in rows if r["arm"] in FACTORS]
    arms = {(arm, FIXED, k) for arm, k in DENSE_ARMS.items()} | set(product(("O3", "O4"), (FIXED,), BUDGETS))
    validate_rows(diagnostic, counts, arms)
    validate_rows(factorial, counts, set(product(FACTORS, (FIXED, CLOSED), (510,))))
    if any(r["symbols_received"] != 510 for r in factorial):
        raise RuntimeError("K510 factorial did not receive exactly 510 FSQ symbols")
    provenance = file_hash(PROVENANCE)
    meta = {"cohort": counts, "provenance_sha256": provenance,
            "command": "python scripts/aip_vtok/run_sparse_decoder.py phase0",
            "fixed_reference": "C1 A1/B2 at I2048/P510", "train_updates": 0}
    diag = meta | diagnostic_summary(diagnostic)
    crossed = meta | {"factors": FACTORS} | factorial_summary(factorial)
    write_csv(OUTPUT / "phase0_per_frame.csv", rows)
    write_json(OUTPUT / "phase0_diagnostics.json", diag)
    write_json(OUTPUT / "phase0_factorial.json", crossed)
    publish(diag, crossed, factorial)
    write_json(DONE, {"rows": len(rows), "integrity": run.verify_integrity(), "provenance_sha256": provenance,
                      "outputs": {name: file_hash(OUTPUT / name) for name in RESULTS}})
