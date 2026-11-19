"""D0-E compares independent decoder histories driven by real FSQ IDs."""

from collections import defaultdict

import numpy as np
import torch
from scipy.stats import rankdata
from torch.nn import functional as F

from aip_vtok.decoder.causal_state import ReferenceState
from aip_vtok.evaluation.fix import HORIZON_VIDEO, rgb_frame, rms
from aip_vtok.evaluation.formal import append_until
from aip_vtok.tokenizer.fsq import FSQ
from cp_adaptok.data import save_rgb

from .io import FIX, OUTPUT, gpu_snapshot, json_at, read_probe, write_probe
from .matched import CLEAN
from .metrics import measure, summarize
from .packets import COARSE, FULL, HYBRID, Transform
from .sparse import BUDGETS, GROUPS, REF_BUDGET

FIX_A = "fix_a_caae55b"
ISOLATION_BUDGETS = (2048, 8192)
HORIZON = 64
CROP = (400, 656, 800, 1056)
RANKING_FIELDS = (("psnr_global_mse", -1), ("lpips", 1), ("dists", 1), ("latent_mse", 1))


def receive_frame(context, receive, render, projector, display_budgets):
    displays = {}
    for k in display_budgets:
        latent = receive(k)
        displays[k] = (render(latent, context)[0], latent)
    reference = receive(REF_BUDGET[context.intra])
    feature = projector(reference, context.intra)
    return displays, context.commit_reference_once(feature, reference)


def rank_family(rows, layouts):
    subset = [row for row in rows if row["layout"] in layouts]
    by_slice = defaultdict(list)
    for row in subset:
        by_slice[row["frame_type"], row["budget"]].append(row)
    scores = defaultdict(list)
    for cases in by_slice.values():
        for field, sign in RANKING_FIELDS:
            ranks = rankdata([sign * row[field] for row in cases])
            for row, rank in zip(cases, ranks):
                scores[row["layout"], row["order"]].append(rank)
    ranking = [
        {"layout": layout, "order": order, "mean_rank": float(np.mean(values))}
        for (layout, order), values in scores.items()
    ]
    return sorted(ranking, key=lambda row: (row["mean_rank"], row["layout"], row["order"]))


def select_candidates(summary):
    rows = [
        row for row in summary
        if row["history"] == CLEAN and row["cohort"] == "all" and row["version"] == "fsq_quantized"
    ]
    return {"pure": rank_family(rows, (FULL, COARSE)), "hybrid": rank_family(rows, (HYBRID,))}


def nested_ids(transform, reference, display):
    inner, outer = (transform.positions(packet.support) for packet in (reference, display))
    overlapping = torch.searchsorted(outer, inner, out_int32=False)
    return (
        bool((overlapping < len(outer)).all())
        and torch.equal(outer[overlapping], inner)
        and torch.equal(display.symbols.values[overlapping], reference.symbols.values)
        and (reference.anchor is display.anchor)
    )


def same_state(a, b):
    return a.frame_id == b.frame_id and all(
        left is right or (left is not None and right is not None and torch.equal(left, right))
        for left, right in zip((a.feature, a.memory, a.latent), (b.feature, b.memory, b.latent))
    )


def sparse_reference_usage(transform, packet):
    slots = transform.positions(packet.support)
    enhancement_positions = torch.unique_consecutive(slots // GROUPS).numel()
    anchor_tokens = 0 if packet.anchor is None else len(packet.anchor.values)
    coverage = 100 * enhancement_positions / (transform.shape[0] * transform.shape[1])
    return {
        "reference_tokens_used": len(slots) + anchor_tokens,
        "reference_spatial_coverage_pct": 100.0 if anchor_tokens else coverage,
        "reference_enhancement_positions": enhancement_positions,
    }


def sparse_frame(codec, fits, fsq, candidate, rgb, states, frame):
    low, high = (codec.backbone.begin_frame(state, frame) for state in states)
    target = codec.backbone.aligned_latent(rgb, low)
    transform = Transform(
        fits[low.intra], candidate["order"], "fsq_quantized", candidate["layout"], fsq, target.shape[-2:],
    )
    packets = transform.encode(target, BUDGETS)
    reference = packets[REF_BUDGET[low.intra]]
    if not all(nested_ids(transform, reference, packets[k]) for k in ISOLATION_BUDGETS):
        raise RuntimeError("reference FSQ IDs are not a subset of the display packet")

    def receive(k):
        return transform.receive(packets[k])

    render = codec.backbone.decode_aligned
    # The first receiver renders at most 2048 before committing. Higher renders use the frozen context.
    low_budgets = tuple(k for k in BUDGETS if k <= ISOLATION_BUDGETS[0])
    displays, first = receive_frame(low, receive, render, codec.projector, low_budgets)
    higher, second = receive_frame(high, receive, render, codec.projector, (4096, ISOLATION_BUDGETS[1]))
    usage = sparse_reference_usage(transform, reference)
    return low.intra, target, {**displays, **higher}, (first, second), usage


def fix_frame(codec, rgb, states, frame):
    encoded = codec.encode_frame(rgb, states[0], frame)
    low = codec.begin_frame(frame, encoded.tokens.tiers, states[0])
    high = codec.begin_frame(frame, encoded.tokens.tiers, states[1])
    displays = {}
    for budget in BUDGETS:
        session = low if budget <= ISOLATION_BUDGETS[0] else high
        append_until(session, encoded.tokens.ids, budget)
        image, _, fused = session.render_frame(with_latents=True)
        displays[budget] = image, fused
    first, second = low.commit_reference_once(), high.commit_reference_once()
    usage = {"reference_tokens_used": low.reference_budget, "reference_spatial_coverage_pct": 100.0}
    return low.context.intra, encoded.native_latent, displays, (first, second), usage


def save_crops(layout, rgb, displays, frame):
    y0, y1, x0, x1 = CROP
    directory = OUTPUT / "figures" / "crops"
    if layout == FIX_A:
        save_rgb(directory / f"f{frame:03d}_original.png", rgb[..., y0:y1, x0:x1])
    for budget, (rec, _) in displays.items():
        save_rgb(directory / f"f{frame:03d}_{layout}_{budget}.png", rec[..., y0:y1, x0:x1])


def closed_rows(suite, candidate, rgb, intra, target, displays, state, usage, frame):
    reference_rms = rms(state.latent)
    target_rms = rms(target)
    reference = {
        "reference_budget": REF_BUDGET[intra], **usage,
        "reference_latent_rms": reference_rms, "native_aligned_latent_rms": target_rms,
        "reference_to_native_rms": reference_rms / target_rms,
        "reference_latent_mse": F.mse_loss(state.latent, target).item(),
        "projected_feature_rms": rms(state.feature),
    }
    return [
        {
            "video": HORIZON_VIDEO, "frame": frame, "frame_type": "I" if intra else "P", "split": "validation",
            "layout": candidate["layout"], "order": candidate["order"], "budget": budget,
            **measure(suite, rgb, rec, latent, target), **reference, "real_bitstream": False,
        }
        for budget, (rec, latent) in displays.items()
    ]


@torch.no_grad()
def roll_candidate(candidate, clip, codec, suite, fits):
    states, rows, fingerprints = (ReferenceState(), ReferenceState()), [], {}
    layout = candidate["layout"]
    fsq = FSQ().cuda()
    for frame, original in enumerate(clip):
        rgb = rgb_frame(original)
        if layout == FIX_A:
            intra, target, displays, next_states, usage = fix_frame(codec, rgb, states, frame)
        else:
            intra, target, displays, next_states, usage = sparse_frame(codec, fits, fsq, candidate, rgb, states, frame)
        if not same_state(*next_states):
            raise RuntimeError(f"display budget changed the next reference: {layout}, frame {frame}")
        states = next_states
        rows.extend(closed_rows(suite, candidate, rgb, intra, target, displays, states[0], usage, frame))
        if frame in (0, 31, 32, 63):
            fingerprints[str(frame)] = states[0].fingerprint()
        if frame in (31, 63):
            save_crops(layout, rgb, displays, frame)
        if (frame + 1) % 16 == 0:
            gpu_snapshot()
            print(f"E {layout} frames={frame + 1}/64", flush=True)
    return rows, {"checked_frames": len(clip), "bitwise_identical": True, "fingerprints": fingerprints}


def summarize_closed(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["layout"], row["order"], row["budget"]].append(row)
    result = []
    for (layout, order, budget), group in groups.items():
        frames = {row["frame"]: row for row in group}
        result.append({
            "layout": layout, "order": order, "budget": budget, **summarize(group),
            "P": summarize([row for row in group if row["frame_type"] == "P"]),
            "late_48_63": summarize([frames[i] for i in range(48, 64)]),
            "early_p_01_16": summarize([frames[i] for i in range(1, 17)]),
            "reference_latent_rms_31": frames[31]["reference_latent_rms"],
            "reference_latent_rms_63": frames[63]["reference_latent_rms"],
            "reference_to_native_rms_31": frames[31]["reference_to_native_rms"],
            "reference_to_native_rms_63": frames[63]["reference_to_native_rms"],
        })
    return result


def historical_replay(rows):
    historic = json_at(FIX / "horizon.json")["rows"]
    reproduced = [row for row in rows if row["layout"] == FIX_A and row["budget"] == 8192]
    fields = ("mse", "reference_latent_rms", "native_aligned_latent_rms")
    return {key: max(abs(row[key] - historic[row["frame"]][key]) for row in reproduced) for key in fields}


def run_closed(catalog, codec, suite, fits):
    ranking = select_candidates(read_probe("D")["summary"])
    candidates = [{"layout": FIX_A, "order": "sealed"}, ranking["pure"][0], ranking["hybrid"][0]]
    clip = catalog.read(HORIZON_VIDEO, 0, HORIZON)
    rows, isolation = [], {}
    for candidate in candidates:
        candidate_rows, proof = roll_candidate(candidate, clip, codec, suite, fits)
        rows.extend(candidate_rows)
        isolation[candidate["layout"]] = proof
    write_probe(
        "E", rows, summarize_closed(rows), candidates=candidates, candidate_ranking=ranking,
        ranking_basis="mean ordinal rank of PSNR, LPIPS, DISTS, latent MSE across I/P and five budgets",
        history="independent per layout, same decoded reference across 2048 and 8192 display loops",
        isolation=isolation, historical_replay_max_abs=historical_replay(rows),
        crop_xyxy=[CROP[2], CROP[0], CROP[3], CROP[1]],
    )
