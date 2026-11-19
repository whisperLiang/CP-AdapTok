"""Same-context, 174-frame quality and 64-frame drift measurements for Gate 1-Fix."""

import csv
from collections import defaultdict

import torch
from torch.nn import functional as F

from aip_vtok.decoder.causal_state import ReferenceState
from aip_vtok.evaluation.fix_metrics import frame_fsq, pooled_fsq
from aip_vtok.evaluation.formal import (
    evaluate_budget,
    intervention_rows,
    metric_suite,
    summarize_interventions,
    validation_inputs,
    write_frame_metrics,
)
from aip_vtok.evaluation.gates import distortion_summary
from aip_vtok.evaluation.smoke import BUDGETS, aggregate_budgets
from aip_vtok.models.codec import create_codec
from aip_vtok.runtime import file_hash, write_json
from aip_vtok.training.checkpoint import load_fix_weights, require_identity
from aip_vtok.training.data import read_window, source_by_id, sources
from aip_vtok.training.determinism import configure_determinism
from aip_vtok.training.fix import CONFIG, directories, fix_settings, phase_parser, read_json, require_release
from aip_vtok.training.formal import training_input
from aip_vtok.training.losses import latent_mse_metrics
from aip_vtok.training.prefix import prefix_latents

HORIZON_VIDEO = "DAVIS2017:car-roundabout"


def rgb_frame(frame):
    return frame[None].cuda().float() / 255


def rms(tensor):
    return None if tensor is None else tensor.float().square().mean().sqrt().item()


def reference_row(codec, encoded, state, rgb, mse):
    direct, feature = codec.backbone.decode_aligned(encoded.native_latent, encoded.context)
    return {
        "frame": encoded.frame_id,
        "frame_type": "I" if encoded.context.intra else "P",
        "mse": mse,
        "direct_mse": F.mse_loss(rgb, direct).item(),
        "native_aligned_latent_rms": rms(encoded.native_latent),
        "reference_latent_rms": rms(state.latent),
        "native_reference_feature_rms": rms(feature),
        "projected_reference_feature_rms": rms(state.feature),
        "memory_rms": rms(state.memory),
        "reference_feature_l1": F.l1_loss(state.feature, feature).item(),
    }


@torch.no_grad()
def horizon_64(codec, clip):
    state, rows = ReferenceState(), []
    for index, frame in enumerate(clip[:64]):
        rgb = rgb_frame(frame)
        encoded = codec.encode_frame(rgb, state, index)
        session = codec.prefix_session(encoded, state, BUDGETS[-1])
        mse = F.mse_loss(rgb, session.render_frame()).item()
        state = session.commit_reference_once()
        rows.append(reference_row(codec, encoded, state, rgb, mse))
    return rows


def diagnose_frame(codec, encoded, rgb, video):
    tokens, context = encoded.tokens, encoded.context
    direct, feature = codec.backbone.decode_aligned(encoded.native_latent, context)
    direct_mse = F.mse_loss(rgb, direct).item()
    endpoints = {str(p): int((tokens.plan.pass_ids <= p).sum()) for p in range(3)}
    rows = []
    for budget in sorted(set(BUDGETS) | set(endpoints.values())):
        raw, fused = prefix_latents(codec, tokens, encoded.prediction, budget, differentiable=False)
        rec, _ = codec.backbone.decode_aligned(fused, context)
        rows.append(
            {
                "video": video,
                "frame": encoded.frame_id,
                "frame_type": "I" if context.intra else "P",
                "budget": budget,
                "mse": F.mse_loss(rgb, rec).item(),
                "direct_mse": direct_mse,
                **latent_mse_metrics(raw, fused, encoded.native_latent),
                "native_latent_rms": rms(encoded.native_latent),
                "pass_endpoint": next((p for p, end in endpoints.items() if end == budget), None),
            }
        )
    return rows, feature


@torch.no_grad()
def matched_six(codec, selected, clips):
    rows = []
    for row, clip in zip(selected, clips):
        state = ReferenceState()
        for index, frame in enumerate(clip[:2]):
            rgb = rgb_frame(frame)
            encoded = codec.encode_frame(rgb, state, index)
            measured, native_feature = diagnose_frame(codec, encoded, rgb, row["content_id"])
            budget = codec.reference_i if encoded.context.intra else codec.reference_p
            state = codec.prefix_session(encoded, state, budget).commit_reference_once()
            for record in measured:
                record["reference_feature_l1"] = F.l1_loss(state.feature, native_feature).item()
            rows.extend(measured)
    return rows


def summarize_matched(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[str(row["budget"])].append(row)
    return {
        budget: {
            **distortion_summary(group),
            "budget": int(budget),
            "matched_direct": distortion_summary([{"mse": r["direct_mse"]} for r in group]),
            **{
                field: sum(r[field] for r in group) / len(group)
                for field in (
                    "raw_latent_mse",
                    "fused_latent_mse",
                    "delta_fusion_mse",
                    "reference_feature_l1",
                )
            },
            "pass_endpoint": group[0]["pass_endpoint"],
        }
        for budget, group in groups.items()
    }


def donor_ids(codec, clip):
    return codec.encode_frame(rgb_frame(clip[0]), ReferenceState(), 0).tokens.ids


@torch.no_grad()
def evaluate_sequence(
    config, codec, suite, video, clip, pictures, *, independent=False, donor=None, health=True
):
    state, rng = ReferenceState(), torch.Generator().manual_seed(7301)
    streams = {budget: ReferenceState() for budget in BUDGETS} if independent else {}
    rows, fsq, mutations, comparisons, horizon = [], [], [], [], []
    for index, frame in enumerate(clip):
        rgb = rgb_frame(frame)
        encoded = codec.encode_frame(rgb, state, index)
        if health:
            fsq.append({"video": video, "frame": index, "passes": frame_fsq(encoded, codec.tokenizer.fsq)})
        session = codec.begin_frame(index, encoded.tokens.tiers, state)
        for budget in BUDGETS:
            measured, following, exact, mse = evaluate_budget(
                config,
                codec,
                suite,
                video,
                index,
                rgb,
                encoded,
                session,
                streams.get(budget),
                budget,
                pictures,
            )
            rows.extend(measured)
            if independent:
                streams[budget] = following
                comparisons.append(exact)
        if donor is not None:
            mutations.extend(
                {"video": video, "frame": index, **row}
                for row in intervention_rows(codec, encoded, session, donor, rgb, mse, rng)
            )
        state = session.commit_reference_once()
        if independent:
            for exact, follow in zip(comparisons[-len(BUDGETS) :], streams.values()):
                exact["equal_reference"] = torch.equal(follow.latent, state.latent) and torch.equal(
                    follow.feature, state.feature
                )
            reference = streams[BUDGETS[-1]]
            horizon.append(reference_row(codec, encoded, reference, rgb, measured[-1]["mse"]))
        if (index + 1) % 16 == 0:
            print(f"{config['fix_phase']} seed={config['seed']} {video} frames={index + 1}", flush=True)
    return rows, fsq, mutations, comparisons, horizon


def summarize_validation(rows):
    result = {"paired": aggregate_budgets([row for row in rows if row["mode"] == "paired"])}
    independent = [row for row in rows if row["mode"] == "independent"]
    if independent:
        result["independent_64"] = aggregate_budgets(independent)
        result["late_48_63"] = aggregate_budgets([row for row in independent if 48 <= row["frame"] <= 63])
    return result


def frame_metrics(directory):
    labels = {"mode", "video", "frame_type", "bpp", "split", "real_bitstream"}
    with (directory / "frame_metrics.csv").open() as handle:
        return [
            {k: v if k in labels else float(v) for k, v in row.items() if v != ""}
            for row in csv.DictReader(handle)
        ]


@torch.no_grad()
def evaluate_validation(config, codec, suite, selected, clips, directory, *, final):
    phase = config["fix_phase"]
    phases_with_horizon = ("b0", "b1", "fix_c")
    interventions = phase == "fix_c" and final
    donors = [donor_ids(codec, clip) for clip in clips] if interventions else [None] * len(clips)
    stack = [[], [], [], [], []]
    for i, (row, clip) in enumerate(zip(selected, clips)):
        video = row["content_id"]
        measured = evaluate_sequence(
            config,
            codec,
            suite,
            video,
            clip,
            directory if final and phase in ("fix_a", "fix_c") else None,
            independent=final and phase in phases_with_horizon and video == HORIZON_VIDEO,
            donor=donors[(i + 1) % len(clips)],
            health=final,
        )
        for target, output in zip(stack, measured):
            target.extend(output)
    rows, health, mutations, exact, horizon = stack
    write_frame_metrics(directory, rows)
    if final:
        write_json(directory / "health.json", health)
    if mutations:
        write_json(directory / "interventions.json", mutations)
    if exact:
        write_json(directory / "independent_exactness.json", exact)
    return {
        "summary": summarize_validation(rows),
        "health_summary": pooled_fsq(health) if health else {},
        "intervention_summary": summarize_interventions(mutations) if mutations else {},
    }, horizon


def load_evaluation_codec(config, directory, checkpoint, step):
    artifact = torch.load(checkpoint, map_location="cpu", weights_only=False)
    hashes = read_json(directory / "inputs.json")["hashes"]
    require_identity(artifact, config, hashes)
    _, current = training_input(config)
    if any(hashes[name] != digest for name, digest in current.items()):
        raise ValueError("frozen inputs differ from training")
    if artifact["step"] != step:
        raise ValueError("evaluation requires the declared checkpoint step")
    codec = create_codec(config, model_version=config["model_version"]).eval()
    load_fix_weights(codec, artifact)
    return codec


@torch.no_grad()
def smoke_training_clip(config, trained, suite, directory):
    rows, _ = sources(config, "train")
    stage = config["stage1"]
    clip = read_window(source_by_id(rows, stage["video"]), stage["start"], stage["sequence_length"])
    torch.manual_seed(config["seed"])
    initial = create_codec(config, model_version=config["model_version"]).eval()
    measurements = {}
    for label, codec in (("initial", initial), ("final", trained)):
        metrics, *_ = evaluate_sequence(config, codec, suite, stage["video"], clip, None, health=False)
        for row in metrics:
            row["split"] = "train"
        measurements[label] = metrics
    write_json(directory / "overfit.json", measurements)


def evaluate_fix(phase, seed, config_path=CONFIG, *, step=None):
    release = require_release(phase)
    config = fix_settings(phase, seed, config_path, joint_projector=release.get("winner") == "b1")
    configure_determinism()
    directory, weights = directories(phase, seed)
    final = step is None or step == config["stage1"]["steps"]
    step = config["stage1"]["steps"] if final else step
    checkpoint = weights / ("latest.pt" if final else f"step_{step:04d}.pt")
    digest = file_hash(checkpoint)
    if final and read_json(directory / "training_complete.json")["checkpoint_sha256"] != digest:
        raise ValueError("training checkpoint differs from the sealed completion record")
    codec = load_evaluation_codec(config, directory, checkpoint, step)
    if not final:
        directory = directory / f"step_{step:04d}"
    directory.mkdir(parents=True, exist_ok=True)
    suite = metric_suite()
    selected, clips, hashes = validation_inputs(config)
    summary, horizon = {}, []
    if phase == "smoke":
        smoke_training_clip(config, codec, suite, directory)
    else:
        summary, horizon = evaluate_validation(config, codec, suite, selected, clips, directory, final=final)
    if final:
        matched = matched_six(codec, selected, clips)
        car = next(clip for row, clip in zip(selected, clips) if row["content_id"] == HORIZON_VIDEO)
        if not horizon:
            horizon = horizon_64(codec, car)
        write_json(directory / "diagnostics.json", {"rows": matched, "summary": summarize_matched(matched)})
        write_json(
            directory / "horizon.json", {"video": HORIZON_VIDEO, "gop": config["gop"], "rows": horizon}
        )
    artifact = {
        "step": step,
        "seed": seed,
        "phase": phase,
        "manifest_hashes": hashes,
        "checkpoint_sha256": digest,
        "precision": "FP32",
        **summary,
    }
    write_json(directory / "evaluation.json", artifact)
    return artifact


def main():
    parser = phase_parser(__doc__)
    parser.add_argument("--step", type=int)
    args = parser.parse_args()
    result = evaluate_fix(args.phase, args.seed, args.config, step=args.step)
    print({k: result[k] for k in ("phase", "seed", "step", "checkpoint_sha256")}, flush=True)


if __name__ == "__main__":
    main()
