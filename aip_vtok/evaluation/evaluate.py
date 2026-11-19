"""Full-frame Gate 0 diagnostic. Continuous controls have no bitrate claim."""

import time

import torch
from torch.nn import functional as F

from aip_vtok.backbones.dcvc_aligned import AlignedDCVC
from aip_vtok.decoder.causal_state import ReferenceState
from aip_vtok.evaluation.benchmark import projector_latency
from aip_vtok.evaluation.gates import distortion_summary, psnr
from aip_vtok.runtime import file_hash, resolve, save_checkpoint, write_json
from aip_vtok.training.data import read_window, source_by_id, sources
from cp_adaptok.data import save_rgb


def create_backbone(config):
    return (
        AlignedDCVC(
            resolve(config["intra_checkpoint"]), resolve(config["ld_checkpoint"]), config["qp"], config["gop"]
        )
        .cuda()
        .eval()
        .to(memory_format=torch.channels_last)
    )


@torch.no_grad()
def calibrate(config, backbone):
    rows, hashes = sources(config, "train")
    intra, predictive, selected = [], [], []
    for content_id in config["calibration"]["videos"]:
        row = source_by_id(rows, content_id)
        count = config["calibration"]["frames_per_video"]
        for start in torch.linspace(0, row["original_frames"] - 2, count).long().tolist():
            clip = read_window(row, start, 2)
            context = backbone.begin_frame(ReferenceState(), 0)
            y = backbone.native_latent(clip[:1].cuda().float() / 255, context)
            intra.append(y.cpu())
            _, feature = backbone.decode_native(y, context)
            context = backbone.begin_frame(context.commit_reference_once(feature), 1)
            predictive.append(backbone.native_latent(clip[1:2].cuda().float() / 255, context).cpu())
            selected.append({"video": content_id, "start": start, "frames": 2})
        print(f"calibration {content_id}", flush=True)
    statistics = backbone.alignment.fit_intra(intra)
    backbone.alignment.fit_predictive(predictive)
    artifact = {
        "alignment": backbone.alignment.state_dict(),
        "statistics": statistics,
        "sources": selected,
        "manifest_hashes": hashes,
        "qp": config["qp"],
    }
    save_checkpoint(resolve(config["checkpoints"]) / "alignment.pt", artifact)
    write_json(
        resolve(config["output"]) / "calibration.json",
        {k: v for k, v in artifact.items() if k != "alignment"},
    )


def load_alignment(config, backbone):
    path = resolve(config["checkpoints"]) / "alignment.pt"
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    _, hashes = sources(config, "train")
    if artifact["qp"] != config["qp"] or artifact["manifest_hashes"] != hashes:
        raise ValueError("alignment calibration identity mismatch")
    backbone.alignment.load_state_dict(artifact["alignment"])


def metric_row(rgb, reconstruction, frame_id, context):
    mse = F.mse_loss(rgb, reconstruction).item()
    return {
        "frame": frame_id,
        "frame_type": "I" if context.intra else "P",
        "mse": mse,
        "psnr": psnr(mse),
        "height": rgb.shape[-2],
        "width": rgb.shape[-1],
    }


def summarize_rollout(rows):
    result = distortion_summary(rows)
    result["late_48_63"] = distortion_summary([r for r in rows if 48 <= r["frame"] <= 63])
    result["I"] = distortion_summary([r for r in rows if r["frame_type"] == "I"])
    result["P"] = distortion_summary([r for r in rows if r["frame_type"] == "P"])
    return result


@torch.no_grad()
def rollout(config, backbone, row, mode, projector=None):
    state, rows = ReferenceState(), []
    directory = resolve(config["output"]) / mode
    directory.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for frame_id in range(config["validation"]["frames"]):
        rgb = read_window(row, frame_id, 1).cuda().float() / 255
        context = backbone.begin_frame(state, frame_id)
        native = backbone.native_latent(rgb, context)
        aligned = backbone.alignment.align(native, context.intra)
        latent = native if mode == "r0_native" else backbone.alignment.restore(aligned, context.intra)
        rec, decoded_reference = backbone.decode_native(latent, context)
        reference = decoded_reference if projector is None else projector(aligned, context.intra)
        rows.append(
            {
                **metric_row(rgb, rec, frame_id, context),
                "alignment_relative_mse": float(
                    (latent - native).square().mean() / native.square().mean().clamp_min(1e-8)
                ),
                "reference_relative_mse": float(
                    (reference - decoded_reference).square().mean()
                    / decoded_reference.square().mean().clamp_min(1e-8)
                ),
                "reference_rms": float(reference.square().mean().sqrt()),
            }
        )
        state = context.commit_reference_once(reference)
        if frame_id in (0, 31, 63):
            save_rgb(directory / f"reconstruction_{frame_id:03d}.png", rec)
            save_rgb(directory / f"original_{frame_id:03d}.png", rgb)
            save_rgb(directory / f"crop_{frame_id:03d}.png", rec[..., 400:656, 800:1056])
        if (frame_id + 1) % 16 == 0:
            print(f"{mode} frames={frame_id + 1} psnr={psnr(rows[-1]['mse']):.4f}", flush=True)
    torch.cuda.synchronize()
    artifact = {
        "mode": mode,
        "summary": summarize_rollout(rows),
        "rows": rows,
        "wall_s_including_io": time.perf_counter() - started,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "rgb_decoder_calls": len(rows),
        "reference_source": "native_decoder" if projector is None else "latent_projector_only",
        "real_bitstream": False,
        "bpp": None,
        "note": "continuous Gate 0 control; no tokenizer, no claimed codec FPS",
    }
    write_json(directory / "rollout.json", artifact)
    return artifact


@torch.no_grad()
def native_bitstream_control(config, row):
    from cp_adaptok.backbones.native_baseline import NativeBaseline

    codec = NativeBaseline(
        resolve(config["ld_checkpoint"]),
        resolve(config["intra_checkpoint"]),
        config["device"],
        qp=config["qp"],
    )
    rows = []
    for frame_id in range(config["validation"]["frames"]):
        rgb = read_window(row, frame_id, 1).cuda().float() / 255
        rec, counts, _ = codec.round_trip(rgb, force_intra=frame_id % config["gop"] == 0)
        mse = F.mse_loss(rgb, rec).item()
        rows.append({"frame": frame_id, "mse": mse, "psnr": psnr(mse), **counts})
    result = {
        "summary": distortion_summary(rows),
        "rows": rows,
        "real_bitstream": True,
        "bpp": sum(r["total_bytes"] for r in rows) * 8 / (len(rows) * 1080 * 1920),
    }
    write_json(resolve(config["output"]) / "native_bitstream_control.json", result)
    return result


def run_baseline(config):
    backbone = create_backbone(config)
    if not (resolve(config["checkpoints"]) / "alignment.pt").exists():
        calibrate(config, backbone)
    else:
        load_alignment(config, backbone)
    rows, _ = sources(config, "validation")
    row = source_by_id(rows, config["validation"]["video"])
    results = {}
    for mode in ("r0_native", "r0_aligned"):
        results[mode] = rollout(config, backbone, row, mode)["summary"]
    # Free the pure-PyTorch control before allocating four native CUDA proxies.
    del backbone
    torch.cuda.empty_cache()
    native_bitstream_control(config, row)
    write_json(resolve(config["output"]) / "baseline.json", results)
    return results


def evaluate_projector(config, backbone, projector):
    rows, _ = sources(config, "validation")
    row = source_by_id(rows, config["validation"]["video"])
    mode = f"projector_w{projector.width}"
    result = rollout(config, backbone, row, mode, projector)
    result["training_steps"] = config["projector_training"]["steps"]
    result["projector_latency"] = projector_latency(projector)
    result["checkpoint_sha256"] = file_hash(resolve(config["checkpoints"]) / mode / "latest.pt")
    write_json(resolve(config["output"]) / mode / "evaluation.json", result)
    return result
