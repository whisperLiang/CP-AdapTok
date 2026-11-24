"""D0 reproductions, query geometry, and controlled interventions on archived R0."""

# ruff: noqa: E402
import argparse
import hashlib
import math
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch

from hla_tok import Geometry, HLATok
from hla_tok.evaluation.benchmark import transfer
from hla_tok.evaluation.bottleneck_diagnostics import INTERVENTIONS, reconstruct_intervention
from hla_tok.evaluation.io import write_csv, write_json
from hla_tok.evaluation.metrics import FullFrameQuality, psnr
from hla_tok.evaluation.representation_diagnostics import (
    attention_summary,
    representation_metrics,
    summarize_prefix,
    trace_cross_attention,
)
from hla_tok.training.checkpoint import read_checkpoint
from scripts.hla_tok.v2.protocol import (
    FIXED_BUDGET,
    OUTPUT,
    audit_legacy,
    audit_read_counts,
    csv_at,
    frozen_data,
    heldout_examples,
    setup,
    verified_checkpoints,
)

GEOMETRY = Geometry(1080, 1920)


def verify_replays(legacy, output):
    original = legacy / "results/hla_tok/phase1"
    rows = []
    fields = ("mse", "global_psnr", "lpips", "lpips_vgg", "dists", "ms-ssim", "boundary_mse", "interior_mse")
    for old, new in (("continuous", "replay_continuous"), ("fixed", "replay_vq"), ("vq_bypass", "replay_bypass")):
        expected, = csv_at(original / old / "validation_metrics.csv")
        replay, = csv_at(output / new / "validation_metrics.csv")
        for key in fields:
            if not math.isclose(float(replay[key]), float(expected[key]), rel_tol=1e-7, abs_tol=1e-8):
                raise RuntimeError(f"R0 replay differs for {old} {key}: {replay[key]} vs {expected[key]}")
        if replay["frames"] != "174" or replay.get("total_bytes") != expected.get("total_bytes"):
            raise RuntimeError("R0 replay must match the complete cohort and actual byte ledger")
        rows.append({"arm": old, "status": "reproduced", **replay})
    write_csv(output / "reconstruction_metrics.csv", rows)
    return rows


def intervention_rows(model, source, encoded, quality, tags, quantized, cases=INTERVENTIONS):
    tokens = encoded.ids if quantized else encoded.continuous
    reference = model.decode_prefix(tokens, GEOMETRY) if quantized else model._decode(tokens, GEOMETRY)
    with torch.autocast("cuda", enabled=False):
        initial = quality(source, reference)
    for case in cases:
        image, count = reconstruct_intervention(model, tokens, GEOMETRY, case, quantized=quantized)
        if case == "original":
            torch.testing.assert_close(image, reference, rtol=0, atol=0)
        with torch.autocast("cuda", enabled=False):
            metrics = quality(source, image)
        yield tags | metrics | {"mode": "quantized" if quantized else "continuous", "intervention": case,
                                "tokens": count, "delta_psnr": metrics["psnr"] - initial["psnr"],
                                "delta_lpips": metrics["lpips"] - initial["lpips"],
                                "delta_dists": metrics["dists"] - initial["dists"],
                                "delta_mse": metrics["mse"] - initial["mse"]}


def representation_rows(model, values, tags):
    selected = summarize_prefix(values, GEOMETRY.num_regions)
    spectra = selected.pop("singular_values").cpu().tolist()
    measures = {key: items.cpu().tolist() for key, items in selected.items()}
    rank = torch.arange(values.shape[1], device=values.device) // GEOMETRY.num_regions
    local = values[0].float()
    means = local[:120 * GEOMETRY.num_regions].reshape(120, GEOMETRY.num_regions, -1).mean(0)
    between = (means - means.mean(0)).square().mean().item()
    rank_rows = [tags | {"rank": r, "output_norm": local[rank == r].norm(dim=-1).mean().item()}
                 for r in range(121)]
    initial = model.query_encoder.query[:121] + model.query_encoder.rank[:121]
    initial_stats = representation_metrics(initial[None])
    initial_projection = model.query_encoder.output(initial).float()
    rank_cosine = torch.nn.functional.cosine_similarity(local, initial_projection[rank], dim=-1).mean().item()
    frame = {key: mean(value) for key, value in measures.items()}
    frame |= tags | {"between_region_mean_variance": between,
                     "input_query_effective_rank_128d": initial_stats["effective_rank"].item(),
                     "input_query_centered_variance_128d": initial_stats["centered_variance"].item(),
                     "output_cosine_with_projected_initial_query": rank_cosine}
    spectrum_rows = [tags | {"region": i, **{f"s{j+1}": v for j, v in enumerate(vals)}}
                     for i, vals in enumerate(spectra)]
    return frame, spectrum_rows, rank_rows


def attention_rows(model, source, tags):
    with trace_cross_attention(model.query_encoder) as (traces, maxima):
        model.encode_training(source, FIXED_BUDGET, quantize=False)
    for layer, blocks in enumerate(traces):
        weights = torch.cat(blocks, dim=2)[:, :, :120]
        for start, stop in ((0, 16), (16, 64), (64, 120)):
            rows, coverage = attention_summary(weights, start, stop)
            for head, row in enumerate(rows):
                yield tags | row | {"layer": layer + 1, "reference_output_max_abs": maxima[layer],
                                    **{f"p{cell}": prob for cell, prob in enumerate(coverage[head].cpu().tolist())}}


@torch.inference_mode()
def analyze(legacy, output):
    reader, _, identity = frozen_data()
    quality = FullFrameQuality("cuda", identity["metric_weights"])
    tables: dict[str, list[dict]] = {key: [] for key in
        ("attention_metrics", "representation_metrics", "rank_spectrum", "rank_norms", "intervention_metrics")}
    model = HLATok().cuda().to(memory_format=torch.channels_last).eval()
    try:
        for stage, path in verified_checkpoints(legacy).items():
            model.load_state_dict(read_checkpoint(path).state["model"], strict=True)
            for row in heldout_examples():
                pixels = reader.read("validation", row["video"], row["index"])
                rgb24 = pixels.permute(1, 2, 0).contiguous().numpy().tobytes()
                if hashlib.sha256(rgb24).hexdigest() != row["decoded_rgb24_sha256"]:
                    raise RuntimeError("diagnostic RGB identity mismatch")
                source = transfer(pixels[None])
                tags = {"stage": stage, "video": row["video"], "frame": row["index"]}
                tables["attention_metrics"] += list(attention_rows(model, source, tags))
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    encoded = model.encode_training(source, FIXED_BUDGET, quantize=stage == "fixed")
                    frame, spectrum, norms = representation_rows(model, encoded.continuous, tags)
                    tables["representation_metrics"].append(frame)
                    tables["rank_spectrum"] += spectrum
                    tables["rank_norms"] += norms
                    tables["intervention_metrics"] += list(intervention_rows(model, source, encoded, quality, tags, False))
                    if stage == "fixed":
                        tables["intervention_metrics"] += list(intervention_rows(model, source, encoded, quality, tags, True))
                print("D0", stage, row["video"], row["index"], flush=True)

    finally:
        audit_read_counts(output / "analysis", reader)
    for key, table in tables.items():
        write_csv(output / f"{key}.csv", table)
    summarize(output, tables)


def summarize(output, tables):
    result = {}
    samples = tables["representation_metrics"]
    for stage in ("continuous", "fixed"):
        selected = [r for r in samples if r["stage"] == stage]
        fields = [k for k, v in selected[0].items() if isinstance(v, float)]
        result[stage] = {key: mean(r[key] for r in selected) for key in fields}
    interventions = []
    for stage, mode in (("continuous", "continuous"), ("fixed", "continuous"), ("fixed", "quantized")):
        for case in INTERVENTIONS:
            rows = [r for r in tables["intervention_metrics"] if
                    (r["stage"], r["mode"], r["intervention"]) == (stage, mode, case)]
            ref = [r for r in tables["intervention_metrics"] if
                   (r["stage"], r["mode"], r["intervention"]) == (stage, mode, "original")]
            fields = ("delta_lpips", "delta_dists", "delta_mse", "delta_psnr")
            interventions.append({"stage": stage, "mode": mode, "intervention": case,
                "global_psnr_delta": psnr(mean(r["mse"] for r in rows)) - psnr(mean(r["mse"] for r in ref)),
                **{key: mean(r[key] for r in rows) for key in fields}})
    write_csv(output / "intervention_summary.csv", interventions)
    result["attention_max_abs_difference"] = max(r["reference_output_max_abs"] for r in tables["attention_metrics"])
    result["centered_variance_definition"] = "mean square after removing each region's rank mean"
    result["engineered_diagnostic_precision"] = "FP32 attention weights and BF16 codec representations"
    write_json(output / "query_summary.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    setup()
    output = OUTPUT / "diag0"
    write_json(output / "legacy_assets.json", audit_legacy(args.legacy_root.resolve()))
    if not args.audit_only:
        verify_replays(args.legacy_root, output)
        analyze(args.legacy_root, output)


if __name__ == "__main__":
    main()
