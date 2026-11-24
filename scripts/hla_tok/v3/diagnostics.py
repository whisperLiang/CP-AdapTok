"""Read-only location of B0 loss before the optional Local Decoder experiment."""

from collections import defaultdict

import torch
from torch.nn import functional as F

from hla_tok.evaluation.benchmark import transfer
from hla_tok.evaluation.bottleneck_diagnostics import reconstruct_intervention
from hla_tok.evaluation.io import write_csv, write_json
from hla_tok.evaluation.metrics import summarize_quality
from scripts.hla_tok.v3.protocol import NATIVE


def _spectrum(coarse, local):
    height, width = coarse.shape[-2:]
    error = torch.fft.rfft2((local - coarse).float(), norm="ortho").abs().square().mean(dim=(0, 1))
    vertical = torch.fft.fftfreq(height, device=error.device)[:, None]
    horizontal = torch.fft.rfftfreq(width, device=error.device)[None, :]
    radius = (vertical.square() + horizontal.square()).sqrt()
    return {name: error[region].mean().item() for name, region in
            (("latent_error_power_low", radius < 0.1),
             ("latent_error_power_mid", (radius >= 0.1) & (radius < 0.25)),
             ("latent_error_power_high", radius >= 0.25))}


@torch.inference_mode()
def diagnose(dense, codec, reader, cohort, quality, budget, output):
    dense.eval()
    codec.eval()
    before = dict(reader.counts)
    observations = defaultdict(list)
    features = []
    try:
        for video, count in cohort.items():
            for index in range(count):
                source = transfer(reader.read("validation", video, index)[None])
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    teacher, _ = dense.coarse(source)
                    _, coarse = codec.encoder(source, NATIVE)
                    tokens = codec.encode_training(source, budget, quantize=False)
                    local = codec._decode_features(tokens.values, NATIVE)
                    comparison = {
                        "phase_a_dense": dense.rgb_decoder(teacher, NATIVE),
                        "b0_dense_through_b0_rgb": codec.rgb_decoder(coarse, NATIVE),
                        "b0_local": codec.rgb_decoder(local, NATIVE),
                    }
                    for case in ("regional_mean", "shuffle_values", "perturb_rank_assignment"):
                        comparison[case] = reconstruct_intervention(codec, tokens.values, NATIVE, case)[0]
                keys = {"video": video, "frame": index}
                features.append(keys | {"b0_dense_vs_local_mse": F.mse_loss(coarse.float(), local.float()).item(),
                                        "phase_a_vs_b0_dense_mse": F.mse_loss(teacher.float(), coarse.float()).item()}
                                | _spectrum(coarse, local))
                for label, image in comparison.items():
                    observations[label].append(keys | {"mode": label} | quality(source, image))
        rows = [{"mode": label} | summarize_quality(items) for label, items in observations.items()]
        write_csv(output / "feature_deltas.csv", features)
        write_csv(output / "per_frame.csv", [row for items in observations.values() for row in items])
        write_csv(output / "metrics.csv", rows)
        return rows
    finally:
        write_json(output / "pixel_reads.json", {key: reader.counts[key] - before[key] for key in before})
