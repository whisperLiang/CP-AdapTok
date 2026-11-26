"""Compare independent minimum-QP and analysis-attenuated Lite Base chains."""

import argparse
from pathlib import Path
from time import perf_counter

import torch

from hla_tok.evaluation.io import write_csv, write_json
from hla_tok.uf.anchor import Segmentation
from hla_tok.uf.base import schedule
from hla_tok.uf.latent import UnifiedRefinement
from hla_tok.uf.official import OfficialAssets
from hla_tok.uf.wire import APP_BASE_OVERHEAD, FrameParser, pack_base
from hla_tok.uf_v2.native_weights import LiteProfile
from hla_tok.uf_v2.runtime import Unified
from scripts.hla_tok.continuous.quality import NativeQuality, summarize
from scripts.hla_tok.temporal.inputs import source_frame
from scripts.hla_tok.uf.inputs import inputs
from scripts.hla_tok.uf_v2.base_sweep import report
from scripts.hla_tok.uf_v2.evaluate import allowance
from scripts.hla_tok.v2.protocol import setup

CAP = 4096
PROBE_BUDGETS = (0.5, 1, 1.3947, 1.395, 2, 3, 5)


class AttenuatedIntra(LiteProfile):
    """Lower hyper-analysis input magnitude as well as native I symbols."""

    gain = 0.5

    def __init__(self):
        super().__init__(1, 0.8)

    def identity(self):
        return f"pre-hyper-i-gain-{self.gain.hex()}/{super().identity()}"

    @torch.no_grad()
    def install(self, endpoint):
        super().install(endpoint)
        intra = endpoint.intra
        with torch.cuda.stream(endpoint.stream):
            projection = intra.enc.enc_2[-1]
            projection.weight.mul_(self.gain)
            projection.bias.mul_(self.gain)
            intra.q_scale_y_dec[0].div_(self.gain)
            intra.proxy.set_param(intra.add_cdf_to_state_dict(intra.state_dict()), intra.gaussian_encoder.skip_thres)
        endpoint.stream.synchronize()


def evaluate(assets, reader, clips, profile, name, metric):
    # Allow a complete diagnostic replay so bytes of later P frames are
    # measured even when the declared service cap would lose the chain.
    encoder = Unified(assets, UnifiedRefinement().cuda(), (0, 0), segmentation=Segmentation(8, 8),
                      base_cap=16384, family_id=name, lite=profile)
    decoder = encoder.new_decoder()
    rows = []
    for clip in clips:
        encoder.reset()
        decoder.reset()
        desync = set()
        for index in range(clip.length):
            rgb = source_frame(reader, clip, index)
            intra, reset = schedule(index)
            start = perf_counter()
            base = encoder.native.encode_base(rgb, index, (0, 0), intra, reset, 16384)
            raw = pack_base(base, encoder.tag)
            parser = FrameParser(encoder.tag)
            parser.feed(raw)
            [restored] = parser.available()
            parser.finish()
            display = decoder.decode(restored)
            latency = 1000 * (perf_counter() - start)
            if not torch.equal(decoder.decoded_latent(), encoder.native.proxy(intra).ufv2_encode_snapshot()[1]):
                raise ValueError("Lite native Base disagrees at an independent receiver")
            row = {"video": clip.video, "frame": index, "type": "I" if intra else "P",
                   "application_bytes": len(raw), "base_overhead_bytes": APP_BASE_OVERHEAD,
                   "native_joint_y_z_bytes": len(base.payload), "total_ms": latency}
            for cap in (CAP, *(allowance(b) for b in PROBE_BUDGETS)):
                if len(raw) > cap:
                    desync.add(cap)
                    status = "BASE_BUDGET_EXCEEDED"
                elif intra:
                    desync.discard(cap)
                    status = "PASS"
                else:
                    status = "DESYNC" if cap in desync else "PASS"
                row[str(cap)] = status
            metric.prepare(rgb)
            rows.append(row | metric.score(display))
        print(name, clip.video, flush=True)
    budgets = {str(b): {"cap": allowance(b),
                         "decoded": sum(row[str(allowance(b))] == "PASS" for row in rows),
                         "base_exceeded": sum(row[str(allowance(b))] == "BASE_BUDGET_EXCEEDED" for row in rows)}
               for b in PROBE_BUDGETS}
    return rows, {**report(rows, CAP), "quality_unrestricted_base": summarize(rows),
                  "service_budgets": budgets, "prefix_family_id": name, "tag": encoder.tag.hex(),
                  "service_4096_decoded": sum(row[str(CAP)] == "PASS" for row in rows)}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    setup()
    assets = OfficialAssets.verify(args.weights)
    reader, plans = inputs()
    metric = NativeQuality()
    outcomes = {}
    for split in ("calibration", "validation"):
        for name, profile in (("L0", LiteProfile(1, 1)), ("L1", AttenuatedIntra())):
            rows, summary = evaluate(assets, reader, plans[split], profile, name, metric)
            write_csv(args.output / f"{split}_{name}.csv", rows)
            outcomes.setdefault(split, {})[name] = summary
    write_json(args.output / "summary.json", {
        "cap": CAP, "profiles": outcomes, "read_counts": reader.counts,
        "candidate": AttenuatedIntra().identity(), "joint_base_rans": "native y and hyperprior z share one payload",
        "proper_base_reference": True, "replay_scope": "complete diagnostic Base; per-cap desync applied in ledger",
    })


if __name__ == "__main__":
    main()
