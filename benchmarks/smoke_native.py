"""Bit-exact smoke on the same sealed-manifest Validation frames."""
import argparse
import json
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from baselines.dcvc_uf.native import NativeNetworks
from baselines.dcvc_uf.official import NativeNAL
from baselines.dcvc_uf.units import NativeSchedule
from benchmarks.pht_tok.closed import run_closed, summarized_speed
from benchmarks.pht_tok.network import WireCodec, cases, rate_models, simulate
from benchmarks.pht_tok.study import Quality, Source, equivalence, fixed_point, uvg_source
from common.inputs import Clip, inputs, source_clips
from common.integrity import tensor_digest
from common.performance import timed
from third_party.DCVC.src.utils.common import set_torch_env
from third_party.DCVC.src.utils.transforms import rgb2ycbcr

VIDEO = "DAVIS2017:car-roundabout"
COUNT = 17

def speed(rows):
    enc, dec, count = map(np.array, zip(*rows, strict=True))
    summary = {"frames": int(count.sum())}
    for key, values in {"encode": enc, "decode": dec, "serial": enc + dec}.items():
        summary[key + "_fps"] = count.sum() * 1000 / values.sum()
        summary[key + "_p50_ms"] = np.percentile(values, 50)
        summary[key + "_p95_ms"] = np.percentile(values, 95)
    return summary

def source():
    reader, _ = inputs()
    clip = Clip("validation", VIDEO, 0, COUNT)
    item, = source_clips(reader, [clip])
    details = clip.manifest(reader) | {"test_pixel_reads": reader.counts["test"]}
    return Source(clip.video, item.fps, item.rgb, rgb2ycbcr(item.rgb).half() - 0.5), details

def official(args, frames):
    variant = args.variant
    runs, timings = [], []
    loader_start = perf_counter()
    tx, rx = [NativeNetworks(args.weights, variant, 0.15) for _ in range(2)]
    torch.cuda.synchronize()
    load_ms = (perf_counter() - loader_start) * 1000
    from inference_extensions_cuda import DMCIProxy
    rx.intra.proxy = DMCIProxy()
    rx.intra.proxy.set_param(rx.intra.add_cdf_to_state_dict(rx.intra.state_dict()), 0.15)
    metric = Quality()
    units = list(NativeSchedule(1 if variant == "ld" else 8, reset_interval=128).units(COUNT))
    torch.cuda.reset_peak_memory_stats()
    for qp in (16, 32):
        checks = []
        for trial in range(4):
            writer, reader = NativeNAL(), NativeNAL()
            for i, unit in enumerate(units):
                (base, encode_ms) = timed(tx.encode, frames.rgb, unit, qp)
                wire = writer.encode(base)
                assert wire
                decoded, decode_ms = timed(lambda: rx.decode(reader.decode(wire, unit.start)))
                assert len(decoded) == unit.coded
                if trial == 0:
                    scores = metric.score(frames.rgb[unit.start:unit.start+unit.valid], decoded[:unit.valid])
                    checks.append({"shape": list(decoded[0].shape), "bytes": len(wire), "sha256": sha256(wire).hexdigest(),
                                   "decoded_sha256": tensor_digest(decoded), "quality": scores})
                elif i:
                    timings.append((encode_ms, decode_ms, unit.valid))
        runs.append({"qp": qp, "units": checks})
    return {"checks": runs, "timing": speed(timings) | {"load_ms": load_ms, "peak_bytes": torch.cuda.max_memory_allocated()}}

def pht(args, frames):
    from pht_tok.bitstream import Mode
    from pht_tok.codec import PHTDecoder, PHTEncoder, model_identifier

    def native():
        return NativeNetworks(args.weights, "hts", 0.15)

    begin = perf_counter()
    model = model_identifier(args.weights, 0.15)
    stock, tx, rx = native(), PHTEncoder(native(), model), PHTDecoder(native(), model)
    torch.cuda.synchronize()
    load_ms = (perf_counter() - begin) * 1000
    equivalent = equivalence([frames], [32], args.weights, args.output.parent)
    quality = Quality()
    fixed = fixed_point(frames, 32, stock, tx, rx, quality, None)
    stable_fixed = [{k: v for k, v in row.items() if not k.endswith("_ms")} for row in fixed]
    units = list(NativeSchedule(8, reset_interval=128).units(COUNT))
    checks = []
    for mode in (Mode.MASK, Mode.PRECISION):
        tx.key = rx.key = None
        anchor = tx.encode_intra(frames.yuv[:1], 32, 0)
        rx.decode_prefix(anchor.data)
        for unit in units[1:]:
            wire = tx.encode_chunk_once(frames.yuv[unit.start:unit.start+unit.valid], 32, unit.start, 2,
                                        reset=unit.reset, mode=mode)
            prefix = wire.prefix_boundaries
            session = rx.begin_chunk()
            start, stage_rows = 0, []
            for stage, end in enumerate(prefix, 1):
                session.append_bytes(wire.data[start:end])
                session.decode_new_layers()
                fresh = PHTDecoder(native(), model)
                if unit == units[1]:
                    fresh.decode_prefix(anchor.data)
                    direct = fresh.decode_prefix(wire.data[:end])
                    assert tensor_digest(direct.reconstruct()) == tensor_digest(session.reconstruct())
                if stage >= 2:
                    assert tx.adapter.reference.digest() == rx.adapter.reference.digest()
                stage_rows.append({"bytes": end, "sha256": sha256(wire.data[:end]).hexdigest(),
                                   "decoded_sha256": tensor_digest(session.reconstruct())})
                start = end
            checks.append({"mode": mode.name, "start": unit.start, "stages": stage_rows,
                           "reference": rx.adapter.reference.digest(), "entropy_decodes": session.entropy_decodes})
    uvg = uvg_source(args.uvg, 129)
    loss = run_closed(uvg, 32, tx, rx, Mode.MASK, 2, "lost_reference")
    assert loss["p_chunks"] == 16
    assert any(row.get("refresh") for row in loss["per_unit"])
    assert any(row.get("reference_rejected") for row in loss["per_unit"])
    trials = []
    for _ in range(4):
        trial = run_closed(uvg, 32, tx, rx, Mode.MASK, 2, "full")
        trials.append(trial["per_unit"])
    perf = [summarized_speed(trial, uvg.fps) for trial in trials[1:]]
    peak = {"peak_bytes": torch.cuda.max_memory_allocated(), "load_ms": load_ms}
    codec = WireCodec(args.weights)
    profile = json.loads(args.profile.read_text())
    # Use the frozen H100 scheduler inputs for deterministic replay.
    fixed_rates = json.loads(args.rates.read_text())
    scenario = next(c for c in cases(args.trace) if c.name.startswith("Verizon"))
    def stable(row):
        if isinstance(row, dict):
            return {k: stable(v) for k, v in row.items() if k not in {"encode_ms", "decode_ms"}}
        if isinstance(row, list):
            return [stable(v) for v in row]
        return row
    network = [stable(simulate(codec, frames, policy, scenario, rate_models(fixed_rates), quality, profile))
               for policy in ("S1_adaptive_qp", "S4_adaptive_prefix")]
    return {"checks": {"equivalence": equivalent, "prefixes": checks, "fixed": stable_fixed,
                       "loss": stable(loss["per_unit"]), "network": network},
            "timing": peak | {"repetitions": perf}}

def main():
    parser = argparse.ArgumentParser()
    for key in ("weights", "uvg", "output", "rates", "profile", "trace"):
        parser.add_argument("--" + key, type=Path)
    parser.add_argument("variant", choices=("ld", "hts", "pht"))
    args = parser.parse_args()
    set_torch_env()
    with torch.inference_mode(), torch.cuda.stream(torch.cuda.Stream()):
        clip, metadata = source()
        result = pht(args, clip) if args.variant == "pht" else official(args, clip)
    args.output.write_text(json.dumps(result | {"input": metadata, "gpu": torch.cuda.get_device_name(),
                                                "torch": torch.__version__}, indent=2, allow_nan=False) + "\n")

if __name__ == "__main__":
    main()
