"""Measure the final Seed 2 symbols with frozen E0, E1 and E2 entropy backends."""

import argparse
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from hla_tok.evaluation.io import write_csv, write_json
from hla_tok.uf.anchor import BUDGETS
from hla_tok.uf.latent import LATENT_UNIT
from hla_tok.uf.official import OfficialAssets, sha256_file
from hla_tok.uf.wire import APP_BASE_OVERHEAD, MAX_FRAME_BYTES, RECORD_HEADER, RECORD_OVERHEAD
from hla_tok.uf_v2.records import RAW_FLAG, BatchRecords
from hla_tok.uf_v2.static_entropy import Histogram
from scripts.hla_tok.continuous.quality import NativeQuality, summarize
from scripts.hla_tok.uf.inputs import inputs
from scripts.hla_tok.uf_v2.calibrate import samples
from scripts.hla_tok.uf_v2.evaluate import allowance, summarize_timing, timing
from scripts.hla_tok.uf_v3.audit import checkpoints
from scripts.hla_tok.uf_v3.endpoint import GEOMETRY, load_v2, parse_base, standard_decoder
from scripts.hla_tok.v2.protocol import json_at, setup

TAG = b"entropy3"
CUTS = (2, 5, 10, 20, 40, "ALL_VALID_RECORDS")


def ordered_headers(family):
    return [RECORD_HEADER.unpack_from(family.raw, start)[:3] for start in family.ends[:-1]]


def canvas_at(bank, base, shape, family, stop):
    start = perf_counter()
    raw = family.raw[family.base_end:family.ends[stop]]
    canvas = bank.canvas(shape, base.info.intra)
    used, decoded = canvas.decode(bytearray(raw))
    if (used, decoded) != (len(raw), stop):
        raise ValueError("backend did not decode exactly the selected integer record support")
    return canvas, (perf_counter() - start) * 1000


def byte_row(family, stop, headers):
    size = family.ends[stop]
    modes = [mode for _, _, mode in headers[:stop]]
    rans = sum(mode < RAW_FLAG for mode in modes)
    return {"bytes": size, "base_bytes": family.base_end, "base_header": APP_BASE_OVERHEAD,
            "native_y_z_bytes": family.base_end - APP_BASE_OVERHEAD, "records": stop,
            "enhancement_header_crc_bytes": RECORD_OVERHEAD * stop, "mode_field_bytes_in_header": stop,
            "rans_termination_bytes": 4 * rans, "raw_records": stop - rans,
            "enhancement_payload_bytes": size - family.base_end - RECORD_OVERHEAD * stop}


def same_symbol(codec, banks, decoder, base, symbols, families, headers, pack_ms, budget, cap, decoded):
    _, stop = families["E0"].prefix(cap)
    shape = tuple(symbols.shape[1:])
    step = decoded.scale.float() * codec.steps[base.info.intra] * LATENT_UNIT
    canvas, rows, ref_image, delta = None, [], None, None
    for name, bank in banks.items():
        family = families[name]
        current, elapsed = canvas_at(bank, base, shape, family, stop)
        current_delta = current.delta(step)
        image = decoder.display(decoded, codec.refiner, current_delta)
        if canvas is not None:
            if not np.array_equal(canvas.symbols, current.symbols) or not np.array_equal(canvas.levels, current.levels):
                raise ValueError("CDF altered the integer input or support")
            if not torch.equal(delta, current_delta) or not torch.equal(ref_image, image):
                raise ValueError("equal integers did not reproduce latent and RGB")
        else:
            canvas, delta, ref_image = current, current_delta, image
        rows.append({"backend": name, "budget": budget, "same_support_latent_rgb": True,
                     "decode_ms": elapsed, "all_records_pack_ms": pack_ms[name], **byte_row(family, stop, headers[name])})
    return rows


def measure(codec, reader, clips, banks, out, limit):
    decoder, metric = standard_decoder(codec), NativeQuality()
    same, rate = [], []
    last = None
    unreachable = set()
    for video, index, rgb, base, symbols in samples(codec.native, codec.model, reader, clips):
        # The V2 generator stays Train-only, even when asked to fit an independent replay.
        if video != last:
            decoder.reset()
            unreachable.clear()
        if index >= limit:
            continue
        last = video
        families, pack_ms = {}, {}
        for name, bank in banks.items():
            start = perf_counter()
            families[name] = bank.pack_symbols(base, TAG, symbols, MAX_FRAME_BYTES)
            pack_ms[name] = (perf_counter() - start) * 1000
        headers = {name: ordered_headers(family) for name, family in families.items()}
        support = [row[:2] for row in headers["E0"]]
        if any(family.raw[:family.base_end] != families["E0"].raw[:families["E0"].base_end]
               or [row[:2] for row in headers[name]] != support for name, family in families.items()):
            raise ValueError("CDF altered the Base or order of eligible records")
        decoded = decoder.shared_base(parse_base(families["E0"], TAG))
        if not torch.equal(decoder.decoded_latent(), codec.native.proxy(base.info.intra).ufv2_encode_snapshot()[1]):
            raise ValueError("CDF replay did not retain exact Base state")
        metric.prepare(rgb)
        step = decoded.scale.float() * codec.steps[base.info.intra] * LATENT_UNIT
        stem = {"video": video, "index": index, "type": "I" if base.info.intra else "P"}
        quality = {}
        for budget in CUTS:
            cap = MAX_FRAME_BYTES if budget == "ALL_VALID_RECORDS" else allowance(budget)
            if cap < families["E0"].base_end or (not base.info.intra and budget in unreachable):
                status = "BASE_BUDGET_EXCEEDED" if cap < families["E0"].base_end else "DESYNC"
                rate.extend(stem | {"backend": name, "budget": budget, "status": status} for name in banks)
                unreachable.add(budget)
                continue
            unreachable.discard(budget)
            same.extend(stem | row for row in same_symbol(
                codec, banks, decoder, base, symbols, families, headers, pack_ms, budget, cap, decoded))
            for name, bank in banks.items():
                family = families[name]
                raw, stop = family.prefix(cap)
                if stop not in quality:
                    current, _ = canvas_at(bank, base, tuple(symbols.shape[1:]), family, stop)
                    image = decoder.display(decoded, codec.refiner, current.delta(step)) if stop else decoded.rgb
                    quality[stop] = metric.score(image)
                rate.append(stem | {"backend": name, "budget": budget, "bytes": len(raw), "records": stop, "status": "PASS",
                                    "unused": cap - len(raw) if budget != "ALL_VALID_RECORDS" else None,
                                    **quality[stop]})
    write_csv(out / "same_symbol.csv", same)
    write_csv(out / "same_rate.csv", rate)
    return same, rate


def aggregate(same, rate):
    by_symbol, by_rate = defaultdict(list), defaultdict(list)
    for row in same:
        by_symbol[row["backend"], row["budget"]].append(row)
    for row in rate:
        by_rate[row["backend"], row["budget"]].append(row)
    result = {}
    for (backend, budget), rows in by_rate.items():
        symbol = by_symbol[backend, budget]
        if len(symbol) != len(rows):
            result.setdefault(backend, {})[budget] = {"status": "BASE_BUDGET_EXCEEDED",
                                                       "decoded": len(symbol), "planned": len(rows)}
            continue
        base = sum(row["base_bytes"] for row in symbol)
        total, e0 = (sum(row["bytes"] for row in group) for group in (symbol, by_symbol["E0", budget]))
        result.setdefault(backend, {})[budget] = {
            "same_symbol": {"bytes": total, "base_bytes": base, "whole_savings": 1 - total / e0,
                            "enhancement_savings": 1 - (total - base) / (e0 - base) if e0 > base else None,
                            "verified_frames": len(symbol), "raw_records": sum(row["raw_records"] for row in symbol)},
            "same_rate": summarize(rows) | {"actual_mbps": sum(row["bytes"] for row in rows) * 240 / 1e6 / len(rows),
                                            "records": sum(row["records"] for row in rows)}}
    return result


def choose(result, timings):
    for backend in ("E2", "E1"):
        if backend not in timings:
            continue
        if backend == "E2" and not result["E2"]["ALL_VALID_RECORDS"]["same_symbol"]["raw_records"]:
            continue
        adequate = all(t["pooled"]["fps"] >= 30 and t["pooled"]["total_ms_p95"] <= 33.33
                       for t in timings[backend].values())
        similar = all(result[backend][b]["same_rate"]["global_psnr"] >= result["E0"][b]["same_rate"]["global_psnr"] - 0.05
                      and result[backend][b]["same_rate"]["lpips"] <= result["E0"][b]["same_rate"]["lpips"] + 0.005
                      for b in BUDGETS)
        if adequate and similar and result[backend][40]["same_symbol"]["whole_savings"] >= 0.03:
            return backend
    return "E0"


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    setup()
    assets = OfficialAssets.verify(args.weights)
    reader, plans = inputs()
    clips = plans["calibration"]
    path = checkpoints(args.v2, 2)[20000]
    codec = load_v2(assets, path)
    histogram = Histogram(GEOMETRY.channels)
    for _, _, _, base, symbols in samples(codec.native, codec.model, reader, clips):
        histogram.observe(base.info.intra, symbols)
    asset = histogram.fit()
    write_json(args.output / "static_cdf.json", asset.as_json())
    scales = tuple(codec.model.scales().tolist())
    banks = {"E0": codec.bank, "E1": BatchRecords(scales, GEOMETRY, asset.tables),
             "E2": BatchRecords(scales, GEOMETRY, asset.tables, raw=True)}
    same, rate = measure(codec, reader, clips, banks, args.output, args.limit)
    del codec
    result = aggregate(same, rate)
    times = {}
    for backend in banks if args.repetitions else ():
        candidate = load_v2(assets, path, static=None if backend == "E0" else asset, raw=backend == "E2")
        rows = timing(candidate, assets, reader, clips, tuple(BUDGETS), args.repetitions, args.limit, False)
        write_csv(args.output / f"{backend}_timing.csv", rows)
        times[backend] = summarize_timing(rows, tuple(BUDGETS), args.repetitions)
        del candidate
        torch.cuda.empty_cache()
    write_json(args.output / "summary.json", {
        "split": "Train Calibration", "checkpoint_sha256": sha256_file(path), "read_counts": reader.counts,
        "cdf": asset.as_json(), "measurements": result, "full_max": times, "default": choose(result, times),
        "gate": "40 Mbps same-symbol savings >=3%; no >0.05 dB or >0.005 LPIPS regression at same cap; Full-max passes",
        "historical_reason": json_at(args.v2 / "results/hla_tok/uf_v2/final_verdict.json")["selected_research_endpoint"],
        "table_bytes_preinstalled": True,
    })


if __name__ == "__main__":
    main()
