import sys
from hashlib import sha256
from pathlib import Path

import torch

from experiments.pht_tok.closed import official_encode
from experiments.pht_tok.study import SKIP, Source, equivalence, write_json
from hla_tok.hts.native import NativeNetworks
from hla_tok.hts.units import NativeSchedule
from hla_tok.uf.official import NativeNAL
from pht_tok.bitstream import Mode, StreamReader
from pht_tok.codec import PHTDecoder, PHTEncoder, ReferenceKey, model_identifier
from pht_tok.official_adapter import Reference
from scripts.hla_tok.hts.audit_native import source_clips
from scripts.hla_tok.uf.inputs import inputs
from third_party.DCVC.src.utils.common import set_torch_env
from third_party.DCVC.src.utils.transforms import rgb2ycbcr

def require(condition, field):
    if not condition:
        raise AssertionError(field)

@torch.inference_mode()
def run(weights, work):
    set_torch_env()
    with torch.cuda.stream(torch.cuda.Stream()):
        reader, plans = inputs()
        sources = [Source(item.clip.video, item.fps, item.rgb, rgb2ycbcr(item.rgb).half() - 0.5)
                   for item in source_clips(reader, plans["validation"])]
        require(reader.counts == {"train": 0, "validation": 174, "test": 0}, "Validation read count")
        native = equivalence(sources, [32], weights, work)
        write_json(work / "native_equivalence.json", native | {"reads": reader.counts})
        identification = model_identifier(weights, SKIP)
        stock = NativeNetworks(weights, "hts", SKIP)
        tx = PHTEncoder(NativeNetworks(weights, "hts", SKIP), identification)
        rx = PHTDecoder(NativeNetworks(weights, "hts", SKIP), identification)
        fresh = PHTDecoder(NativeNetworks(weights, "hts", SKIP), identification)
        entries = []
        for source in sources:
            units = list(NativeSchedule(8, reset_interval=128).units(len(source.rgb)))
            writer, parser = NativeNAL(), NativeNAL()
            streams = [official_encode(stock, source, unit, 32, writer) for unit in units]
            for peer in (tx, rx, fresh):
                peer.key = None
            anchor = tx.encode_intra(source.yuv[:1], 32, 0)
            for peer in (rx, fresh):
                peer.decode_prefix(anchor.data)
            native_anchor = stock.decode(parser.decode(streams[0], 0))[0]
            require(torch.equal(fresh.session.reconstruct()[0], native_anchor[..., :1080, :1920]),
                    "I-frame pixels")
            entries.append({"source": source.name, "frame": 0, "native_bytes": len(streams[0]),
                            "mode": "INTRA", "bytes": list(anchor.prefix_boundaries),
                            "sha256": sha256(anchor.data).hexdigest()})
            for last, unit, wire in zip(units[:-1], units[1:], streams[1:], strict=True):
                before = Reference(last.intra or last.reset,
                                   tuple(stock.video.proxy.audit_reference(last.intra or last.reset)))
                for peer in (tx, rx, fresh):
                    peer.adapter.commit_reference(before)
                    peer.key = ReferenceKey(last.start, 0, 1080, 1920)
                expected = [frame.clone() for frame in stock.decode(parser.decode(wire, unit.start))]
                z, y, *_ = stock.video.proxy.pht_symbols(False)
                for mode in (Mode.MASK, Mode.PRECISION):
                    stream = tx.encode_chunk_once(source.yuv[unit.start:unit.start + unit.valid],
                                                  32, unit.start, 2, mode=mode, commit=False, reset=unit.reset)
                    session = rx.begin_chunk(commit=False)
                    previous = 0
                    for boundary in stream.prefix_boundaries:
                        prefix = stream.truncate_to_budget(boundary)
                        require(prefix == stream.data[:boundary] and boundary > previous, "physical prefix")
                        session.append_bytes(stream.data[previous:boundary])
                        session.decode_new_layers()
                        output = session.reconstruct()
                        standalone = fresh.decode_prefix(prefix, commit=False).reconstruct()
                        require(all(torch.equal(a, b) for a, b in zip(output, standalone, strict=True)),
                                "incremental and independent pixels")
                        previous = boundary
                    require(torch.equal(session.residuals(), y), "physical Y integers")
                    template = rx.adapter.proxy.pht_symbols(False)[0]
                    hyper = StreamReader().append_bytes(stream.data)[0]
                    require(torch.equal(rx.entropy.decode_z(hyper, 32, tuple(template.shape), template.device), z),
                            "physical Z integers")
                    require(all(torch.equal(a, b[..., :1080, :1920])
                                for a, b in zip(output, expected[:unit.valid], strict=True)), "Full pixels")
                    entries.append({"source": source.name, "frame": unit.start, "native_bytes": len(wire),
                                    "mode": mode.name, "bytes": list(stream.prefix_boundaries),
                                    "sha256": sha256(stream.data).hexdigest()})
            print("frozen fixed reference", source.name, flush=True)
        write_json(work / "prefix_validation.json", {"reads": reader.counts, "entries": entries})

if __name__ == "__main__":
    run(Path(sys.argv[1]), Path(sys.argv[2]))
