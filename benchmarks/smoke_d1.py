import argparse
import json
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import torch

from baselines.dcvc_uf.official import OfficialAssets
from baselines.hla_tok_uf_v2.evaluate import allowance, encode, load_codec, receive, summarize_timing, timing
from common.inputs import Clip, inputs, setup, source_frame
from common.integrity import tensor_digest
from common.metrics import NativeQuality


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("weights", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    setup()
    reader, _ = inputs()
    clip = Clip("validation", "DAVIS2017:car-roundabout", 0, 17)
    assets = OfficialAssets.verify(args.weights)
    begin = perf_counter()
    codec = load_codec(args.checkpoint, assets)
    torch.cuda.synchronize()
    load_ms = (perf_counter() - begin) * 1000
    budgets = (5, 10, 20, 40)
    torch.cuda.reset_peak_memory_stats()
    phases = timing(codec, assets, reader, [clip], budgets, 3, clip.length, False)
    perf = summarize_timing(phases, budgets, 3) | {"peak_bytes": torch.cuda.max_memory_allocated(), "load_ms": load_ms}
    metric = NativeQuality()
    codec.reset()
    peers = {b: codec.new_decoder() for b in (0, *budgets, -1)}
    checks = []
    for idx in range(clip.length):
        rgb = source_frame(reader, clip, idx)
        metric.prepare(rgb)
        encoded = encode(codec, rgb, idx)
        family = encoded.family
        row = {"index": idx, "full_sha256": sha256(family.raw).hexdigest(), "full_bytes": len(family.raw), "points": {}}
        base = None
        for budget in (0, *budgets):
            cap = allowance(budget) if budget else family.base_end
            prefix, count = family.prefix(cap)
            assert prefix == family.raw[:len(prefix)]
            result, stats = receive(codec, peers[budget], family, cap)
            assert result is not None, stats
            receiver, image = result
            base = receiver.base if base is None else base
            assert torch.equal(base, receiver.base)
            assert codec.matches(encoded, peers[budget], receiver)
            assert image.shape == (1, 3, 1080, 1920)
            row["points"][budget] = {"bytes": len(prefix), "sha256": sha256(prefix).hexdigest(), "records": count,
                                     "shape": list(image.shape), "quality": metric.score(image),
                                     "reference": tensor_digest([peers[budget].decoded_latent()]),
                                     "decoded_sha256": tensor_digest([image])}
            if budget == 40:
                direct = codec.receiver(peers[-1])
                direct.feed(prefix)
                assert direct.decode_available() == count + 1
                direct.finish()
                assert torch.equal(direct.render(), image)
                assert codec.matches(encoded, peers[-1], direct)
        checks.append(row)
    args.output.write_text(json.dumps({"input": clip.manifest(reader) | {"test_pixel_reads": reader.counts["test"]},
                                       "checks": checks, "timing": perf, "gpu": torch.cuda.get_device_name(),
                                       "torch": torch.__version__, "model_tag": codec.tag.hex()},
                                      indent=2, allow_nan=False) + "\n")

if __name__ == "__main__":
    main()
