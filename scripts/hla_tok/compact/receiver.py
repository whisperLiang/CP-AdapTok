"""Isolated Compact receiver that opens only synthesis weights, assets and packets."""

import argparse
import json
import os
from pathlib import Path

import torch

from hla_tok.compression.profiles import PROFILES_BY_LABEL
from hla_tok.evaluation.metrics import sha256_file
from hla_tok.models.local import LocalTokenDecoder
from hla_tok.models.spatial import RGBDecoder
from hla_tok.streaming.compact_wire import FrameAssembly, deserialize
from hla_tok.streaming.continuous import ContinuousReceiver
from scripts.hla_tok.compact.assets import load_asset
from scripts.hla_tok.compact.inputs import compact_orders
from scripts.hla_tok.continuous.matrix import compute
from scripts.hla_tok.continuous.receiver import tensor_digest
from scripts.hla_tok.v2.protocol import json_at, setup


@torch.inference_mode()
def decode_requests(root: Path) -> dict:
    setup()
    torch.use_deterministic_algorithms(True)
    request = json_at(root / "request.json")
    weights = root / "synthesis.pt"
    if request["synthesis_sha256"] != sha256_file(weights):
        raise ValueError("synthesis artifact identity differs")
    receiver = ContinuousReceiver(LocalTokenDecoder(), RGBDecoder("D0"))
    receiver.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    receiver.eval().requires_grad_(False).to(device="cuda", memory_format=torch.channels_last)
    orders = compact_orders()
    decoded = []
    for job in request["jobs"]:
        spec = job["asset"]
        codec, _ = load_asset(root / spec["filename"], spec, PROFILES_BY_LABEL[spec["profile"]],
                              orders[job["order"]], request["model_state_sha256"])
        packets = [(root / filename).read_bytes() for filename in job["packets"]]
        recipient = FrameAssembly(codec.identity, deserialize(packets[0]).frame_id)
        contiguous = [recipient.accept(raw) for raw in packets]
        prefix = recipient.commit()
        if prefix is None:
            raise ValueError("missing base prefix")
        with compute(job["compute_dtype"]):
            image = codec.decode_payload(receiver, prefix)
        decoded.append({"tag": job["tag"], "contiguous": contiguous, "rgb_shape": list(image.shape),
                        "rgb_sha256": tensor_digest(image), "bytes": sum(map(len, packets))})
    return {"pid": os.getpid(), "synthesis_modules": list(dict(receiver.named_children())), "decoded": decoded}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    print(json.dumps(decode_requests(parser.parse_args().folder)), flush=True)


if __name__ == "__main__":
    main()
