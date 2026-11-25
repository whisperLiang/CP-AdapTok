"""Decode recorded HLA-Tok-C packets with independently loaded synthesis weights."""

import argparse
import json
import os
from hashlib import sha256
from pathlib import Path

import torch
from torch import Tensor

from hla_tok.models.local import LocalTokenDecoder
from hla_tok.models.spatial import RGBDecoder
from hla_tok.streaming.continuous import ContinuousReceiver
from hla_tok.streaming.continuous_wire import FrameReceiver, deserialize_continuous_tokens
from scripts.hla_tok.continuous.matrix import compute
from scripts.hla_tok.continuous.ordering import load_orders
from scripts.hla_tok.v2.protocol import setup


def compute_values(values: Tensor, precision: str) -> Tensor:
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[precision]
    return values.to(dtype)


def tensor_digest(values: Tensor) -> str:
    data = values.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return sha256(data).hexdigest()


@torch.inference_mode()
def decode_files(weights: Path, ordering: Path, files: list[Path], precision: str) -> dict:
    setup()
    synthesis = ContinuousReceiver(LocalTokenDecoder(), RGBDecoder("D0"))
    synthesis.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    synthesis.eval().requires_grad_(False).to(device="cuda", memory_format=torch.channels_last)
    payloads = [path.read_bytes() for path in files]
    first = deserialize_continuous_tokens(payloads[0])
    assembly = FrameReceiver(first.frame_id, first.dtype, first.order)
    for raw in payloads:
        assembly.accept(raw)
    received = assembly.commit()
    if received is None:
        raise ValueError("frame lacks a deliverable contiguous base prefix")
    order, = (item for item in load_orders(ordering) if item.name == first.order)
    with compute(precision):
        output = synthesis.decode(compute_values(received.cuda(), precision), order)
    return {"pid": os.getpid(), "frame_id": first.frame_id, "tokens": received.shape[1],
            "bytes": sum(map(len, payloads)), "rgb_shape": list(output.shape),
            "rgb_sha256": tensor_digest(output), "received_tokens_sha256": tensor_digest(received),
            "modules": list(dict(synthesis.named_children())), "wire_dtype": first.dtype.value,
            "compute_dtype": precision}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True, help="strict synthesis-only state_dict")
    parser.add_argument("--ordering", type=Path, required=True, help="directory with the frozen Rank manifest")
    parser.add_argument("--precision", choices=("bf16", "fp32"), required=True)
    parser.add_argument("packets", type=Path, nargs="+")
    args = parser.parse_args()
    print(json.dumps(decode_files(args.weights, args.ordering, args.packets, args.precision)), flush=True)


if __name__ == "__main__":
    main()
