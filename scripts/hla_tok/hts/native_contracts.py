"""Cross-check the audited codec against the pinned official test_video.py."""

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch
from torch import Tensor

from hla_tok.evaluation.io import write_json
from hla_tok.hts.native import NativeNetworks, restore_native_rgb, tensor_digest
from hla_tok.hts.units import NativeSchedule
from hla_tok.uf.official import NativeNAL
from scripts.hla_tok.hts.audit_native import Point
from scripts.hla_tok.uf.inputs import inputs
from scripts.hla_tok.v2.protocol import ROOT
from third_party.DCVC.src.utils.common import set_torch_env
from third_party.DCVC.src.utils.transforms import ycbcr2rgb


class ArrayFrames:
    def __init__(self, frames):
        self.frames = iter(frames)
        self.reads = 0

    def read_one_frame(self):
        self.reads += 1
        return next(self.frames)

    def close(self):
        pass


def official_entry():
    sys.path.insert(0, str(ROOT / "third_party/DCVC"))
    return importlib.import_module("test_video")


def direct_replay(official: Any, models: NativeNetworks, rgb: Tensor, point: Point, reset: int, directory: Path):
    frames = rgb.mul(255).round().byte().cpu().numpy()
    frame_readers: list[ArrayFrames] = []
    observations: dict[str, list[str]] = {"rgb": [], "context": [], "reference": []}
    original_p = models.video.decompress
    seed = models.video.add_ref_feature_from_frame
    distortion = official.get_distortion

    def capture_i(frame, apply_feature_adaptor=True):
        seed(frame, apply_feature_adaptor)
        if not apply_feature_adaptor:
            observations["reference"].append(tensor_digest(models.video.proxy.audit_reference(True)))

    def capture_p(payload, sps, qp, ec_part, memory_reset):
        result = original_p(payload, sps, qp, ec_part, memory_reset)
        observations["context"].append(tensor_digest(models.video.proxy.audit_context()))
        observations["reference"].append(tensor_digest(models.video.proxy.audit_reference(bool(memory_reset))))
        return result

    def capture_rgb(args, raw, y, u, v, original):
        observations["rgb"].append(tensor_digest([ycbcr2rgb(raw + 0.5).float()]))
        return distortion(args, raw, y, u, v, original)

    def new_reader(_):
        reader = ArrayFrames(frames)
        frame_readers.append(reader)
        return reader

    directory.mkdir(parents=True, exist_ok=True)
    height, width = rgb.shape[-2:]
    stream = directory / "official.bin"
    args = {
        "check_existing": False, "frame_num": len(rgb), "save_decoded_frame": False,
        "verbose": 0, "reset_interval": reset, "intra_period": -1, "verbose_json": False,
        "src_height": height, "src_width": width, "src_type": "png",
        "qp_i": point.qp_i, "qp_p": point.qp_p, "force_intra": False, "calc_ssim": False,
        "curr_bin_path": stream, "curr_json_path": directory / "upstream.json",
    }
    with (
        patch.object(official, "g_frame_delay", 8),
        patch.object(official, "get_src_reader", new_reader),
        patch.object(official, "get_distortion", capture_rgb),
        patch.object(models.video, "add_ref_feature_from_frame", capture_i),
        patch.object(models.video, "decompress", capture_p),
    ):
        official.run_one_point_with_stream(models.video, models.intra, args)
    assert [reader.reads for reader in frame_readers] == [len(rgb), len(rgb)]
    return stream.read_bytes(), observations


def wrapped_replay(models: NativeNetworks, rgb: Tensor, point: Point, reset: int) -> tuple[list[bytes], dict]:
    writer, reader = NativeNAL(), NativeNAL()
    units = list(NativeSchedule(8, reset_interval=reset).units(len(rgb)))
    packets, encode_context = [], []
    for unit in units:
        encoded = models.encode(rgb, unit, point.qp(unit))
        packets.append(writer.encode(encoded))
        encode_context.append(tensor_digest(models.video.proxy.audit_context()))
    height, width = rgb.shape[-2:]
    observations: dict[str, list[str]] = {"rgb": [], "context": [], "reference": []}
    for unit, packet in zip(units, packets, strict=True):
        base = reader.decode(packet, unit.start)
        yuv = models.decode(base)
        result = restore_native_rgb(yuv, unit, height, width)
        observations["rgb"].extend(tensor_digest([image]) for image in result.split(1))
        observations["reference"].append(models.reference_hash(unit))
        if not unit.intra:
            observations["context"].append(tensor_digest(models.video.proxy.audit_context()))
    if len(units) > 1 and observations["context"] != encode_context[:-1]:
        raise RuntimeError("official encoder and decoder use different context on the same dependency edge")
    return packets, observations


def parameters_digest(models: NativeNetworks) -> str:
    return tensor_digest([value for net in (models.intra, models.video) for value in net.state_dict().values()])


def compare_case(official, models: NativeNetworks, rgb: Tensor, point: Point, directory: Path,
                 reset_interval: int = 32) -> dict:
    original, reference = direct_replay(official, models, rgb, point, reset_interval, directory)
    packets, observed = wrapped_replay(models, rgb, point, reset_interval)
    return {"frames": len(rgb), "units": len(packets), "bytes": len(original),
            "payload_and_nal_parity": b"".join(packets) == original,
            "rgb_parity": observed["rgb"] == reference["rgb"],
            "context_parity": observed["context"] == reference["context"],
            "reference_parity": observed["reference"] == reference["reference"]}


def causality_probe(models: NativeNetworks, rgb: Tensor, point: Point) -> tuple[bool, bool]:
    original, original_state = wrapped_replay(models, rgb, point, 32)
    disturbed = torch.cat((rgb[:17], 1 - rgb[17:]))
    changed, changed_state = wrapped_replay(models, disturbed, point, 32)
    independent_i, i_state = wrapped_replay(models, rgb[:1], point, 32)
    causal = (original[:3] == changed[:3] and original[3:] != changed[3:]
              and original_state["rgb"][:17] == changed_state["rgb"][:17]
              and original_state["reference"][:3] == changed_state["reference"][:3])
    standalone = (independent_i == original[:1] and i_state["rgb"] == original_state["rgb"][:1]
                  and i_state["reference"] == original_state["reference"][:1])
    return causal, standalone


def contract_cases(models: NativeNetworks, rgb: Tensor, directory: Path) -> dict:
    official = official_entry()
    point = Point("diagnostic", 7, 19)
    frozen = parameters_digest(models)
    checks = [compare_case(official, models, rgb[:length], point, directory / str(length))
              for length in (*range(1, 9), 32)]
    causal, standalone = causality_probe(models, rgb, point)
    immutable = parameters_digest(models) == frozen
    parity = all(all(value for key, value in row.items() if key.endswith("_parity")) for row in checks)
    return {"cases": checks, "chunk_causality": causal, "independent_i": standalone,
            "frozen_weight_and_buffer_sha256": frozen, "weights_and_buffers_unchanged": immutable,
            "verdict": "PASS" if causal and standalone and immutable and parity else "FAIL"}


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    set_torch_env()
    reader, plans = inputs()
    clip = plans["calibration"][0]
    frames = [reader.read(clip.split, clip.video, clip.start + i) for i in range(clip.length)]
    rgb = torch.stack(frames).cuda().float().div_(255)
    with torch.cuda.stream(torch.cuda.Stream()):
        trace = contract_cases(NativeNetworks(args.weights, "hts", 0.15), rgb, args.output / "official_streams")
    write_json(args.output / "native_correctness.json", trace | {"pixels_read": reader.counts})
    print(trace["verdict"], flush=True)
    if trace["verdict"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
