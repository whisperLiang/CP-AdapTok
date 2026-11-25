"""Real native Compact prefix, causality, assembly and receiver isolation probes."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from hla_tok.compression.profiles import PROFILES_BY_LABEL
from hla_tok.evaluation.io import write_json
from hla_tok.evaluation.metrics import sha256_file
from hla_tok.streaming.compact import CompactCodec, encode_frames
from hla_tok.streaming.compact_wire import HEADER_BYTES, FrameAssembly, deserialize
from hla_tok.streaming.continuous import MAX_K, TAIL_REGIONS, ContinuousReceiver, encode_continuous
from scripts.hla_tok.compact.assets import load_asset
from scripts.hla_tok.compact.inputs import OUTPUT, compact_inputs, compact_orders
from scripts.hla_tok.continuous.contracts import prefix_plan
from scripts.hla_tok.continuous.inputs import Inputs, locked_output, run_arguments
from scripts.hla_tok.continuous.matrix import compute, representative, validation_frames
from scripts.hla_tok.continuous.receiver import tensor_digest
from scripts.hla_tok.v2.protocol import ROOT, json_at

RANGES = ((0, 510), (510, 6120), (6120, MAX_K))


def _prefixes(inputs: Inputs, codec: CompactCodec, source: torch.Tensor, budgets: list[int], raw_full) -> None:
    full_codes = codec.unpack(raw_full.payload)
    stride = codec.profile.bytes_per_token
    for count in budgets:
        encoded = codec.encode(inputs.model, source, count, 0)
        packet = deserialize(encoded.serialize_range(0, count))
        if not torch.equal(codec.unpack(packet.payload).view(torch.uint8), full_codes[:, :count].view(torch.uint8)):
            raise ValueError(f"integer or floating wire prefix changed for K{count}")
        if packet.payload != raw_full.payload[:count * stride] or packet.count != count:
            raise ValueError(f"Compact payload prefix changed for K{count}")


def _receive(codec: CompactCodec, receiver: ContinuousReceiver, frame) -> tuple[list[bytes], str]:
    packets = [frame.serialize_range(a, b) for a, b in RANGES]
    assembler = FrameAssembly(codec.identity, frame.frame_id)
    counts = [assembler.accept(packets[i]) for i in (2, 0, 1)]
    prefix = assembler.commit()
    if counts != [0, 510, MAX_K] or prefix != frame.payload:
        raise ValueError("out-of-order enhancement did not preserve a contiguous prefix")
    all_at_once = frame.serialize_range(0, MAX_K)
    if sum(map(len, packets)) != len(all_at_once) + 2 * HEADER_BYTES:
        raise ValueError("duplicate payload or missing incremental header bytes")
    one_shot = codec.decode(receiver, all_at_once)
    if prefix is None or not torch.equal(codec.decode_payload(receiver, prefix), one_shot):
        raise ValueError("reassembly changed receiver output")
    missing = FrameAssembly(codec.identity, frame.frame_id)
    missing.accept(packets[2])
    missing.accept(packets[0])
    if missing.commit() != frame.payload[:510 * codec.profile.bytes_per_token]:
        raise ValueError("receiver used information beyond a missing enhancement")
    return packets, tensor_digest(one_shot)


def _unreceived(codec: CompactCodec, inputs: Inputs, source: torch.Tensor) -> None:
    k = 510
    values = encode_continuous(inputs.model, source)
    ordered = codec.order.apply(values).clone()
    ordered[:, k:] = 3 * ordered[:, k:] + 1.5
    altered = codec.order.restore_full(ordered)
    first = codec.frame(codec.encode_tokens(values, k), 0)
    second = codec.frame(codec.encode_tokens(altered, k), 0)
    if first.payload != second.payload:
        raise ValueError("untransmitted continuous vectors affected the transmitted prefix")


def _causality(inputs: Inputs, receiver: ContinuousReceiver, codec: CompactCodec, frames: list[torch.Tensor]) -> dict:
    def stream(sources):
        requests = zip(sources, (510, MAX_K, 2, 6120), strict=True)
        for frame in encode_frames(inputs.model, codec, requests):
            packet = frame.serialize_range(0, frame.count)
            yield tensor_digest(codec.decode(receiver, packet)), tensor_digest(codec.unpack(frame.payload))

    sources = [frames[0], frames[1], frames[2], frames[0]]
    perturbed = [*sources[:2], 1 - sources[2], 1 - sources[3]]
    a, b = list(stream(sources)), list(stream(perturbed))
    if a[:2] != b[:2]:
        raise ValueError("future pixels changed an already produced frame")
    if a[2:] == b[2:]:
        raise ValueError("future-perturbation control did not change any perturbed output")
    for i, budget in enumerate((510, MAX_K, 2, 6120)):
        replay = codec.encode(inputs.model, sources[i], budget, i)
        digest = tensor_digest(codec.decode(receiver, replay.serialize_range(0, budget)))
        if digest != a[i][0]:
            raise ValueError("switching budget changed history-independent reconstruction")
    return {"budgets": [510, MAX_K, 2, 6120], "unchanged_prefix_frames": 2, "lookahead_frames": 0,
            "history": "none", "budget_switch_rgb_bit_exact": True}


def _prepare_request(root: Path, receiver: ContinuousReceiver, inputs: Inputs) -> dict:
    output = root / "synthesis.pt"
    torch.save(receiver.state_dict(), output)
    return {"synthesis_sha256": sha256_file(output), "model_state_sha256": inputs.checkpoint["state_dict_sha256"],
            "jobs": []}


def _record_job(root: Path, request: dict, output: Path, spec: dict, keys: dict, packets: list[bytes]) -> None:
    if not (root / spec["filename"]).is_file():
        shutil.copyfile(output / "profiles" / spec["filename"], root / spec["filename"])
    tag = "_".join(keys.values())
    paths = []
    for i in (2, 0, 1):
        path = root / f"{tag}_{i}.packet"
        path.write_bytes(packets[i])
        paths.append(path.name)
    request["jobs"].append(keys | {"tag": tag, "asset": spec, "packets": paths})


def _independent(root: Path, request: dict, expected: dict) -> dict:
    write_json(root / "request.json", request)
    result = json.loads(subprocess.check_output(
        [sys.executable, "-m", "scripts.hla_tok.compact.receiver", str(root)], cwd=ROOT, text=True))
    hashes = {r["tag"]: r["rgb_sha256"] for r in result["decoded"]}
    if hashes != expected or result["pid"] == os.getpid() or result["synthesis_modules"] != ["local_decoder", "rgb_decoder"]:
        raise ValueError("isolated receiver disagrees with the physical one-shot decode")
    return result | {"sender_pid": os.getpid(), "rgb_bit_exact": True, "synthesis_only": True,
                     "synthesis_sha256": request["synthesis_sha256"]}


def _case(inputs: Inputs, codec: CompactCodec, receiver: ContinuousReceiver, frames: list[torch.Tensor],
          budgets: list[int], precision: str) -> tuple[list[bytes], str, dict]:
    with compute(precision):
        full = codec.encode(inputs.model, frames[0], MAX_K, 0)
        packet = deserialize(full.serialize_range(0, MAX_K))
        _prefixes(inputs, codec, frames[0], budgets, packet)
        _unreceived(codec, inputs, frames[0])
        packets, rgb_sha = _receive(codec, receiver, full)
        causal = _causality(inputs, receiver, codec, frames)
    return packets, rgb_sha, causal


@torch.inference_mode()
def audit(inputs: Inputs, config: dict, output: Path, assets: dict, *, evidence: Path | None = None) -> None:
    evidence = output if evidence is None else evidence
    torch.use_deterministic_algorithms(True)
    frames = [s for _, _, s in validation_frames(inputs, representative(inputs.cohort))]
    receiver = ContinuousReceiver.from_model(inputs.model)
    orders = compact_orders()
    knobs = inputs.config["prefix"]
    boundaries = [*knobs["boundaries"], TAIL_REGIONS - 1, TAIL_REGIONS, TAIL_REGIONS + 1]
    plan = prefix_plan({"budgets": config["budgets"], "prefix": knobs | {
        "boundaries": boundaries, "seed": config["prefix_seed"],
    }})
    budgets = plan["all"]
    rows, causal, expected = [], [], {}
    with TemporaryDirectory(prefix="hla-compact-receiver-") as name:
        root = Path(name)
        request = _prepare_request(root, receiver, inputs)
        for asset, spec in assets.items():
            for order in ("O2", "O1"):
                codec, _ = load_asset(output / "profiles" / spec["filename"], spec, PROFILES_BY_LABEL[spec["profile"]],
                                      orders[order], inputs.checkpoint["state_dict_sha256"])
                for precision in ("fp32", "bf16"):
                    keys = {"profile_asset": asset, "order": order, "compute_dtype": precision}
                    packets, rgb, timing_scope = _case(inputs, codec, receiver, frames, budgets, precision)
                    _record_job(root, request, output, spec, keys, packets)
                    expected["_".join(keys.values())] = rgb
                    rows.append(keys | {"cases": len(budgets), "physical_bytes_prefix": True, "code_prefix": True,
                                        "unused_token_perturbation": "PASS", "incremental": "PASS"})
                    causal.append(keys | timing_scope)
            print(f"Compact streaming checks passed for {asset}", flush=True)
        independent = _independent(root, request, expected)
    write_json(evidence / "prefix_audit.json", {"status": "PASS", "plan": plan, "results": rows})
    write_json(evidence / "causality_audit.json", {"status": "PASS", "results": causal})
    write_json(evidence / "wire_audit.json", {"status": "PASS", "independent_process": independent,
                                           "header_bytes": HEADER_BYTES, "padding_bytes": 0,
                                           "diagnostics_test_reads": inputs.reader.counts["test"]})


def main() -> None:
    parser = run_arguments(__doc__)
    parser.set_defaults(output=OUTPUT)
    parser.add_argument("--trained", action="store_true")
    args = parser.parse_args()
    with locked_output(args.output):
        inputs, config = compact_inputs(args.v3, args.output)
        assets = json_at(args.output / "compact_profile_manifest.json")
        if args.trained:
            trained = json_at(args.output / "training_selection.json")
            if trained["adopted"]:
                audit(inputs, config, args.output, {trained["asset"]: trained["spec"]},
                      evidence=args.output / "trained_correctness")
                inputs.finish(args.output / "trained_checks_inputs.json")
        else:
            audit(inputs, config, args.output, assets)
            inputs.finish(args.output / "checks_inputs.json")


if __name__ == "__main__":
    main()
