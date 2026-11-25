"""Real C0 bit-exact prefix and temporal streaming checks."""

from collections.abc import Iterator
from itertools import combinations
from pathlib import Path
from random import Random

import torch
from torch import Tensor

from hla_tok.evaluation.io import write_json
from hla_tok.streaming.continuous import (
    MAX_K,
    ContinuousReceiver,
    RankOrder,
    encode_continuous,
    stream_frames,
    truncate,
)
from scripts.hla_tok.continuous.inputs import Inputs
from scripts.hla_tok.continuous.matrix import compute, representative, validation_frames


def prefix_plan(config: dict) -> dict:
    fixed = set(config["budgets"])
    boundaries = set(config["prefix"]["boundaries"])
    population = sorted(set(range(1, MAX_K + 1)) - fixed - boundaries)
    rng = Random(config["prefix"]["seed"])
    other = rng.sample(population, config["prefix"]["nonpreset_count"])
    return {"formal": sorted(fixed), "boundary": sorted(boundaries), "nonpreset": sorted(other),
            "all": sorted(fixed | boundaries | set(other))}


def _pair_failures(prefixes: dict[int, Tensor], formal: list[int]) -> list[list[int]]:
    return [[small, large] for small, large in combinations(formal, 2)
            if not torch.equal(prefixes[small], prefixes[large][:, :small])]


def _on_source(model, receiver, source: Tensor, order: RankOrder, plan: dict, *, native: bool) -> dict:
    full = encode_continuous(model, source)
    original = model.encode_training(source, MAX_K, quantize=False)
    canonical = order.apply(full)
    public = {count: truncate(canonical, count) for count in plan["all"]}
    public_mismatch = _pair_failures(public, plan["formal"])
    max_equal = torch.equal(order.restore_full(canonical), original.continuous)
    rgb_equal = torch.equal(receiver.decode(canonical, order), model.decode_training(original, MAX_K))
    legal_shapes = all(value.shape == (1, count, 32) for count, value in public.items())
    public_ok = legal_shapes and not public_mismatch and max_equal and rgb_equal and all(
        torch.equal(prefix, canonical[:, :count]) for count, prefix in public.items())
    row = {
        "order": order.name.name, "maximum_token_tensor_equal": max_equal, "maximum_rgb_bit_exact": rgb_equal,
        "public_all_prefixes_exact": public_ok, "public_formal_pair_failures": public_mismatch,
        "tensor_shape": list(full.shape), "tensor_dtype": str(full.dtype),
    }
    if native:
        independent = {count: encode_continuous(model, source, count) for count in plan["all"]}
        mismatches = [count for count, prefix in independent.items() if not torch.equal(prefix, full[:, :count])]
        row |= {"native_vs_max_failures": mismatches,
                "native_formal_pair_failures": _pair_failures(independent, plan["formal"])}
    return row


@torch.inference_mode()
def audit_prefix(inputs: Inputs, output: Path, orders: list[RankOrder], *, native: bool) -> None:
    plan = prefix_plan(inputs.config)
    receiver = ContinuousReceiver.from_model(inputs.model)
    rows = []
    for video, frame, source in validation_frames(inputs, representative(inputs.cohort)):
        for precision in inputs.config["prefix"]["precisions"]:
            with compute(precision):
                for order in orders:
                    result = _on_source(inputs.model, receiver, source, order, plan,
                                        native=native and order.name.name == "O0")
                    rows.append(result | {"video": video, "frame": frame, "compute_dtype": precision})
    public_pass = all(row["public_all_prefixes_exact"] for row in rows)
    native_pass = all(not row.get("native_vs_max_failures") and not row.get("native_formal_pair_failures")
                      for row in rows)
    output_name = "prefix_audit.json" if native else "ordered_prefix_audit.json"
    write_json(output / output_name, inputs.provenance() | {
        "status": "PASS" if public_pass else "FAIL", "native_budget_encoder":
            ("PASS" if native_pass else "FAIL") if native else "NOT_APPLICABLE",
        "public_max_then_truncate": "PASS" if public_pass else "FAIL",
        "formal_pairs_per_source_and_precision": len(list(combinations(plan["formal"], 2))),
        "legal_history_states": "NOT_APPLICABLE", "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "nonpresets_checked_against_independently_encoded_maximum": native,
        "rank_chunk_count_at_maximum": 4, "maximal_encode_per_public_frame": True,
        "budgets": plan, "rows": rows,
    })
    if not public_pass:
        raise RuntimeError("public continuous prefix or maximum reconstruction failed")


def _variants(frames: list[Tensor], generator: torch.Generator) -> Iterator[tuple[str, list[tuple[Tensor, int]]]]:
    past, current, future, extension = frames
    common = [(past, 8160), (current, 4080)]
    noise = torch.rand(future.shape, device=future.device, generator=generator)
    yield "reference", common + [(future, 510), (extension, MAX_K)]
    yield "modify_future_pixels", common + [(1 - future, 510), (extension, MAX_K)]
    yield "replace_future_with_noise", common + [(noise, 510), (extension, MAX_K)]
    yield "delete_future", common
    yield "shorten_future", common + [(future, 510)]
    yield "extend_future", common + [(future, 510), (extension, MAX_K), (noise, MAX_K)]
    yield "change_future_budget", common + [(future, MAX_K), (extension, 1)]


def _first_four(inputs: Inputs, video: str, count: int) -> list[Tensor]:
    start = count // 2 - 1
    return [source for _, _, source in validation_frames(inputs, ((video, i) for i in range(start, start + 4)))]


@torch.inference_mode()
def audit_causality(inputs: Inputs, output: Path) -> None:
    rows = []
    order = RankOrder.native("cuda")
    rng = torch.Generator(device="cuda").manual_seed(inputs.config["dynamics"]["seed"])
    for video, count in inputs.cohort.items():
        frames = _first_four(inputs, video, count)
        for precision in inputs.config["prefix"]["precisions"]:
            with compute(precision):
                variants = _variants(frames, rng)
                _, reference = next(variants)
                original = list(stream_frames(inputs.model, reference, order))[:2]
                for variation, source in variants:
                    measured = list(stream_frames(inputs.model, source, order))[:2]
                    exact = [torch.equal(left, right) for a, b in zip(original, measured, strict=True)
                             for left, right in zip(a, b, strict=True)]
                    rows.append({"video": video, "current_frame": count // 2, "variation": variation,
                                 "compute_dtype": precision, "tokens_and_rgb_exact": all(exact),
                                 "committed_history": None})
    passed = all(row["tokens_and_rgb_exact"] for row in rows)
    write_json(output / "causality_audit.json", inputs.provenance() | {
        "status": "PASS" if passed else "FAIL", "causality": "frame", "block_frames": 1,
        "lookahead_frames": 0, "first_frame_wait_frames": 0, "algorithmic_lookahead_ms": 0,
        "compute_latency": "measured separately on the serialized path", "rows": rows,
    })
    if not passed:
        raise RuntimeError("a future perturbation affected frame-local C0 output")
