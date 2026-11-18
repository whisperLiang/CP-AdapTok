"""Measure spurious high frequencies on flat source pixels in real K2048 replays."""

import argparse
import hashlib
import json

import torch
from torch.nn import functional as F

from adaptok_hr.codec.codec import Codec
from adaptok_hr.data.dataset import read_clip
from adaptok_hr.data.manifest import load_index
from adaptok_hr.runtime import file_hash, seed_everything, write_json
from adaptok_hr.training.checkpoint import load_model
from experiments.adaptok_hr.fine_recovery.diagnose import highpass

from .evaluate import folder
from .run import PROTOCOL, ROOT, checkpoint


def flat_error(original, decoded):
    # At most two native 8-bit levels of local range in every RGB channel.
    spread = F.max_pool2d(original, 3, 1, 1) + F.max_pool2d(-original, 3, 1, 1)
    flat = spread.amax(dim=1) <= 2 / 255 + 1e-7
    error = highpass(decoded.float() - original).abs()
    large = flat & (error.amax(dim=1) > 4 / 255)
    return {"flat_pixels": int(flat.sum()), "pixels": flat.numel(), "large_error_pixels": int(large.sum()),
            "abs_error_sum": float((error * flat[:, None]).sum(dtype=torch.float64))}


def aggregate(rows):
    flat = sum(r["flat_pixels"] for r in rows)
    return {"frames": len(rows), "flat_pixels": flat, "flat_fraction": flat / sum(r["pixels"] for r in rows),
            "flat_hp_mae": sum(r["abs_error_sum"] for r in rows) / (3 * flat) if flat else None,
            "over_4_levels_fraction": sum(r["large_error_pixels"] for r in rows) / flat if flat else None}


def replay(name):
    step = 1300
    base = ROOT / name / f"validation_{step}_64"
    summary = json.loads((base / "summary.json").read_text())
    out = ROOT / name / f"flat_regions_{step}.json"
    identity = {"checkpoint_sha256": file_hash(checkpoint(name, step)), "packet_metrics_sha256": file_hash(base / "closed_loop.jsonl"),
                "implementation_sha256": file_hash(__file__), "manifest": summary["identity"]["manifest"]}
    if identity["checkpoint_sha256"] != summary["identity"]["checkpoint_sha256"]:
        raise ValueError("quality measurements belong to another checkpoint")
    if out.exists() and json.loads(out.read_text())["identity"] == identity:
        return
    model, saved = load_model(checkpoint(name, step))
    receiver = Codec(model, True)
    index = load_index(saved["config"]["data"]["index"])
    if index["identity"] != identity["manifest"]:
        raise ValueError("replay source manifest differs from measured validation")
    reference = {}
    for line in (base / "closed_loop.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["tokens"] == 2048:
            reference[r["content_id"], r["frame"]] = r
    sources = {r["content_id"]: r for r in index["splits"]["validation"]}
    rows = []
    for content in PROTOCOL["sources"]:
        source = sources[content]
        clip, _ = read_clip(source, 0, min(summary["identity"]["frames_per_source"], source["original_frames"]))
        state = None
        for n, cpu in enumerate(clip):
            ref = reference[content, n]
            packet = (folder(base, "closed_loop", 2048, content) / f"{n:04d}.bin").read_bytes()
            rgb, state = receiver.decode_frame(packet, state)
            original = cpu[None].cuda()
            if (state.chain.hex() != ref["packet_sha256"] or hashlib.sha256(cpu.numpy().tobytes()).hexdigest() != ref["input_rgb_sha256"]
                    or float(F.mse_loss(rgb.float(), original)) != ref["mse"]):
                raise ValueError("flat-region replay does not match the archived inputs, packet or reconstruction")
            rows.append({"content_id": content, "frame": n, "packet_sha256": state.chain.hex(), **flat_error(original, rgb)})
    pooled = aggregate(rows)
    by_source = {s: aggregate([r for r in rows if r["content_id"] == s]) for s in PROTOCOL["sources"]}
    write_json(out, {"scope": "descriptive_only", "budget": 2048, "identity": identity,
                     "rows": rows, "pooled": pooled, "sources": by_source})
    print(json.dumps({"name": name, **pooled}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--names", nargs="+")
    args = parser.parse_args()
    torch.set_num_threads(4)
    seed_everything(20261118)
    names = args.names or sorted(p.parents[1].name for p in ROOT.glob("S*/validation_1300_64/summary.json"))
    if not names:
        raise ValueError("requires a completed decoder-stage validation")
    for name in names:
        replay(name)
