"""Compare every budget through a native I/P transition before training a fork."""

import argparse
import hashlib

import torch

from adaptok_hr.codec.bitstream import parse
from adaptok_hr.codec.codec import Codec
from adaptok_hr.data.dataset import read_clip
from adaptok_hr.data.manifest import load_index
from adaptok_hr.runtime import file_hash, seed_everything, source_identity, write_json
from adaptok_hr.training.checkpoint import load_model

from .run import ROOT, checkpoint


def compare(parent, child):
    torch.set_num_threads(4)
    seed_everything(0)
    paths = [checkpoint(name, 1000) for name in (parent, child)]
    old, saved = load_model(paths[0])
    new, _ = load_model(paths[1])
    index = load_index(saved["config"]["data"]["index"])
    source = next(s for s in index["splits"]["validation"] if s["content_id"] == "DAVIS2017:car-roundabout")
    frames, inputs = read_clip(source, 0, 2)
    inputs["rgb_sha256"] = [hashlib.sha256(f.numpy().tobytes()).hexdigest() for f in frames]
    original, candidate = Codec(old, True), Codec(new, True)
    rows = []
    for budget in old.config.budgets:
        old_state = new_state = None
        for frame in frames.cuda():
            data, old_state = original.encode_frame(frame[None], old_state, budget, source["content_id"])
            other, new_state = candidate.encode_frame(frame[None], new_state, budget, source["content_id"])
            ids_equal = parse(data).ids == parse(other).ids
            rgb_equal = old_state.rgb.dtype == new_state.rgb.dtype and torch.equal(old_state.rgb, new_state.rgb)
            feature_equal = (old_state.feature.dtype == new_state.feature.dtype == torch.bfloat16
                             and torch.equal(old_state.feature, new_state.feature))
            rows.append({"budget": budget, "frame": old_state.frame_index, "type": old_state.frame_type,
                         "bytes": [len(data), len(other)], "id_equal": ids_equal, "rgb_equal": rgb_equal,
                         "feature_equal": feature_equal, "feature_dtypes": [str(s.feature.dtype) for s in (old_state, new_state)],
                         "packet_sha256": [s.chain.hex() for s in (old_state, new_state)],
                         "max_abs_rgb_difference": float((old_state.rgb - new_state.rgb).abs().max()),
                         "max_abs_feature_difference": float((old_state.feature.float() - new_state.feature.float()).abs().max())})
    passed = all(r["id_equal"] and r["rgb_equal"] and r["feature_equal"] for r in rows)
    record = {"parent": parent, "child": child, "step": 1000, "pass": passed,
              "source_sha256": source_identity(), "checkpoint_sha256": [file_hash(p) for p in paths],
              "manifest": index["identity"], "input": inputs, "rows": rows}
    write_json(ROOT / child / "initial_reference.json", record)
    if not passed:
        raise ValueError(f"native I/P reference mismatch before training {child}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent")
    parser.add_argument("child")
    args = parser.parse_args()
    compare(args.parent, args.child)
