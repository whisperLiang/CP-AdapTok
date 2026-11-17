"""Run in the existing CP interpreter, read-only, to prove RGB and metric-weight parity."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import torch

from cp_adaptok.data import VideoSpec, read_video

from .artifacts import write_json
from .video_reader import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/original_adaptok_1080p/cp_input_parity.json"))
    parser.add_argument("--reference", type=Path, default=Path("results/large_scale/formal_20261116/final_test"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    reports = {b: json.loads((args.reference / b / "validation.json").read_text()) for b in ["B0", "B1"]}
    sources = {r["source"]["content_id"]: r["source"] for r in reports["B0"]["records"]}
    keys = [{(r["source"]["content_id"], r["frame_id"]) for r in report["records"]} for report in reports.values()]
    if keys[0] != keys[1]:
        raise ValueError("sealed CP B0/B1 use different frames")
    windows = []
    for identity, spec in sources.items():
        for start in range(spec.get("start", 0), spec["frames"], 16):
            clip = read_video(VideoSpec(**{**spec, "start": start, "frames": min(16, spec["frames"] - start)}))
            windows.append({"content_id": identity, "start": start, "shape": list(clip.shape),
                            "rgb_sha256": hashlib.sha256(clip.numpy().tobytes()).hexdigest()})
    import DISTS_pytorch
    import lpips

    paths = [Path(DISTS_pytorch.__file__).with_name("weights.pt"), Path(lpips.__file__).parent / "weights/v0.1/alex.pth"]
    paths += [Path(torch.hub.get_dir()) / "checkpoints" / p for p in ["alexnet-owt-7be5be79.pth", "vgg16-397923af.pth"]]
    output = {"reference": str(args.reference.resolve()), "windows": windows, "identical_cp_frame_keys": True,
              "frame_count": len(keys[0]), "packages": {p: importlib.metadata.version(p) for p in ["lpips", "DISTS-pytorch", "pytorch-msssim", "opencv-python-headless"]},
              "metric_weight_sha256": {str(p): file_sha256(p) for p in paths},
              "reference_sha256": {b: file_sha256(args.reference / b / "validation.json") for b in reports},
              "cp_metric_source_sha256": file_sha256("cp_adaptok/metrics.py")}
    write_json(args.output, output)
    print(json.dumps({"frame_count": output["frame_count"], "windows": len(windows)}))


if __name__ == "__main__":
    main()
