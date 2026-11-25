"""Record exact native and metric identities for the UF-V2 experiments."""
import argparse
import importlib.util
import platform
import subprocess
from pathlib import Path

import torch

from hla_tok.evaluation.io import write_json
from hla_tok.uf.official import OfficialAssets, sha256_file
from scripts.hla_tok.v2.protocol import ROOT


def installed(module: str) -> Path:
    spec = importlib.util.find_spec(module)
    if spec is None or spec.origin is None:
        raise ImportError(f"cannot locate {module}")
    return Path(spec.origin)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assets = OfficialAssets.verify(args.weights)
    metric_sources = {}
    # Match the tested V3 archival metric identities independently of weight loading.
    hub = Path(torch.hub.get_dir()) / "checkpoints"
    metric_sources["imagenet_alexnet"] = hub / "alexnet-owt-7be5be79.pth"
    metric_sources["imagenet_vgg16"] = hub / "vgg16-397923af.pth"
    metric_sources["lpips_alex_v01"] = installed("lpips").parent / "weights/v0.1/alex.pth"
    metric_sources["dists"] = installed("DISTS_pytorch").parent / "weights.pt"
    metrics = {name: sha256_file(path) for name, path in metric_sources.items()}
    import json

    cohort = json.loads((ROOT / "configs/hla_tok/validation_cohort.json").read_text())
    if any(value != cohort["metric_weights"][key] for key, value in metrics.items()):
        raise ValueError("installed evaluation metric differs from the frozen cohort")
    driver = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], text=True).strip()
    packages = ("MLCodec_extensions_cpp", "inference_extensions_cuda", "hla_uf_rans", "hla_uf_v2_records")
    write_json(args.output, {
        "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu_driver_memory": driver, "machine": platform.machine(),
        "official_weight_sha256": {path.name: sha256_file(path) for path in (assets.intra, assets.ld)},
        "native_binary_sha256": {key: sha256_file(installed(key)) for key in packages},
        "metric_sha256": metrics, "frozen_manifest_sha256": cohort["manifest_hashes"],
        "native_graph": "pinned FP16 CUDA proxy; FP32 metrics and latent arithmetic",
    })


if __name__ == "__main__":
    main()
