"""Execute V3 training-claim fault injection in separate test processes."""

import argparse
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from hla_tok.evaluation.io import write_json
from scripts.hla_tok.uf_v2.mutations import Fault, Finder
from scripts.hla_tok.v2.protocol import ROOT

TRAIN = "scripts.hla_tok.uf_v3.train"
JOINT = "test_joint.py::test_joint_probe_rejects_frozen_encoder_and_a_missing_rate_path"
FAULTS = (
    Fault("freeze an encoder during the joint step", TRAIN,
          "if branch not in named or not named[branch].requires_grad:", "if branch not in named:", JOINT),
    Fault("remove the independent rate-gradient gate", TRAIN, 'for name in ("distortion", "rate"):',
          'for name in ("distortion",):', JOINT),
    Fault("admit cached features as video pixels", TRAIN,
          "if frames.ndim != 4 or frames.shape[1] != 3 or not frames.is_floating_point() or min(frames.shape[-2:]) < 64:",
          "if frames.ndim != 4:", "test_joint.py::test_training_accepts_raw_rgb_and_rejects_cached_latent_or_feature"),
    Fault("execute the full P encoder again", "hla_tok.uf_v3.joint",
          "z = model.hyper_encoder(y)", "model.encoder(pixels, context, qp_row(model.q_encoder, qp, context.dtype))"
          "\n        z = model.hyper_encoder(y)",
          "test_joint.py::test_raw_rgb_unroll_routes_separate_encoder_gradients_and_keeps_base_state"),
    Fault("leave the CUDA proxy on its old parameters", "hla_tok.uf_v3.deploy",
          "network.proxy.set_param(network.add_cdf_to_state_dict(network.state_dict()),\n"
          "                                        network.gaussian_encoder.skip_thres)",
          "pass", "test_deploy.py::test_deployment_takes_the_checkpoint_tensor_and_refreshes_the_proxy"),
    Fault("emit a different frame index in the valid CRC header", "hla_tok.uf.wire",
          "BASE_HEADER.pack(MAGIC, tag, info.index,", "BASE_HEADER.pack(MAGIC, tag, info.index + 1,",
          "test_deploy.py::test_wire_identity_preserves_signalled_shape_qp_and_independent_frame"),

)


def run_child(fault):
    import pytest

    sys.meta_path.insert(0, Finder(fault))
    return pytest.main(["-q", "--tb=short", f"tests/hla_tok/uf_v3/{fault.test}"])


def execute(out, fault):
    result = subprocess.run([sys.executable, "-m", "scripts.hla_tok.uf_v3.qa", "--inject", fault.name],
                            cwd=ROOT, check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    killed = result.returncode == 1 and " failed" in result.stdout and "ERROR" not in result.stdout
    mode = "KILLED" if killed else "SURVIVED" if result.returncode == 0 else "ERROR"
    path = out / f"case-{FAULTS.index(fault) + 1}.log"
    path.write_text(result.stdout)
    return asdict(fault) | {"result": mode, "returncode": result.returncode, "log": path.name}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--inject")
    args = parser.parse_args()
    if args.inject:
        return run_child(next(fault for fault in FAULTS if fault.name == args.inject))
    if not args.output:
        parser.error("--output is required for the suite")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = [execute(args.output, case) for case in FAULTS]
    counts = {key: sum(row["result"] == key for row in rows) for key in ("KILLED", "SURVIVED", "ERROR")}
    write_json(args.output / "summary.json", {"cases": rows, "counts": counts})
    return int(counts["KILLED"] != len(rows))


if __name__ == "__main__":
    sys.exit(main())
