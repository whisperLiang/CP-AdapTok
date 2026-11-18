"""Execute the gated native study serially on one GPU and retain every real result."""

import argparse
import json
import time
from pathlib import Path

from .run import ARCHITECTURES, ARMS, ROOT, execute


def wait_process(pid):
    path = Path(f"/proc/{pid}/stat")
    while path.exists():
        fields = path.read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return
        time.sleep(5)


def invoke(module, arguments, name):
    execute([f"experiments.adaptok_hr.unified.{module}", *arguments], ROOT / "logs" / f"{name}.log")


def evaluate_and_time(names, step, label):
    invoke("evaluate", ["--names", *names, "--steps", step, "--frames", 64], f"{label}_evaluate")
    invoke("benchmark", ["--names", *names, "--steps", step], f"{label}_benchmark")


def scratch(step, loss):
    # Start with the unchanged-width N0 architecture and optimize its unified
    # training before allocating runtime to the fusion comparisons.
    for arm in ARCHITECTURES:
        for seed in (1, 2):
            invoke("run", ["scratch", "--seeds", seed, "--arms", arm, "--steps", step,
                           "--loss-arm", loss], f"S{seed}_{arm}_driver_{step}")
    names = [f"S{s}_{a}" for s in (1, 2) for a in ARCHITECTURES]
    evaluate_and_time(names, step, f"B_{step}")
    invoke("decision", ["B", "--steps", step, "--frames", 64], f"B_{step}_decision")
    return json.loads((ROOT / f"decision_B_{step}.json").read_text())["advance"]


def study(wait_pid):
    if wait_pid:
        wait_process(wait_pid)
    invoke("run", ["pilot", "--seeds", 1, 2], "pilot_resume")
    names = [f"S{s}_{a}" for s in (1, 2) for a in ARMS]
    evaluate_and_time(names, 1600, "A_1600")
    invoke("decision", ["A", "--steps", 1600, "--frames", 64], "A_decision")
    decision = json.loads((ROOT / "decision_A.json").read_text())
    for step in (300, 1000, 3000, 5000):
        if not scratch(step, decision["winner"]):
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-pilot-pid", type=int)
    study(parser.parse_args().wait_pilot_pid)
