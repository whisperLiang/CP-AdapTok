"""Run the preregistered decision tree and prove each within-seed comparison is matched."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from adaptok_hr.runtime import stable_hash, write_json

from .decisions import quality, result, select
from .run import CHECKPOINTS, PROTOCOL, ROOT, checkpoint, execute, run_name, train


def read(path):
    return json.loads(Path(path).read_text())


def training_rows(name, steps):
    with (ROOT / name / "training.jsonl").open() as f:
        return [r for line in f if (r := json.loads(line))["step"] <= steps]


def schedule(row):
    return row["input"], row["budgets"], [f["auxiliary_budget"] for f in row["frame_metrics"]]


def match(seed, arms, steps):
    schedules, initials, sources, counts, settings, manifests = {}, {}, {}, {}, {}, {}
    for arm in arms:
        name = run_name(seed, arm)
        rows = training_rows(name, steps)
        if [r["step"] for r in rows] != list(range(1, steps + 1)):
            raise ValueError(f"optimizer steps do not match for {name}")
        schedules[name] = stable_hash([schedule(r) for r in rows])
        counts[name] = sum(r["frames"] for r in rows)
        initial = read(checkpoint(name, 0).with_suffix(".json"))
        initials[name] = initial["model_sha256"]
        config = initial["config"]
        config["model"].pop("fine_input")
        config["training"]["losses"].pop("suffix_detail")
        settings[name] = stable_hash({k: config[k] for k in ("training", "model", "evaluation", "data")})
        snapshots = [read(p) for p in checkpoint(name, steps).parent.glob("step_*.json")
                     if int(p.stem.split("_")[1]) <= steps]
        sources[name] = sorted({m["source_sha256"] for m in snapshots})
        manifests[name] = initial["manifest_identity"]
    for values in (schedules, counts, initials, settings, manifests):
        if len(set(values.values())) != 1:
            raise ValueError(f"unmatched comparison: {values}")
    if set(counts.values()) != {4 * steps} or len({s for values in sources.values() for s in values}) != 1:
        raise ValueError("training frames or production code differ")
    historical_equal = None
    historical_weights = None
    if seed == 1:
        old = Path("results/adaptok_hr_fine_recovery/A1/training.jsonl")
        rows = [json.loads(line) for line in old.read_text().splitlines()][:steps]
        control = training_rows("S1_R00", steps)
        historical_equal = (stable_hash([schedule(r) for r in rows]) == schedules["S1_R00"]
                            and all(a["mse"] == b["mse"] for a, b in zip(rows, control)))
        original_meta = read(Path("checkpoints/adaptok_hr_fine_recovery/A1") / f"step_{steps:06d}.json")
        historical_weights = (original_meta["model_sha256"] ==
                              read(checkpoint("S1_R00", steps).with_suffix(".json"))["model_sha256"])
        if not historical_equal or not historical_weights:
            raise ValueError("new control does not reproduce original A1 training")
    evidence = {"seed": seed, "steps": steps, "training_frames": counts, "schedule_sha256": schedules,
                "common_initial_full_weights": initials, "common_settings": settings, "production_source": sources,
                "manifest": manifests, "historical_A1_each_step_mse_and_plan_match": historical_equal,
                "historical_A1_all_final_model_tensors_match": historical_weights}
    write_json(ROOT / f"match_S{seed}_{steps}.json", evidence)
    return evidence


def evaluate_arm(name, steps, frames):
    execute(["experiments.adaptok_hr.fine_detail_v2.evaluate", "--names", name, "--steps", steps, "--frames", frames],
            ROOT / "logs" / f"{name}_eval_{steps}.log")


def screen():
    match(1, list(PROTOCOL["arms"]), 300)
    winner = select()
    write_json(ROOT / "selection.json", {"winner": winner, "protocol": stable_hash(PROTOCOL),
               "screen": {a: {"information": (r := result(run_name(1, a), 300))["information"],
                              "gain": r["same_state_gain_db"], "lpips": r["closed_loop"]["2048"]["lpips"]}
                          for a in PROTOCOL["arms"]}})
    return winner


def confirmation(seed, winner):
    arms = [winner, "R00"]
    # Quality uses different processes and has no timed throughput claims.
    with ThreadPoolExecutor(max_workers=1) as evaluations:
        jobs = []
        for arm in arms:
            train(seed, arm, 1000)
            jobs.append(evaluations.submit(evaluate_arm, run_name(seed, arm), 1000, 64))
        for job in jobs:
            job.result()
    match(seed, arms, 1000)
    candidate, control = (result(run_name(seed, arm), 1000) for arm in arms)
    return {"information": candidate["information"], "quality": quality(candidate, control, read(PROTOCOL["historical_validation"]))}


def confirm_and_decide():
    winner = screen()
    outcomes = {}
    for seed in (1, 2):
        outcomes[str(seed)] = confirmation(seed, winner)
        write_json(ROOT / f"confirmation_S{seed}.json", outcomes[str(seed)])
        if not outcomes[str(seed)]["information"]["pass"]:
            break
    informed = len(outcomes) == 2 and all(r["information"]["pass"] for r in outcomes.values())
    quality_ready = informed and all(r["quality"]["pass"] for r in outcomes.values())
    write_json(ROOT / "decision.json", {"winner": winner, "seeds": outcomes, "information_both_seeds": informed,
               "decoder_stage_allowed": informed, "quality_before_decoder_pass": quality_ready,
               "negative_information_stop": not informed, "new_test_used": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("screen", "confirm"))
    args = parser.parse_args()
    if args.stage == "screen":
        print(screen())
    else:
        confirm_and_decide()
