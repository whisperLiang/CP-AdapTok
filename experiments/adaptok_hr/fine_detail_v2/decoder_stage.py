"""Matched conditional decoder continuations, including a synchronous original control."""

import json
from concurrent.futures import ThreadPoolExecutor

import torch
import yaml

from adaptok_hr.runtime import read_config, stable_hash, write_json

from .decisions import quality, result
from .fork import fork_at
from .run import CHECKPOINTS, PACKAGE, PROTOCOL, ROOT, checkpoint, execute, run_name
from .study import evaluate_arm, read, schedule, training_rows

SPEC = read(PACKAGE / "decoder_protocol.json")
STOP = 1000 + PROTOCOL["decoder"]["matched_additional_updates"]


def prepare_child(seed, arm, winner):
    parent = run_name(seed, "R00" if arm == "C0" else winner)
    name = run_name(seed, arm)
    config = read_config(ROOT / "configs" / f"{parent}.yaml")
    config["run_scope"] = {"overfit": None, "overfit_frames": 4}
    residual, detail_weight = SPEC["variants"].get(arm, [False, 0.0])
    if residual:
        config["model"]["fine_decoder"] = "masked_s32_v1"
    if detail_weight:
        config["training"]["losses"]["detail"] = detail_weight
    config["experiment"] = {**config["experiment"], "name": name, "arm": arm, "parent": parent,
                            "initialization": "matched_step_1000_fork"}
    config_path = ROOT / "configs" / f"{name}.yaml"
    if config_path.exists() and read_config(config_path) != config:
        raise ValueError("existing continuation configuration differs")
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    fork_at(parent, name, config)
    execute(["experiments.adaptok_hr.fine_detail_v2.zero_reference", parent, name],
            ROOT / "logs" / f"{name}_initial_reference.log")
    return name, config_path


def train_child(seed, arm, winner):
    name, config_path = prepare_child(seed, arm, winner)
    end = ROOT / name / "completion.json"
    if end.exists() and read(end)["completed_steps"] == STOP and checkpoint(name, STOP).is_file():
        return
    committed = sorted(p for p in (CHECKPOINTS / name).glob("step_*.pt") if p.with_suffix(".json").is_file())
    execute(["adaptok_hr.training.train", "--config", config_path, "--output", ROOT / name,
             "--checkpoints", CHECKPOINTS, "--steps", STOP, "--resume", committed[-1]],
            ROOT / "logs" / f"{name}_train_{STOP}.log")


def match_children(seed, arms):
    schedules, parents, sources = {}, {}, {}
    for arm in arms:
        name = run_name(seed, arm)
        rows = training_rows(name, STOP)
        suffix = rows[1000:]
        if [r["step"] for r in suffix] != list(range(1001, STOP + 1)) or sum(r["frames"] for r in suffix) != 4 * (STOP - 1000):
            raise ValueError("wrong number of additional updates or frames")
        schedules[arm] = stable_hash([schedule(r) for r in suffix])
        parents[arm] = read(ROOT / name / "fork.json")
        sources[arm] = read(checkpoint(name, STOP).with_suffix(".json"))["source_sha256"]
    if len(set(schedules.values())) != 1 or len(set(sources.values())) != 1:
        raise ValueError("unmatched continuation exposure or training source")
    same_parent = [r["lineage"]["parent_checkpoint_sha256"] for a, r in parents.items() if a != "C0"]
    if len(set(same_parent)) != 1:
        raise ValueError("decoder variants do not share the exact parent")
    write_json(ROOT / f"match_D_S{seed}.json", {"schedules": schedules, "parents": parents,
                                               "production_source": sources, "additional_updates": STOP - 1000})


def runs(seed, arms, winner):
    with ThreadPoolExecutor(max_workers=1) as evaluations:
        jobs = []
        for arm in arms:
            train_child(seed, arm, winner)
            jobs.append(evaluations.submit(evaluate_arm, run_name(seed, arm), STOP, 64))
        for job in jobs:
            job.result()
    match_children(seed, arms)
    control = result(run_name(seed, "C0"), STOP)
    historical = read(PROTOCOL["historical_validation"])
    outcomes = {}
    for arm in arms:
        if arm != "C0":
            row = result(run_name(seed, arm), STOP)
            outcomes[arm] = {"quality": quality(row, control, historical), "information": row["information"]["pass"],
                             "psnr": row["closed_loop"]["2048"]["psnr_global"], "lpips": row["closed_loop"]["2048"]["lpips"],
                             "dists": row["closed_loop"]["2048"]["dists"]}
    write_json(ROOT / f"decoder_S{seed}.json", outcomes)
    return outcomes


def rank(arm, outcomes):
    r = outcomes[arm]
    return (-int(r["information"] and r["quality"]["pass"]), -int(r["information"]),
            -r["psnr"], r["lpips"], r["dists"], arm)


def main():
    torch.set_num_threads(4)
    prior = read(ROOT / "decision.json")
    if not prior["decoder_stage_allowed"]:
        raise ValueError("decoder stage requires both real suffix information confirmations")
    primary = runs(1, [*SPEC["variants"], "C0"], prior["winner"])
    best = min(primary, key=lambda arm: rank(arm, primary))
    write_json(ROOT / "decoder_selection.json", {"winner": best, "primary": primary})
    confirmation = runs(2, list(dict.fromkeys(["D0", best, "C0"])), prior["winner"])
    qualifies = all(r[best]["quality"]["pass"] and r[best]["information"] for r in (primary, confirmation))
    # The replicated 1000-step parent remains eligible if extra training hurts quality.
    parent_ready = prior["quality_before_decoder_pass"]
    selected = ({"arm": best, "step": STOP} if qualifies else
                {"arm": prior["winner"], "step": 1000} if parent_ready else None)
    decision = {"decoder_winner": best, "seeds": {"1": primary, "2": confirmation},
                "adoption_ready": qualifies, "performance_candidate": selected,
                "performance_optimization_allowed": selected is not None}
    write_json(ROOT / "decoder_decision.json", decision)
    lines = ["## 匹配 Decoder 消融", "",
             f"各组从同种子第 1000 步的共同父模型起步，再走相同的 300 次更新。C0 同步续训 R00。第一种子选中 {best}，第二种子复验该方案和 D0。",
             "", "| 种子 | 模型 | K2048 PSNR | LPIPS | DISTS | 内容门槛 | 采用门槛 | 未过项目 |",
             "|---|---|---:|---:|---:|---:|---:|---|"]
    for seed, outcomes in decision["seeds"].items():
        for arm, value in outcomes.items():
            failed = ", ".join(k for k, passing in value["quality"]["checks"].items() if not passing)
            lines.append(f"| {seed} | {arm} | {value['psnr']:.5f} | {value['lpips']:.6f} | {value['dists']:.6f} | "
                         f"{value['information']} | {value['quality']['pass']} | {failed or '全部通过'} |")
    lines.extend(["", f"下表给出 {best} 相对同种子同步数 C0 的逐档差值。晚期 K2048 只比较第 48 帧起的相同画面。",
                  "", "| 种子 | 预算 / 帧段 | ΔPSNR (dB) | ΔLPIPS | ΔDISTS |", "|---|---|---:|---:|---:|"])
    for seed, outcomes in decision["seeds"].items():
        for scope, delta in outcomes[best]["quality"]["protected_delta"].items():
            lines.append(f"| {seed} | {scope} | {delta['psnr_global']:+.5f} | {delta['lpips']:+.6f} | "
                         f"{delta['dists']:+.6f} |")
    if not selected:
        lines.extend(["", "双种子的整体画质门槛未同时满足。此处结束模型扩展，后面的吞吐数字为所测模型原路径。"])
    (ROOT / "decoder_report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"winner": best, "performance_candidate": selected}))


if __name__ == "__main__":
    main()
