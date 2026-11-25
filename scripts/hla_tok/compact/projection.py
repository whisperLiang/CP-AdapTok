"""Projection-only ablations and a frozen choice of representation for each width."""

from pathlib import Path

import numpy as np
from torch import inference_mode

from hla_tok.compression.profiles import PCA_PROFILES, Profile
from hla_tok.evaluation.io import write_csv, write_json
from hla_tok.evaluation.metrics import sha256_file
from hla_tok.streaming.continuous import MAX_K
from scripts.hla_tok.compact.assets import load_asset, save_asset, transform_for
from scripts.hla_tok.compact.inputs import OUTPUT, compact_inputs, compact_orders
from scripts.hla_tok.compact.matrix import Case, evaluate, summarize_matrix
from scripts.hla_tok.continuous.inputs import METRICS, Inputs, locked_output, run_arguments
from scripts.hla_tok.v2.protocol import json_at


def asset_key(profile: Profile, kind: str) -> str:
    return f"{profile.label}_{kind}"


def projection_specs(config: dict) -> list[tuple[Profile, str]]:
    return [(Profile.RAW32, "identity")] + [(PCA_PROFILES[d], kind)
                                          for d in config["projection"]["dimensions"] for kind in ("global", "rank")]


def projection_assets(inputs: Inputs, config: dict, output: Path) -> dict:
    file = output / "rank_covariance.npz"
    if sha256_file(file) != json_at(output / "representation_audit.json")["covariance_sha256"]:
        raise ValueError("covariance does not match the frozen Train audit")
    specs = {}
    orders = compact_orders()
    with np.load(file, allow_pickle=False) as stats:
        for profile, kind in projection_specs(config):
            key = asset_key(profile, kind)
            path = output / "profiles" / f"{key}.npz"
            transform = transform_for(dict(stats), kind, profile.dimension)
            spec = save_asset(path, profile, kind, transform, None)
            _, deployment = load_asset(path, spec, profile, orders["O2"], inputs.checkpoint["state_dict_sha256"])
            specs[key] = spec | deployment
    write_json(output / "projection_assets.json", specs)
    return specs


def choose_projection(rows: list[dict], config: dict) -> tuple[dict, list[dict]]:
    primary = config["projection"]["primary_order"]
    tolerance = config["projection"]["tolerances"]
    lookup = {(row["profile"], row["projection"], row["budget"]): row for row in rows if row["order"] == primary}
    selection, detail = {}, []
    for dimension in config["projection"]["dimensions"]:
        profile = PCA_PROFILES[dimension].label
        wins = dict.fromkeys(("rank", "global"), 0)
        for k in config["budgets"]:
            g, r = (lookup[profile, kind, k] for kind in ("global", "rank"))
            gains = {name: (r[name] - g[name]) * (1 if name == "global_psnr" else -1) for name in METRICS}
            wins["rank"] += sum(delta > tolerance[name] for name, delta in gains.items())
            wins["global"] += sum(delta < -tolerance[name] for name, delta in gains.items())
            detail.append({"dimension": dimension, "budget": k} | {f"rank_gain_{m}": v for m, v in gains.items()})
        selected = "rank" if wins["rank"] > wins["global"] else "global"
        selection[str(dimension)] = {"projection": selected, "material_metric_wins": wins}
    return selection, detail


@inference_mode()
def run(inputs: Inputs, config: dict, output: Path) -> None:
    assets = projection_assets(inputs, config, output)
    primary = config["projection"]["primary_order"]
    maximum = [Case(key, primary, MAX_K) for key in assets]
    rows = evaluate(inputs, config, assets, output, "projection_maximum", maximum)
    write_csv(output / "projection_maximum_rd.csv", summarize_matrix(rows))
    cases = [Case(key, primary, k) for key in assets for k in config["budgets"] if k != MAX_K]
    cases.extend(Case(key, "O1", k) for key in assets for k in config["projection"]["o1_budgets"])
    rows.extend(evaluate(inputs, config, assets, output, "projection_prefixes", cases))
    summary = summarize_matrix(rows)
    write_csv(output / "projection_only_per_frame.csv", rows)
    write_csv(output / "projection_only_by_video.csv", summarize_matrix(rows, by_video=True))
    write_csv(output / "projection_only_rd.csv", summary)
    selection, detail = choose_projection(summary, config)
    write_csv(output / "global_vs_rank_pca.csv", detail)
    write_json(output / "projection_selection.json", selection)
    inputs.finish(output / "projection_inputs.json")


def main() -> None:
    parser = run_arguments(__doc__)
    parser.set_defaults(output=OUTPUT)
    args = parser.parse_args()
    with locked_output(args.output):
        inputs, config = compact_inputs(args.v3, args.output)
        run(inputs, config, args.output)


if __name__ == "__main__":
    main()
