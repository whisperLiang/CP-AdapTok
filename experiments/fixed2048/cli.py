"""Explicit JSON/YAML configuration boundary for the isolated experiment."""

import json

from cp_adaptok.runtime import read_config as read_yaml_config
from cp_adaptok.runtime import resolve, seed_everything


def read_config(path):
    path = resolve(path)
    if path.suffix.lower() == ".json":
        config = json.loads(path.read_text())
    elif path.suffix.lower() in (".yaml", ".yml"):
        config = read_yaml_config(path)
    else:
        raise ValueError("experiment configuration must be JSON or YAML")
    if not isinstance(config, dict):
        raise ValueError("config must be a mapping")
    return config


def seeded_config(path):
    config = read_config(path)
    seed_everything(config["seed"], config["device"], 0.9)
    return config
