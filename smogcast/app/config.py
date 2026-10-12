"""Configuration of the app: the YAML files in conf/ + settings from the environment.

Two kinds of files: the data contract copied from the pipeline repo (cities, thresholds, model, gold_dictionary – see
COPIED.txt, never edited here) and this repo's own app.yaml (language model, SQL limits, knowledge base). The app has
no dependency on smogcast-core; the result has the shape the pipeline's configuration has (``cities``,
``pollutants``, ``thresholds``, ``model``, ``app``, ``gold_dictionary``, ``run.cities``, ``env``).
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

CONF = Path(__file__).resolve().parent / "conf"
FILES = ("cities.yaml", "thresholds.yaml", "model.yaml", "app.yaml", "gold_dictionary.yaml")
# Secrets and per-machine settings come from the environment, never from YAML in git (same names as the pipeline).
ENV_OVERRIDES = {
    "SMOGCAST_LLM_BASE_URL": ("app", "llm", "base_url"),
    "SMOGCAST_LLM_MODEL": ("app", "llm", "model"),
    "SMOGCAST_LLM_API_KEY": ("app", "llm", "api_key"),
}


def deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(overrides: dict[str, Any] | None = None) -> dict:
    cfg: dict = {}
    for name in FILES:
        cfg = deep_merge(cfg, yaml.safe_load((CONF / name).read_text(encoding="utf-8")) or {})
    # The app serves every configured city (the pipeline's dev/prod files list all ten as run.cities).
    cfg["run"] = {"cities": list(cfg["cities"])}
    for var, path in ENV_OVERRIDES.items():
        if os.environ.get(var):
            node = cfg
            for key in path[:-1]:
                node = node.setdefault(key, {})
            node[path[-1]] = os.environ[var]
    if overrides:
        cfg = deep_merge(cfg, overrides)
    cfg["env"] = os.environ.get("SMOGCAST_CATALOG", "smogcast_dev").removeprefix("smogcast_")
    unknown = set(cfg["run"]["cities"]) - set(cfg["cities"])
    if unknown:
        raise ValueError(f"run.cities contains unknown cities: {sorted(unknown)}")
    return cfg
