"""Load the YAML config, with optional `section.key=value` overrides from the CLI."""

from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    with open(path, encoding="utf-8") as file:
        config = yaml.safe_load(file)
    for item in overrides or []:
        key, separator, raw_value = item.partition("=")
        if not separator:
            raise ValueError(f"Override must look like section.key=value, got: {item}")
        node = config
        *parents, leaf = key.split(".")
        for parent in parents:
            if parent not in node:
                raise KeyError(f"Unknown config section in override: {key}")
            node = node[parent]
        if leaf not in node:
            raise KeyError(f"Unknown config key in override: {key}")
        node[leaf] = yaml.safe_load(raw_value)
    return config
