"""Utility helpers shared by RL CLI commands."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml

# Default configuration scaffold used when no explicit config is supplied.
DEFAULT_CONFIG: Dict[str, Any] = {
    "dataset": {
        "size": "small",
    },
    "train": {
        "algo": "ppo",
        "total_timesteps": 50_000,
        "max_iterations": 100,
        "sampling_strategy": "random",
        "seed": 42,
        "learning_rate": 3e-4,
    },
    "evaluation": {
        "deterministic": True,
        "seeds": [42],
    },
    "logging": {
        "base_dir": "runs",
    },
    "modules": {
        "action": None,
        "state": None,
        "reward": None,
    },
}


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``updates`` into ``base``."""

    for key, value in updates.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    """Load YAML/JSON config, falling back to defaults."""

    merged = copy.deepcopy(DEFAULT_CONFIG)
    if not config_path:
        return merged

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        if path.suffix.lower() in {".json"}:
            user_cfg = json.load(fh)
        else:
            user_cfg = yaml.safe_load(fh) or {}

    if not isinstance(user_cfg, dict):
        raise ValueError("Configuration root must be a mapping")

    deep_update(merged, user_cfg)
    return merged


def get_config_value(
    config: Dict[str, Any],
    path: Iterable[str],
    default: Any,
) -> Any:
    """Retrieve a nested value from a dict."""

    current: Any = config
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_yaml(data: Dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)


def save_json(data: Dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
