#!/usr/bin/env python3
"""Utility for generating train/test datasets from ALNS configuration templates."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import yaml

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from py_alns.data_generator import (
    generate_base,
    generate_installation_dataframe,
    generate_vessels_dataframe,
)


DATASET_CONFIGS: Dict[str, Dict[str, Iterable[str]]] = {
    "small": {
        "train": ["SMALL_TRAIN_1", "SMALL_TRAIN_2", "SMALL_TRAIN_3"],
        "test": ["SMALL_TEST_1"],
    },
    "medium": {
        "train": ["MEDIUM_TRAIN_1", "MEDIUM_TRAIN_2", "MEDIUM_TRAIN_3"],
        "test": ["MEDIUM_TEST_1"],
    },
    "large": {
        "train": ["LARGE_TRAIN_1", "LARGE_TRAIN_2", "LARGE_TRAIN_3"],
        "test": ["LARGE_TEST_1"],
    },
}

DEFAULT_SAMPLES_PER_CONFIG = 5
OUTPUT_ROOT = Path("data/generated")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def to_native(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: to_native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_native(v) for v in value]
    return value


def generate_samples(samples_per_config: int) -> None:
    base_template = generate_base()
    longitude = float(getattr(base_template, "longitude", 0.0))
    latitude = float(getattr(base_template, "latitude", 0.0))
    base_info = {
        "name": getattr(base_template, "name", ""),
        "service_time": getattr(base_template, "service_time", 0),
        "time_window": list(getattr(base_template, "time_window", [])),
        "longitude": longitude,
        "latitude": latitude,
        "location": [longitude, latitude],
    }
    summary: Dict[str, Dict[str, int]] = {}

    config_names = [
        name for groups in DATASET_CONFIGS.values() for configs in groups.values() for name in configs
    ]

    for cfg_index, config_name in enumerate(config_names):
        summary[config_name] = {"samples": samples_per_config}
        base_seed = 1000 + cfg_index * 100

        # Determine size/split for folder structure
        size_name = next(size for size, groups in DATASET_CONFIGS.items() if any(
            config_name in cfgs for cfgs in groups.values()
        ))
        split_name = next(split for split, cfgs in DATASET_CONFIGS[size_name].items() if config_name in cfgs)

        for sample_id in range(1, samples_per_config + 1):
            seed = base_seed + sample_id
            np.random.seed(seed)

            inst_df = generate_installation_dataframe(config_name)
            vessel_df = generate_vessels_dataframe(config_name)

            sample_dir = OUTPUT_ROOT / size_name / split_name / f"{config_name.lower()}_{sample_id:02d}"
            ensure_dir(sample_dir)

            inst_df.to_csv(sample_dir / "installations.csv", index=False)
            vessel_df.to_csv(sample_dir / "vessels.csv", index=False)

            meta = {
                "config": config_name,
                "size": size_name,
                "split": split_name,
                "seed": seed,
                "installations": len(inst_df),
                "vessels": len(vessel_df),
                "base": to_native(base_info),
            }

            with (sample_dir / "meta.yaml").open("w", encoding="utf-8") as meta_file:
                yaml.safe_dump(to_native(meta), meta_file, sort_keys=False)

            print(
                f"Generated {size_name}/{split_name} sample {config_name} #{sample_id} "
                f"(seed={seed}, installs={len(inst_df)}, vessels={len(vessel_df)})"
            )

    summary_path = OUTPUT_ROOT / "SUMMARY.yaml"
    ensure_dir(summary_path.parent)
    with summary_path.open("w", encoding="utf-8") as summary_file:
        yaml.safe_dump(summary, summary_file, sort_keys=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ALNS training/test datasets.")
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES_PER_CONFIG,
        help="Number of samples to generate per configuration (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_samples(samples_per_config=max(1, args.samples))


if __name__ == "__main__":
    main()
