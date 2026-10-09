#!/usr/bin/env python3
"""Utility for generating train/test datasets from ALNS configuration templates."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

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


def _train_configs(size_prefix: str) -> list:
    """Original 3 hand-tuned configs + the TW1-5/DF1-5 variety axes added 2026-09-04.

    TW1-5: time-window density axis (0%, 25%, 50%, 75%, 100% of installations get a
    real time window), install count spread across the size tier's requested range.
    DF1-5: demand+frequency spectrum (small-demand+frequent -> large-demand+infrequent,
    3 mixes between), same install-count spread. See
    py_alns/resource/generation_config.yaml and
    docs/experiments/diary_entry_decisive_time_scaled_reward_test_launch.md.
    """
    base = [f"{size_prefix}_TRAIN_{i}" for i in (1, 2, 3)]
    tw = [f"{size_prefix}_TRAIN_TW{i}" for i in range(1, 6)]
    df = [f"{size_prefix}_TRAIN_DF{i}" for i in range(1, 6)]
    return base + tw + df


DATASET_CONFIGS: Dict[str, Dict[str, Iterable[str]]] = {
    "small": {
        "train": _train_configs("SMALL"),
        "test": ["SMALL_TEST_1"],
    },
    "medium": {
        "train": _train_configs("MEDIUM"),
        "test": ["MEDIUM_TEST_1"],
    },
    "large": {
        "train": _train_configs("LARGE"),
        "test": ["LARGE_TEST_1"],
    },
}

CONFIG_LOOKUP = {
    config_name: (size, split)
    for size, splits in DATASET_CONFIGS.items()
    for split, config_names in splits.items()
    for config_name in config_names
}

CONFIG_SEQUENCE = list(CONFIG_LOOKUP)

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


def generate_samples(
    samples_per_config: int,
    sizes: Optional[Sequence[str]] = None,
    only_split: Optional[str] = None,
) -> None:
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

    size_filter = set(sizes) if sizes else set(DATASET_CONFIGS.keys())
    known_sizes = set(DATASET_CONFIGS.keys())
    unknown_sizes = size_filter - known_sizes
    if unknown_sizes:
        raise ValueError(f"Unknown dataset sizes requested: {sorted(unknown_sizes)}")

    selected_configs = [
        name for name in CONFIG_SEQUENCE if CONFIG_LOOKUP[name][0] in size_filter
    ]
    if only_split:
        selected_configs = [
            name for name in selected_configs if CONFIG_LOOKUP[name][1] == only_split
        ]

    for cfg_index, config_name in enumerate(selected_configs):
        summary[config_name] = {"samples": samples_per_config}
        base_seed = 1000 + cfg_index * 100

        # Determine size/split for folder structure
        size_name, split_name = CONFIG_LOOKUP[config_name]

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
    parser.add_argument(
        "--sizes",
        nargs="+",
        choices=sorted(DATASET_CONFIGS.keys()),
        help="Restrict generation to the provided dataset sizes",
    )
    parser.add_argument(
        "--only-split",
        choices=("train", "test"),
        default=None,
        help="Restrict generation to only the train or only the test split",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_samples(
        samples_per_config=max(1, args.samples),
        sizes=args.sizes,
        only_split=args.only_split,
    )


if __name__ == "__main__":
    main()
