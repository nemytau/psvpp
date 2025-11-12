"""Utilities for preparing generated datasets for ALNS training and evaluation."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import yaml


class GeneratedDatasetManager:
    """Prepare generated train/test datasets for the Rust ALNS engine."""

    def __init__(
        self,
        generated_root: Path | str = Path("data/generated"),
        processed_root: Path | str = Path("data/processed/alns"),
    ) -> None:
        self.generated_root = Path(generated_root)
        self.processed_root = Path(processed_root)

    def prepare_size(self, size: str) -> Dict[str, List[str]]:
        """Prepare train/test splits for the requested dataset size."""
        size_root = self.generated_root / size
        if not size_root.exists():
            raise FileNotFoundError(f"Generated dataset folder not found: {size_root}")

        splits: Dict[str, List[str]] = {}
        for split in ("train", "test"):
            split_root = size_root / split
            if not split_root.exists():
                raise FileNotFoundError(f"Missing split '{split}' in {size_root}")

            prepared_dirs = [
                self._prepare_sample_dir(dir_path)
                for dir_path in sorted(d for d in split_root.iterdir() if d.is_dir())
            ]
            if not prepared_dirs:
                raise RuntimeError(f"No datasets found under {split_root}")
            splits[split] = prepared_dirs
        return splits

    def _prepare_sample_dir(self, source_dir: Path) -> str:
        relative = source_dir.relative_to(self.generated_root)
        target_dir = self.processed_root / relative
        target_dir.mkdir(parents=True, exist_ok=True)

        processed_files = {
            "installations": target_dir / "installations.csv",
            "vessels": target_dir / "vessels.csv",
            "base": target_dir / "base.csv",
        }

        if not all(path.exists() for path in processed_files.values()):
            self._convert_sample(source_dir, target_dir, processed_files)

        return str(target_dir.resolve())

    def _convert_sample(
        self,
        source_dir: Path,
        target_dir: Path,
        processed_files: Dict[str, Path],
    ) -> None:
        installations_path = source_dir / "installations.csv"
        vessels_path = source_dir / "vessels.csv"
        meta_path = source_dir / "meta.yaml"

        if not installations_path.exists() or not vessels_path.exists():
            raise FileNotFoundError(
                f"Dataset {source_dir} is missing required CSV files"
            )
        if not meta_path.exists():
            raise FileNotFoundError(f"Dataset {source_dir} is missing meta.yaml")

        self._convert_installations(installations_path, processed_files["installations"])
        self._convert_vessels(vessels_path, processed_files["vessels"])
        self._convert_base(meta_path, processed_files["base"])

    def _convert_installations(self, source: Path, destination: Path) -> None:
        fieldnames = [
            "idx",
            "name",
            "inst_type",
            "deck_demand",
            "visit_frequency",
            "location",
            "departure_spread",
            "deck_service_speed",
            "time_window",
            "service_time",
        ]

        with source.open("r", encoding="utf-8") as infile:
            reader = csv.DictReader(infile)
            rows = list(reader)

        if not rows:
            raise RuntimeError(f"No installation records found in {source}")

        for idx, row in enumerate(rows, start=1):
            deck_demand = float(row.get("deck_demand", 0.0))
            service_speed = float(row.get("deck_service_speed", 0.0))
            service_time = deck_demand / service_speed if service_speed else deck_demand
            row["idx"] = idx
            row["location"] = f"[{row.get('longitude', 0.0)}, {row.get('latitude', 0.0)}]"
            row["service_time"] = _round_value(service_time)

        with destination.open("w", encoding="utf-8", newline="") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in fieldnames})

    def _convert_vessels(self, source: Path, destination: Path) -> None:
        fieldnames = [
            "idx",
            "name",
            "deck_capacity",
            "bulk_capacity",
            "speed",
            "vessel_type",
            "fcs",
            "fcw",
            "cost",
        ]

        with source.open("r", encoding="utf-8") as infile:
            reader = csv.DictReader(infile)
            rows = list(reader)

        if not rows:
            raise RuntimeError(f"No vessel records found in {source}")

        for idx, row in enumerate(rows):
            row.setdefault("idx", idx)

        with destination.open("w", encoding="utf-8", newline="") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in fieldnames})

    def _convert_base(self, meta_path: Path, destination: Path) -> None:
        with meta_path.open("r", encoding="utf-8") as meta_file:
            meta = yaml.safe_load(meta_file) or {}

        base_info = meta.get("base") or {}
        time_window = base_info.get("time_window", [0, 24])
        if isinstance(time_window, Iterable):
            time_window = list(time_window)
        if len(time_window) != 2:
            raise ValueError(f"Invalid time_window in {meta_path}: {time_window}")

        fieldnames = [
            "name",
            "idx",
            "service_time",
            "time_window",
            "longitude",
            "latitude",
            "location",
        ]

        location_long = base_info.get("longitude", 0.0)
        location_lat = base_info.get("latitude", 0.0)

        row = {
            "name": base_info.get("name", "BASE"),
            "idx": 0,
            "service_time": base_info.get("service_time", 8),
            "time_window": f"({time_window[0]}, {time_window[1]})",
            "longitude": location_long,
            "latitude": location_lat,
            "location": f"[{location_long}, {location_lat}]",
        }

        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8", newline="") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(row)


def _round_value(value: float, digits: int = 4) -> float:
    if math.isfinite(value):
        return round(value, digits)
    return 0.0
