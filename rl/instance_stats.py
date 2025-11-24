"""Utilities for extracting per-instance statistics for the ALNS RL environment."""

from __future__ import annotations

import ast
import csv
import math
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

EARTH_RADIUS_KM = 6371.0
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def get_instance_statistics(problem_instance: str) -> Dict[str, float]:
    """Return cached statistics for the provided dataset identifier or path."""

    return dict(_compute_instance_statistics(problem_instance))


@lru_cache(maxsize=256)
def _compute_instance_statistics(problem_instance: str) -> Dict[str, float]:
    install_path, vessel_path, base_path = _resolve_dataset_files(problem_instance)

    installations = list(_read_installations(install_path))
    vessels = list(_read_csv_dicts(vessel_path))
    base_location = _read_base_location(base_path)

    num_installations = len(installations)
    total_visits = sum(inst.visit_count for inst in installations)
    total_demand = sum(inst.deck_demand * inst.visit_count for inst in installations)
    num_vessels = len(vessels)

    distances = [
        _haversine_km(base_location, inst.location)
        for inst in installations
        if inst.location is not None and base_location is not None
    ]
    avg_distance = float(sum(distances) / len(distances)) if distances else 0.0
    max_distance = float(max(distances)) if distances else 0.0

    return {
        "num_installations": float(num_installations),
        "total_visits": float(total_visits),
        "num_vessels": float(num_vessels),
        "total_deck_demand": float(total_demand),
        "avg_distance_km": avg_distance,
        "max_distance_km": max_distance,
    }


def _resolve_dataset_files(problem_instance: str) -> Tuple[Path, Path, Path]:
    """Resolve dataset CSV file paths for installations, vessels, and base."""

    candidate_paths = _candidate_paths(problem_instance)
    for candidate in candidate_paths:
        resolved = _resolve_from_path(candidate)
        if resolved is not None:
            return resolved

    sample_resolved = _resolve_sample_dataset(problem_instance)
    if sample_resolved is not None:
        return sample_resolved

    raise FileNotFoundError(f"Unable to resolve dataset files for '{problem_instance}'.")


def _candidate_paths(problem_instance: str) -> Iterable[Path]:
    candidate = Path(problem_instance)
    yield candidate
    if not candidate.is_absolute():
        yield PROJECT_ROOT / candidate


def _resolve_from_path(path: Path) -> Optional[Tuple[Path, Path, Path]]:
    if not path.exists():
        return None

    if path.is_dir():
        directory = path
    else:
        directory = path.parent

    install = directory / "installations.csv"
    vessels = directory / "vessels.csv"
    base = directory / "base.csv"
    if install.exists() and vessels.exists() and base.exists():
        return install, vessels, base

    # Handle legacy sample layout: sample/installations/<name>/i_*.csv
    if directory.parent.name in {"installations", "vessels", "base"}:
        maybe_sample = _resolve_sample_from_directory(directory)
        if maybe_sample is not None:
            return maybe_sample

    return None


def _resolve_sample_dataset(problem_instance: str) -> Optional[Tuple[Path, Path, Path]]:
    dataset_name = Path(problem_instance).name
    sample_root = PROJECT_ROOT / "sample"
    if not sample_root.exists():
        return None

    install_dir = sample_root / "installations" / dataset_name
    vessel_dir = sample_root / "vessels" / dataset_name
    base_dir = sample_root / "base" / dataset_name

    try:
        install_file = _first_csv(install_dir)
        vessel_file = _first_csv(vessel_dir)
        base_file = _first_csv(base_dir)
    except FileNotFoundError:
        return None

    return install_file, vessel_file, base_file


def _resolve_sample_from_directory(directory: Path) -> Optional[Tuple[Path, Path, Path]]:
    if directory.parent.name not in {"installations", "vessels", "base"}:
        return None

    dataset_name = directory.name
    sample_root = directory.parent.parent
    install_dir = sample_root / "installations" / dataset_name
    vessel_dir = sample_root / "vessels" / dataset_name
    base_dir = sample_root / "base" / dataset_name

    try:
        install_file = _first_csv(install_dir)
        vessel_file = _first_csv(vessel_dir)
        base_file = _first_csv(base_dir)
    except FileNotFoundError:
        return None

    return install_file, vessel_file, base_file


def _first_csv(path: Path) -> Path:
    if path.is_file() and path.suffix.lower() == ".csv":
        return path
    if path.is_dir():
        csv_files = sorted(child for child in path.iterdir() if child.suffix.lower() == ".csv")
        if csv_files:
            return csv_files[0]
    raise FileNotFoundError(f"No CSV files found under {path}.")


class _InstallationRecord:
    __slots__ = ("deck_demand", "visit_count", "location")

    def __init__(self, deck_demand: float, visit_count: float, location: Optional[Tuple[float, float]]) -> None:
        self.deck_demand = deck_demand
        self.visit_count = visit_count
        self.location = location


def _read_installations(path: Path):
    with path.open("r", encoding="utf-8", newline="") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            deck_demand = _to_float(row.get("deck_demand"), default=0.0)
            visit_count = _to_float(row.get("visit_frequency"), default=0.0)
            location = _parse_location(row.get("location"))
            yield _InstallationRecord(deck_demand, visit_count, location)


def _read_csv_dicts(path: Path):
    with path.open("r", encoding="utf-8", newline="") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            yield row


def _read_base_location(path: Path) -> Optional[Tuple[float, float]]:
    with path.open("r", encoding="utf-8", newline="") as infile:
        reader = csv.DictReader(infile)
        row = next(reader, None)
    if not row:
        return None

    location = _parse_location(row.get("location"))
    if location is not None:
        return location

    lon = _to_float(row.get("longitude"))
    lat = _to_float(row.get("latitude"))
    if math.isfinite(lat) and math.isfinite(lon):
        return (lat, lon)

    return None


def _parse_location(value: Optional[str]) -> Optional[Tuple[float, float]]:
    if value is None:
        return None
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return None
    if isinstance(parsed, (list, tuple)) and len(parsed) >= 2:
        lat = _to_float(parsed[0])
        lon = _to_float(parsed[1])
        if math.isfinite(lat) and math.isfinite(lon):
            return (lat, lon)
    return None


def _to_float(value: Optional[str], default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _haversine_km(coord_a: Optional[Tuple[float, float]], coord_b: Optional[Tuple[float, float]]) -> float:
    if coord_a is None or coord_b is None:
        return 0.0

    lat1, lon1 = coord_a
    lat2, lon2 = coord_b

    if not all(math.isfinite(val) for val in (lat1, lon1, lat2, lon2)):
        return 0.0

    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    sin_dlat = math.sin(dlat / 2.0)
    sin_dlon = math.sin(dlon / 2.0)
    a = sin_dlat * sin_dlat + math.cos(lat1_rad) * math.cos(lat2_rad) * sin_dlon * sin_dlon
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return EARTH_RADIUS_KM * c
