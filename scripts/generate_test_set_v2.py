#!/usr/bin/env python3
"""Diversified test sets for the Experiment 8 rerun (2026-10): 50 instances per size, each its
own configuration (the earlier test sets were 50 draws from one `{SIZE}_TEST_1` config).

Per size, a full factorial of
  * time-window density t in {0, 25, 50, 75, 100} % of the installations (a literal count per
    installation type, like the `{SIZE}_TRAIN_TW1..5` configs),
  * demand / visit-frequency level DF1..DF5 (the parameter points of `{SIZE}_TRAIN_DF1..5`:
    small demand + frequent visits -> large demand + infrequent visits; weekly demand per
    installation stays roughly constant),
  * 2 replicates,
with the number of installations spread over the size tier's range (small 6-12, medium 12-20,
large 20-40) by a Latin-square assignment of 10 equally spaced quantiles within each DF level, so
every time-window level sees every installation-count stratum (large DF1/DF2 are capped at 28
installations, see N_RANGE_OVERRIDES). Fleet size (large: 18 vessels; only vessels used are paid), speeds
and capacities are the tier's train/test values; the tight time window is 7-19 for every
instance so that only the density varies.

Output (under --output-root, default $PSVPP_GENERATED_ROOT or data/generated):
    <size>/test/<size>_test_tw<TTT>_df<D>_<RR>/{installations.csv,vessels.csv,meta.yaml}
    test_set_v2_design.csv       one row per instance: factors + realised properties

    python scripts/generate_test_set_v2.py --output-root data/rerun26/generated
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from py_alns.data_generator import (  # noqa: E402
    generation_yaml_config,  # the dict the generator functions read (py_alns.data_generator loads its own copy)
    generate_base,
    generate_installation_dataframe,
    generate_vessels_dataframe,
)

SIZE_SPECS: Dict[str, Dict] = {
    "small": dict(prefix="SMALL", n_range=(6, 12), vessels=9, speeds=(9, 10), index=0),
    "medium": dict(prefix="MEDIUM", n_range=(12, 20), vessels=9, speeds=(8, 9), index=1),
    "large": dict(prefix="LARGE", n_range=(20, 40), vessels=18, speeds=(7, 8), index=2),
}

TW_LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)

# Same parameter points as {SIZE}_TRAIN_DF1..5 (drill, prod).
DF_LEVELS: Dict[int, Dict] = {
    1: dict(mu=(18, 14), sigma=(4, 3), freq=([3, 4], [0.4, 0.6]), spread=0),
    2: dict(mu=(24, 19), sigma=(5, 4), freq=([2, 3, 4], [0.2, 0.3, 0.5]), spread=0),
    3: dict(mu=(30, 24), sigma=(7, 5), freq=([1, 2, 3], [0.2, 0.4, 0.4]), spread=1),
    4: dict(mu=(37, 30), sigma=(8, 7), freq=([1, 2, 3], [0.4, 0.4, 0.2]), spread=1),
    5: dict(mu=(45, 38), sigma=(10, 9), freq=([1, 2], [0.7, 0.3]), spread=1),
}

# Installation-count range per (size, DF level) where it differs from the tier's range. Large DF1/DF2
# (3-4 visits per installation) with up to 40 installations give 100-140 visits per week: far above the
# earlier large test set (82), the train sets (52-83) and ~85% of the total evaluation time (a
# reference run is ~visits^3.5-4). Capped at 28 installations (<= ~100 visits); use
# --no-cap to restore the tier's full range.
N_RANGE_OVERRIDES: Dict[tuple, tuple] = {("large", 1): (20, 28), ("large", 2): (20, 28)}

DRILL_SHARE = 0.35
TIGHT_WINDOW = "7, 19"
SEED_BASE = 880_000  # far from the train seeds (1000 + 100 * config index + sample)
REPLICATES = 2


def design(size: str) -> List[Dict]:
    """Per DF level, the 10 (TW level x replicate) cells take the installation counts of 10 equally
    spaced quantiles of that level's range; the stratum of a TW level is rotated with the DF level
    (Latin square), so every TW level sees every stratum once across the DF levels."""
    spec = SIZE_SPECS[size]
    rows: List[Dict] = []
    for i, tw in enumerate(TW_LEVELS):
        for j, df in enumerate(sorted(DF_LEVELS)):
            lo, hi = N_RANGE_OVERRIDES.get((size, df), spec["n_range"])
            stratum = (i + 2 * j) % 5
            for r in range(REPLICATES):
                n = lo + int(round((hi - lo) * (2 * stratum + r + 0.5) / (5 * REPLICATES)))
                n_drill = max(1, int(round(DRILL_SHARE * n)))
                n_prod = n - n_drill
                name = f"{size}_test_tw{int(round(tw * 100)):03d}_df{df}_{r + 1:02d}"
                rows.append(
                    dict(
                        size=size,
                        name=name,
                        config_name=name.upper(),
                        tw_level=tw,
                        df_level=df,
                        replicate=r + 1,
                        n_installations=n,
                        n_drill=n_drill,
                        n_prod=n_prod,
                        seed=SEED_BASE + 10_000 * spec["index"] + 100 * (5 * i + j) + r,
                    )
                )
    return rows


def _register_config(row: Dict) -> None:
    """Add this instance's installation + fleet configuration to the in-memory generator config."""
    spec = SIZE_SPECS[row["size"]]
    df = DF_LEVELS[row["df_level"]]
    blocks = []
    for k, (typ, prefix, num) in enumerate((("drill", "d", row["n_drill"]), ("prod", "p", row["n_prod"]))):
        blocks.append(
            dict(
                type=typ,
                num=int(num),
                time_windows=int(math.floor(row["tw_level"] * num + 0.5)),
                demand_mu=df["mu"][k],
                demand_sigma=df["sigma"][k],
                frequencies=dict(values=list(df["freq"][0]), probs=list(df["freq"][1])),
                name_prefix=prefix,
                default_time_window=TIGHT_WINDOW,
                departure_spread=df["spread"],
                deck_service_speed=spec["speeds"][k],
            )
        )
    reference_fleet = generation_yaml_config["fleet_generation_params"][f"{spec['prefix']}_TEST_1"][0]
    fleet = [dict(reference_fleet, num=spec["vessels"])]
    generation_yaml_config["inst_generation_params"][row["config_name"]] = blocks
    generation_yaml_config["fleet_generation_params"][row["config_name"]] = fleet


def _to_native(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _to_native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_native(v) for v in value]
    return value


def generate(output_root: Path, sizes: List[str]) -> List[Dict]:
    base = generate_base()
    longitude, latitude = float(base.longitude), float(base.latitude)
    base_info = {
        "name": base.name,
        "service_time": base.service_time,
        "time_window": list(base.time_window),
        "longitude": longitude,
        "latitude": latitude,
        "location": [longitude, latitude],
    }
    design_rows: List[Dict] = []
    for size in sizes:
        for row in design(size):
            _register_config(row)
            np.random.seed(row["seed"])
            inst_df = generate_installation_dataframe(row["config_name"])
            vessel_df = generate_vessels_dataframe(row["config_name"])

            sample_dir = output_root / size / "test" / row["name"]
            sample_dir.mkdir(parents=True, exist_ok=True)
            inst_df.to_csv(sample_dir / "installations.csv", index=False)
            vessel_df.to_csv(sample_dir / "vessels.csv", index=False)
            meta = {
                "config": row["config_name"],
                "size": size,
                "split": "test",
                "seed": row["seed"],
                "installations": len(inst_df),
                "vessels": len(vessel_df),
                "tw_level": row["tw_level"],
                "df_level": row["df_level"],
                "replicate": row["replicate"],
                "base": _to_native(base_info),
            }
            with (sample_dir / "meta.yaml").open("w", encoding="utf-8") as handle:
                yaml.safe_dump(_to_native(meta), handle, sort_keys=False)

            freq = inst_df["visit_frequency"].astype(int)
            demand = inst_df["deck_demand"].astype(float)
            tight = inst_df["time_window"].apply(lambda w: tuple(w) != (0, 24))
            design_rows.append(
                dict(
                    row,
                    vessels=len(vessel_df),
                    visits=int(freq.sum()),
                    mean_demand=round(float(demand.mean()), 3),
                    mean_frequency=round(float(freq.mean()), 3),
                    weekly_demand=round(float((demand * freq).sum()), 1),
                    tight_share=round(float(tight.mean()), 4),
                )
            )
            print(f"{row['name']}: n={len(inst_df)} visits={int(freq.sum())} tight={tight.mean():.2f} seed={row['seed']}")
    return design_rows


def write_design(output_root: Path, rows: List[Dict]) -> Path:
    path = output_root / "test_set_v2_design.csv"
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the diversified test sets (50 per size).")
    parser.add_argument("--output-root", default=os.environ.get("PSVPP_GENERATED_ROOT", "data/generated"))
    parser.add_argument("--sizes", nargs="+", default=list(SIZE_SPECS), choices=list(SIZE_SPECS))
    parser.add_argument("--print-design", action="store_true", help="Print the design only; write nothing")
    parser.add_argument("--no-cap", action="store_true", help="Use the tier's full installation range in every DF cell")
    parser.add_argument(
        "--vessels",
        nargs="*",
        default=[],
        metavar="SIZE=N",
        help="Override the fleet size of a tier, e.g. --vessels large=18 (default: small 9, medium 9, large 18)",
    )
    args = parser.parse_args()
    if args.no_cap:
        N_RANGE_OVERRIDES.clear()
    for item in args.vessels:
        size_name, count = item.split("=")
        SIZE_SPECS[size_name]["vessels"] = int(count)

    if args.print_design:
        for size in args.sizes:
            for row in design(size):
                print(row)
        return
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    rows = generate(root, args.sizes)
    print(f"design table: {write_design(root, rows)}")


if __name__ == "__main__":
    main()
