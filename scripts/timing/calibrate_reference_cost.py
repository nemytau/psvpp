#!/usr/bin/env python3
"""Measure what the time-fair reference costs on given instances (to plan an experiment's runtime).

Runs Kisialiou's ALNS exactly as the reference does (5 restarts x 300 iterations, search CPU
time measured in the Rust engine) on each named instance and reports search CPU seconds, wall
seconds and the repair-operator mix of the run (to check that the pool really contains
first-fit insertion). Run several copies in parallel to measure under the 4-process load the
experiments use.

    PSVPP_GENERATED_ROOT=... PSVPP_PROCESSED_ROOT=... python scripts/timing/calibrate_reference_cost.py \
        --size large --instances large_test_tw050_df3_01 large_test_tw000_df1_02 --out calib.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl.dataset_manager import GeneratedDatasetManager  # noqa: E402
from rl.episode_runner import run_kisialiou_episode  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", required=True, choices=["small", "medium", "large"])
    parser.add_argument("--instances", nargs="+", required=True, help="test instance directory names")
    parser.add_argument("--split", default="test")
    parser.add_argument("--baseline-action-module", default="adaptive_weighted_v2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None, help="append one JSON line per instance")
    args = parser.parse_args()

    paths = {Path(p).name: p for p in GeneratedDatasetManager().prepare_size(args.size)[args.split]}
    for name in args.instances:
        log_dir = Path(tempfile.mkdtemp(prefix="calib_"))
        started = time.time()
        stats, rows = run_kisialiou_episode(
            dataset_path=paths[name],
            seed=args.seed,
            max_iterations=200000,
            restart_max_iterations=300,
            log_dir=log_dir,
            mode_label="kisialiou",
            size_label=args.size,
            action_module=args.baseline_action_module,
            state_module="features_v3",
            reward_module="delta_cost_v9",
            target_runtime_seconds=None,
            chain_max_consecutive_improvements=3,
            reference_restarts=5,
            progress_horizon=300,
        )
        wall = time.time() - started
        repair = Counter()
        for csv_path in log_dir.glob("*.csv"):
            with csv_path.open() as handle:
                for row in csv.DictReader(handle):
                    if row.get("operator_type") == "repair":
                        repair[row["operator_name"]] += 1
        record = dict(
            size=args.size,
            instance=name,
            search_cpu_seconds=stats.search_cpu_seconds,
            solver_seconds=stats.solver_runtime_seconds,
            wall_seconds=wall,
            best_cost=stats.best_cost,
            initial_cost=stats.initial_cost,
            restarts=len(rows),
            repair_mix=dict(repair),
        )
        print(json.dumps(record), flush=True)
        if args.out:
            with open(args.out, "a") as handle:
                handle.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
