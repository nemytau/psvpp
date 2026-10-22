#!/usr/bin/env python3
"""Machine-speed probe for time-budgeted experiments.

Runs one fixed, deterministic piece of search work (one Kisialiou restart, 300 iterations, seed 42,
on the first small test instance) and prints one CSV line: timestamp, search CPU seconds, wall
seconds, busy processes. Identical work, so changes in the CPU time are changes in machine speed
(load, thermal throttling). The Experiment 8 rerun queue calls it every 30 minutes and appends to
logs/rerun26_queue/speed_probe.csv, to document that the references and the evaluations (days
apart) ran at the same machine speed.

    PSVPP_GENERATED_ROOT=... PSVPP_PROCESSED_ROOT=... python scripts/timing/speed_probe.py --busy 4
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl.dataset_manager import GeneratedDatasetManager  # noqa: E402
from rl.episode_runner import run_kisialiou_episode  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--busy", type=int, default=-1, help="number of busy processes right now (for the log line)")
    parser.add_argument("--header", action="store_true", help="print the CSV header and exit")
    args = parser.parse_args()
    if args.header:
        print("timestamp,search_cpu_seconds,wall_seconds,busy_processes")
        return
    path = GeneratedDatasetManager().prepare_size("small")["test"][0]
    started = time.time()
    stats, _ = run_kisialiou_episode(
        dataset_path=path,
        seed=42,
        max_iterations=200000,
        restart_max_iterations=300,
        log_dir=Path(tempfile.mkdtemp(prefix="probe_")),
        mode_label="probe",
        size_label="small",
        action_module="adaptive_weighted_v2",
        state_module="features_v3",
        reward_module="delta_cost_v9",
        target_runtime_seconds=None,
        chain_max_consecutive_improvements=3,
        reference_restarts=1,
        progress_horizon=300,
    )
    print(f"{datetime.now().isoformat(timespec='seconds')},{stats.search_cpu_seconds:.4f},{time.time() - started:.3f},{args.busy}")


if __name__ == "__main__":
    main()
