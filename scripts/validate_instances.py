#!/usr/bin/env python3
"""Smoke-validate generated instances before spending compute on them.

For every instance of the chosen sizes/splits: build the Rust engine, construct the initial
solution for --restarts different restart seeds (the same seeds an evaluation uses: seed +
restart * 1009), then run --steps ALNS iterations on the last one. A Rust panic (e.g. the
constructive heuristic failing to find a feasible schedule) or an incomplete/infeasible
solution is reported per instance. Also prints initial-solution cost, fleet size and fixed /
variable cost shares, a cheap sanity check of the cost model.

    PSVPP_GENERATED_ROOT=... PSVPP_PROCESSED_ROOT=... python scripts/validate_instances.py \
        --sizes small medium large --splits train test
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl.dataset_manager import GeneratedDatasetManager  # noqa: E402
from rl.rl_alns_environment import ALNSEnvironment  # noqa: E402


def validate(path: str, size: str, restarts: int, steps: int, seed: int):
    problems = []
    initial_costs, fixed_shares, vessels = [], [], []
    env = None
    for restart in range(restarts):
        restart_seed = seed + restart * 1009
        env = ALNSEnvironment(
            problem_instance=path,
            max_iterations=max(steps, 1),
            progress_horizon=300,
            seed=restart_seed,
            problem_instance_paths=[path],
            problem_sampling_strategy="round_robin",
            enable_operator_logging=False,
            action_module="adaptive_weighted_v2",
            state_module="features_v3",
            reward_module="delta_cost_v9",
            algorithm_mode="baseline",
            size_bucket="small" if size == "small" else "large",
        )
        try:
            env.reset(seed=restart_seed, options={"problem_path": path})
        except BaseException as exc:  # pyo3 PanicException derives from BaseException
            problems.append(f"restart {restart} (seed {restart_seed}): {type(exc).__name__}: {exc}")
            continue
        initial_costs.append(float(env.initial_cost))
        vessels.append(int(env.initial_num_vessels_used or 0))
        fixed = float(env.initial_fixed_cost or 0.0)
        fixed_shares.append(fixed / float(env.initial_cost))
    if env is not None and not problems:
        try:
            for _ in range(steps):
                env.step(0)
        except BaseException as exc:
            problems.append(f"iteration run: {type(exc).__name__}: {exc}")
    return problems, initial_costs, fixed_shares, vessels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", nargs="+", default=["small", "medium", "large"])
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument("--restarts", type=int, default=20)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manager = GeneratedDatasetManager()
    failures = 0
    for size in args.sizes:
        splits = manager.prepare_size(size)
        for split in args.splits:
            costs, shares, fleets = [], [], []
            started = time.time()
            for path in splits[split]:
                problems, c, s, v = validate(path, size, args.restarts, args.steps, args.seed)
                costs += c
                shares += s
                fleets += v
                if problems:
                    failures += 1
                    print(f"FAIL {Path(path).name}")
                    for line in problems[:3]:
                        print(f"     {line}")
            if costs:
                print(
                    f"{size:6s} {split:5s} {len(splits[split]):3d} instances ok "
                    f"| initial cost mean {statistics.mean(costs) / 1e6:6.2f}M NOK "
                    f"| vessels {statistics.mean(fleets):4.1f} (max {max(fleets)}) "
                    f"| fixed share {statistics.mean(shares) * 100:4.1f}% "
                    f"| {time.time() - started:.0f}s"
                )
    print(f"instances with problems: {failures}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
