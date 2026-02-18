#!/usr/bin/env python3
"""Run a short ALNS baseline rollout over every generated dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl.dataset_manager import GeneratedDatasetManager
from rl.train_alns_rl import normalize_algorithm_mode, run_episode_with_policy


def _available_sizes(root: Path) -> List[str]:
    candidates = ["small", "medium", "large", "mixed"]
    return [size for size in candidates if (root / size).exists()]


def _iterate_datasets(
    sizes: Iterable[str],
    iterations: int,
    seed: int,
    algorithm_mode: str,
) -> Tuple[int, List[Tuple[str, str, str, str]]]:
    manager = GeneratedDatasetManager()
    total_runs = 0
    failures: List[Tuple[str, str, str, str]] = []

    for size in sizes:
        try:
            splits = manager.prepare_size(size)
        except Exception as exc:  # pragma: no cover - defensive guard
            failures.append((size, "-", "<prepare>", str(exc)))
            continue

        for split, paths in splits.items():
            for path in paths:
                total_runs += 1
                problem_path = Path(path)
                try:
                    run_episode_with_policy(
                        policy_model=None,
                        problem_path=str(problem_path),
                        seed=seed,
                        max_iterations=iterations,
                        deterministic=False,
                        algorithm_mode=algorithm_mode,
                    )
                except Exception as exc:
                    failures.append((size, split, str(problem_path), str(exc)))
    return total_runs, failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test ALNS baseline over all datasets")
    parser.add_argument(
        "--sizes",
        nargs="*",
        default=None,
        help="Dataset sizes to include (default: discover available sizes)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of ALNS iterations per dataset (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Seed forwarded to the environment (default: %(default)s)",
    )
    parser.add_argument(
        "--algorithm-mode",
        default="baseline",
        choices=["baseline", "kisialiou", "reinforcement_learning", "rl"],
        help="High-level ALNS variant (baseline, kisialiou, reinforcement_learning)",
    )
    args = parser.parse_args()

    generated_root = REPO_ROOT / "data" / "generated"
    if args.sizes:
        sizes = args.sizes
    else:
        sizes = _available_sizes(generated_root)
        if not sizes:
            raise SystemExit(f"No generated datasets found under {generated_root}")

    print(f"Running ALNS baseline for sizes: {', '.join(sizes)}")
    algorithm_mode = normalize_algorithm_mode(args.algorithm_mode)
    total_runs, failures = _iterate_datasets(sizes, args.iterations, args.seed, algorithm_mode)

    print(f"Completed {total_runs} dataset rollouts with {len(failures)} failures")
    if failures:
        print("Failed datasets:")
        for size, split, path, error in failures:
            print(f"  [{size}/{split}] {path}")
            print(f"      -> {error}")
        raise SystemExit(1)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
