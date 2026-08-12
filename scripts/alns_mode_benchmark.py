#!/usr/bin/env python3
"""Benchmark ALNS algorithm modes over generated datasets."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from rl.dataset_manager import GeneratedDatasetManager
from rl.rl_alns_environment import ALNSEnvironment


DEFAULT_SIZES = ["small", "medium"]
DEFAULT_MODES = ["baseline", "kisialiou", "reinforcement_learning"]
DEFAULT_ITERATIONS = 1000
DEFAULT_DATASETS_PER_SIZE = 1
RESULTS_PATH = REPO_ROOT / "output" / "alns_mode_benchmark.json"
LOG_PATH = REPO_ROOT / "output" / "alns_mode_benchmark.log"


LOG_LINES: List[str] = []


def log(message: str) -> None:
    LOG_LINES.append(message)
    print(message)


def normalize_algorithm_mode(value: Any) -> str:
    mode = str(value or "baseline").strip().lower()
    return "reinforcement_learning" if mode == "rl" else mode


def get_iterations_for_mode(base_iterations: int, mode: str) -> int:
    """Return iteration count for mode. All modes use the same iteration count."""
    return base_iterations


def run_episode(
    *,
    problem_path: str,
    seed: int,
    max_iterations: int,
    algorithm_mode: str,
    kisialiou_improvement_sequence: Optional[List[str]] = None,
    operator_logging_dir: Optional[str] = None,
    mode_label: Optional[str] = None,
    size_label: Optional[str] = None,
    fixed_pair_idx: Optional[int] = None,
) -> Dict[str, Any]:
    # Extract dataset filename for operator logging
    dataset_name = Path(problem_path).stem if problem_path else None
    
    env_kwargs: Dict[str, Any] = {
        "problem_instance": problem_path,
        "max_iterations": max_iterations,
        "seed": seed,
        "problem_instance_paths": [problem_path],
        "problem_sampling_strategy": "round_robin",
        "enable_operator_logging": True,
        "operator_logging_format": "csv",
        "operator_logging_mode": algorithm_mode,
        "operator_logging_future_window": 5,
        "operator_logging_mode_label": mode_label or algorithm_mode,
        "operator_logging_size": size_label,
        "operator_logging_dataset_name": dataset_name,
        "algorithm_mode": algorithm_mode,
        "force_baseline_improvement": False,
        "baseline_improvement_idx": None,
        "kisialiou_improvement_sequence": kisialiou_improvement_sequence,
        # Kisialiou et al. (2018) Table 1: destroy-operator removal-fraction
        # calibration depends on instance size (<15 installations vs >=15).
        # Our "small" size (10 installations) is the only one below that
        # threshold; medium (20) and large (40) both use the "large" bucket.
        "size_bucket": "small" if size_label == "small" else "large",
    }
    if fixed_pair_idx is not None:
        # Force one specific destroy/repair pair for every iteration of the run
        # (no adaptive weighting, no random pair switching) -- Exp 3, testing
        # whether a single well-chosen pair matches or beats adaptive selection.
        env_kwargs["action_module"] = "op_pair_v1"
    elif algorithm_mode in ("baseline", "kisialiou"):
        # Both modes must use the adaptive weighted engine for destroy/repair
        # selection (see rl/modules/actions.py:AdaptiveWeightedActionSpace).
        # Without this, ALNSEnvironment falls back to its default action module
        # (op_pair_v1), which is a *random*-pair action space intended for RL
        # comparisons, not a real baseline -- see op_pair_v1's own docstring.
        env_kwargs["action_module"] = "adaptive_weighted_v1"
    if operator_logging_dir:
        env_kwargs["operator_logging_dir"] = operator_logging_dir

    env = ALNSEnvironment(**env_kwargs)

    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    steps = 0
    done = False
    truncated = False

    if fixed_pair_idx is not None:
        num_pairs = int(env.action_impl.num_pairs())
        if not (0 <= fixed_pair_idx < num_pairs):
            raise ValueError(f"fixed_pair_idx {fixed_pair_idx} out of range [0, {num_pairs})")

    while not done and not truncated and steps < max_iterations:
        if fixed_pair_idx is not None:
            action = fixed_pair_idx
        else:
            action = env.action_space.sample()
            if hasattr(env, "action_impl") and hasattr(env.action_impl, "num_pairs"):
                try:
                    num_pairs = int(env.action_impl.num_pairs())
                except Exception:
                    num_pairs = -1
                if num_pairs and num_pairs > 0:
                    action = int(rng.integers(0, num_pairs))
        obs, reward, done, truncated, info = env.step(action)
        steps += 1

    stats = env.get_episode_statistics()
    env.close()

    stats.setdefault("iterations", steps)
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark a single ALNS algorithm mode over generated datasets")
    parser.add_argument(
        "--sizes",
        nargs="*",
        default=DEFAULT_SIZES,
        help=f"Dataset sizes to include (default: {', '.join(DEFAULT_SIZES)})",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="baseline",
        help="Algorithm mode to evaluate (default: baseline). One mode per invocation.",
    )
    parser.add_argument(
        "--dataset-paths",
        type=str,
        default=None,
        help="Comma-separated dataset paths to run (overrides --sizes selection)",
    )
    parser.add_argument(
        "--size-label",
        type=str,
        default=None,
        help="Optional size label when using --dataset-paths",
    )
    parser.add_argument(
        "--mode-label",
        type=str,
        default=None,
        help="Optional mode label to persist in output rows (defaults to --mode)",
    )
    parser.add_argument(
        "--kisialiou-improvement-sequence",
        type=str,
        default=None,
        help="Comma-separated ordered improvement operator names for kisialiou override",
    )
    parser.add_argument(
        "--fixed-pair-idx",
        type=int,
        default=None,
        help=(
            "Force one specific destroy/repair pair (0-8, op_pair_v1 indexing) for "
            "every iteration instead of adaptive/random selection (Exp 3)."
        ),
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_ITERATIONS,
        help="ALNS iterations per dataset (default: %(default)s)",
    )
    parser.add_argument(
        "--datasets-per-size",
        type=int,
        default=DEFAULT_DATASETS_PER_SIZE,
        help="Number of test datasets per size (default: %(default)s)",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=4242,
        help="Base seed offset applied per dataset index (default: %(default)s)",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=RESULTS_PATH,
        help="Destination JSON file for per-run results",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=LOG_PATH,
        help="Destination log file capturing benchmark progress",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory for operator usage logs (default: logs/)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    LOG_LINES.clear()
    results: List[Dict[str, Any]] = []

    mode: str = normalize_algorithm_mode(args.mode)
    mode_label: str = str(args.mode_label or mode)
    kisialiou_improvement_sequence: Optional[List[str]] = None
    if args.kisialiou_improvement_sequence is not None:
        kisialiou_improvement_sequence = [
            item.strip()
            for item in str(args.kisialiou_improvement_sequence).split(",")
            if item.strip()
        ]

    size_label_override = args.size_label
    dataset_entries: List[Tuple[str, Path]] = []

    manager = None
    sizes: List[str] = []
    if args.dataset_paths:
        dataset_paths = [Path(item.strip()) for item in args.dataset_paths.split(",") if item.strip()]
        if not dataset_paths:
            raise RuntimeError("--dataset-paths provided but no valid paths were parsed")
        size_label = str(size_label_override or "unknown")
        dataset_entries = [(size_label, path) for path in dataset_paths]
        sizes = [size_label]
    else:
        manager = GeneratedDatasetManager()
        sizes = list(args.sizes or [])

    if not sizes:
        log("No dataset sizes specified; exiting")
        return

    datasets_per_size = max(1, int(args.datasets_per_size))

    if not dataset_entries:
        for size in sizes:
            log(f"Preparing datasets for size={size}")
            splits = manager.prepare_size(size) if manager is not None else {}
            test_paths = splits.get("test", [])[:datasets_per_size]
            if len(test_paths) < datasets_per_size:
                raise RuntimeError(
                    f"Expected at least {datasets_per_size} test dataset(s) for size '{size}', found {len(test_paths)}"
                )
            dataset_entries.extend((size, Path(path)) for path in test_paths)

    for idx, (size, dataset_path) in enumerate(dataset_entries):
        dataset_path = Path(dataset_path)
        dataset_label = dataset_path.name
        seed = int(args.base_seed) + idx

        mode_iterations = get_iterations_for_mode(args.iterations, mode)
        log(
            f"Running mode={mode_label:<24} (algorithm_mode={mode}) size={size:<6} dataset={dataset_label:<20} seed={seed}"
        )
        start_time = time.perf_counter()
        episode = run_episode(
            problem_path=str(dataset_path),
            seed=seed,
            max_iterations=mode_iterations,
            algorithm_mode=mode,
            kisialiou_improvement_sequence=kisialiou_improvement_sequence,
            operator_logging_dir=args.log_dir,
            mode_label=mode_label,
            size_label=size,
            fixed_pair_idx=args.fixed_pair_idx,
        )
        wall_time = time.perf_counter() - start_time

        elapsed_sequence = episode.get("elapsed_seconds") or []
        runtime_seconds = float(elapsed_sequence[-1]) if elapsed_sequence else wall_time

        initial_cost = float(episode.get("initial_cost") or 0.0)
        best_cost = float(episode.get("best_cost") or 0.0)
        final_cost = float(episode.get("final_cost") or 0.0)
        iterations = int(episode.get("iterations") or episode.get("total_iterations") or 0)

        improvement_abs = max(0.0, initial_cost - best_cost) if initial_cost else 0.0
        improvement_pct = (
            (improvement_abs / initial_cost) * 100.0 if initial_cost and initial_cost > 1e-9 else 0.0
        )

        results.append(
            {
                "size": size,
                "dataset": dataset_label,
                "dataset_path": str(dataset_path),
                "mode": mode_label,
                "algorithm_mode": mode,
                "seed": seed,
                "iterations": iterations,
                "initial_cost": initial_cost,
                "best_cost": best_cost,
                "final_cost": final_cost,
                "improvement_abs": improvement_abs,
                "improvement_pct": improvement_pct,
                "runtime_seconds": runtime_seconds,
                "kisialiou_improvement_sequence": kisialiou_improvement_sequence,
            }
        )

    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    with args.results_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    log(f"Wrote {len(results)} runs to {args.results_path}")

    summary = build_summary(results)
    print_summary(summary, sizes, [mode_label])

    with args.log_path.open("w", encoding="utf-8") as fh:
        for line in LOG_LINES:
            fh.write(f"{line}\n")
    log(f"Log written to {args.log_path}")


def build_summary(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    summary: Dict[str, Dict[str, Dict[str, List[float]]]] = {}

    for entry in results:
        size = str(entry["size"])
        mode = str(entry["mode"])
        bucket = summary.setdefault(size, {}).setdefault(
            mode,
            {"best_cost": [], "impr_pct": [], "runtime": []},
        )
        bucket["best_cost"].append(float(entry["best_cost"]))
        bucket["impr_pct"].append(float(entry["improvement_pct"]))
        bucket["runtime"].append(float(entry["runtime_seconds"]))

    aggregated: Dict[str, Dict[str, Dict[str, float]]] = {}
    for size, modes in summary.items():
        aggregated[size] = {}
        for mode, values in modes.items():
            best_cost_values = values["best_cost"]
            improvement_values = values["impr_pct"]
            runtime_values = values["runtime"]

            aggregated[size][mode] = {
                "runs": float(len(best_cost_values)),
                "mean_best_cost": statistics.mean(best_cost_values) if best_cost_values else float("nan"),
                "stdev_best_cost": statistics.pstdev(best_cost_values) if len(best_cost_values) > 1 else 0.0,
                "mean_improvement_pct": statistics.mean(improvement_values) if improvement_values else 0.0,
                "mean_runtime_seconds": statistics.mean(runtime_values) if runtime_values else float("nan"),
            }
    return aggregated


def print_summary(
    summary: Dict[str, Dict[str, Dict[str, float]]],
    sizes: Iterable[str],
    modes: Iterable[str],
) -> None:
    header = f"{'Size':<8} {'Mode':<24} {'Runs':>4} {'Best Cost':>12} {'Δ%':>7} {'Runtime(s)':>11}"
    separator = "=" * len(header)
    sizes = list(sizes)
    modes = list(modes)

    log(separator)
    log(header)
    log(separator)

    normalized_modes = [normalize_algorithm_mode(mode) for mode in modes]

    for size in sizes:
        summary_modes = summary.get(size, {})
        for mode in normalized_modes:
            stats = summary_modes.get(mode)
            if not stats:
                continue
            log(
                f"{size:<8} {mode:<24} {int(stats['runs']):>4} "
                f"{stats['mean_best_cost']:>12.2f} {stats['mean_improvement_pct']:>7.2f} "
                f"{stats['mean_runtime_seconds']:>11.1f}"
            )
    log(separator)

    for size in sizes:
        summary_modes = summary.get(size, {})
        if not summary_modes:
            continue
        best_mode, best_stats = min(
            summary_modes.items(),
            key=lambda item: item[1].get("mean_best_cost", float("inf")),
        )
        log(f"Best mode for {size}: {best_mode} (mean best cost {best_stats['mean_best_cost']:.2f})")


if __name__ == "__main__":
    main()
