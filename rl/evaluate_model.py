"""Evaluate a trained PPO model on a single ALNS dataset instance."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import PPO

from rl.train_alns_rl import (
    compare_model_against_baseline,
    run_episode_with_policy,
    _to_relative_path,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PPO ALNS model on a dataset")
    parser.add_argument("dataset", help="Path to processed dataset directory")
    parser.add_argument(
        "--model-path",
        default="models/ppo_alns_model.zip",
        help="Path to trained PPO model",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=100,
        help="Maximum ALNS iterations per episode",
    )
    parser.add_argument("--seed", type=int, default=42, help="Episode seed")
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic actions (default deterministic)",
    )
    parser.add_argument(
        "--output-dir",
        default="logs/ppo_alns/single_eval",
        help="Directory for summary artifacts",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip random baseline comparison",
    )
    parser.add_argument(
        "--action-module",
        default=None,
        help="Registry key for the action space implementation",
    )
    parser.add_argument(
        "--state-module",
        default=None,
        help="Registry key for the state encoder implementation",
    )
    parser.add_argument(
        "--reward-module",
        default=None,
        help="Registry key for the reward function implementation",
    )
    return parser.parse_args()


def _prepare_dataset_path(path_str: str) -> str:
    path_obj = Path(path_str)
    if path_obj.is_absolute():
        return _to_relative_path(path_obj)
    return str(path_obj)


def _ensure_output_dir(path: str) -> Path:
    out_dir = PROJECT_ROOT / path if not Path(path).is_absolute() else Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _write_summary(
    output_dir: Path,
    summary: Dict[str, Any],
    dataset_slug: str,
) -> None:
    json_path = output_dir / f"summary_{dataset_slug}.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    if summary.get("rl", {}).get("cost_history"):
        rl_cost = summary["rl"].get("cost_history", [])
        rl_best = summary["rl"].get("best_cost_history", [])
        baseline_cost = summary.get("baseline", {}).get("cost_history", [])
        baseline_best = summary.get("baseline", {}).get("best_cost_history", [])

        rl_time = summary["rl"].get("elapsed_seconds", [])
        baseline_time = summary.get("baseline", {}).get("elapsed_seconds", [])

        if not rl_time:
            rl_time = [float(i) for i in range(len(rl_cost))]
        if not baseline_time:
            baseline_time = [float(i) for i in range(len(baseline_cost))]

        def _prepend_zero(time_vals, values):
            if not time_vals or not values:
                return time_vals, values
            if time_vals[0] <= 0.0:
                return time_vals, values
            return [0.0] + time_vals, [values[0]] + values

        rl_time_plot, rl_cost_plot = _prepend_zero(rl_time, rl_cost)
        _, rl_best_plot = _prepend_zero(rl_time, rl_best)
        baseline_time_plot, baseline_cost_plot = _prepend_zero(baseline_time, baseline_cost)
        _, baseline_best_plot = _prepend_zero(baseline_time, baseline_best)

        plt.figure(figsize=(8, 5))
        plt.plot(rl_time_plot, rl_cost_plot, label="RL current cost", color="tab:blue")
        if rl_best_plot:
            plt.plot(rl_time_plot, rl_best_plot, label="RL best cost", color="tab:blue", linestyle="--")
        if baseline_cost_plot:
            plt.plot(baseline_time_plot, baseline_cost_plot, label="Baseline current cost", color="tab:orange")
        if baseline_best_plot:
            plt.plot(
                baseline_time_plot,
                baseline_best_plot,
                label="Baseline best cost",
                color="tab:orange",
                linestyle="--",
            )
        plt.xlabel("Elapsed time (s)")
        plt.ylabel("Solution cost")
        plt.title(f"RL vs baseline convergence ({dataset_slug})")
        plt.legend()
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        plt.tight_layout()
        plot_path = output_dir / f"convergence_{dataset_slug}.png"
        plt.savefig(plot_path)
        plt.close()


def main() -> None:
    args = _parse_args()
    dataset_rel = _prepare_dataset_path(args.dataset)
    model = PPO.load(args.model_path)

    deterministic = not args.stochastic
    rl_stats = run_episode_with_policy(
        policy_model=model,
        problem_path=dataset_rel,
        seed=args.seed,
        max_iterations=args.max_iterations,
        deterministic=deterministic,
        capture_snapshot=True,
        action_module=args.action_module,
        state_module=args.state_module,
        reward_module=args.reward_module,
    )
    shared_snapshot = rl_stats.pop("initial_snapshot", None)

    baseline_stats: Optional[Dict[str, Any]] = None
    if not args.skip_baseline:
        baseline_stats = run_episode_with_policy(
            policy_model=None,
            problem_path=dataset_rel,
            seed=args.seed,
            max_iterations=args.max_iterations,
            deterministic=deterministic,
            shared_snapshot=shared_snapshot,
            action_module=rl_stats.get("modules", {}).get("action", args.action_module),
            state_module=rl_stats.get("modules", {}).get("state", args.state_module),
            reward_module=rl_stats.get("modules", {}).get("reward", args.reward_module),
        )

    dataset_label = Path(dataset_rel).name
    rl_best_cost = rl_stats.get("best_cost")
    rl_final_cost = rl_stats.get("final_cost")

    print(f"Dataset: {dataset_rel} (seed {args.seed})")
    if rl_best_cost is not None:
        print(f"RL best cost: {rl_best_cost:.4f}")
    else:
        print("Warning: RL best cost missing from statistics")
    if rl_final_cost is not None:
        print(f"RL final cost: {rl_final_cost:.4f}")

    baseline_best_cost: Optional[float] = None
    if baseline_stats:
        baseline_best_cost = baseline_stats.get("best_cost")
        if baseline_best_cost is not None:
            delta = baseline_best_cost - (rl_best_cost or baseline_best_cost)
            print(f"Baseline best cost: {baseline_best_cost:.4f}")
            print(f"Best-cost delta (baseline - RL): {delta:+.4f}")
        else:
            print("Warning: Baseline best cost missing from statistics")

    output_dir = _ensure_output_dir(args.output_dir)
    summary: Dict[str, Any] = {
        "dataset": dataset_rel,
        "seed": args.seed,
        "deterministic": deterministic,
        "rl": rl_stats,
    }
    if baseline_stats:
        summary["baseline"] = baseline_stats
        if rl_best_cost is not None and baseline_best_cost is not None:
            summary["best_cost_delta"] = baseline_best_cost - rl_best_cost

    _write_summary(output_dir, summary, dataset_label)

    if not args.skip_baseline:
        compare_model_against_baseline(
            model=model,
            problem_paths=[dataset_rel],
            seeds=[args.seed],
            max_iterations=args.max_iterations,
            output_dir=str(output_dir / "model_vs_baseline"),
            deterministic=deterministic,
            action_module=args.action_module,
            state_module=args.state_module,
            reward_module=args.reward_module,
        )

if __name__ == "__main__":
    main()
