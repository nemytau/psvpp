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
        steps = list(range(1, len(summary["rl"]["cost_history"]) + 1))
        plt.figure(figsize=(8, 5))
        plt.plot(steps, summary["rl"]["cost_history"], label="RL current cost", color="tab:blue")
        if summary.get("rl", {}).get("best_cost_history"):
            plt.plot(steps, summary["rl"]["best_cost_history"], label="RL best cost", color="tab:blue", linestyle="--")
        if summary.get("baseline", {}).get("cost_history"):
            plt.plot(steps, summary["baseline"]["cost_history"], label="Baseline current cost", color="tab:orange")
        if summary.get("baseline", {}).get("best_cost_history"):
            plt.plot(steps, summary["baseline"]["best_cost_history"], label="Baseline best cost", color="tab:orange", linestyle="--")
        plt.xlabel("Iteration")
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
    )

    baseline_stats: Optional[Dict[str, Any]] = None
    if not args.skip_baseline:
        baseline_stats = run_episode_with_policy(
            policy_model=None,
            problem_path=dataset_rel,
            seed=args.seed,
            max_iterations=args.max_iterations,
            deterministic=deterministic,
        )

    dataset_label = Path(dataset_rel).name
    print(f"Dataset: {dataset_rel} (seed {args.seed})")
    print(f"RL best cost: {rl_stats['best_cost']:.4f}")
    print(f"RL final cost: {rl_stats['final_cost']:.4f}")

    if baseline_stats:
        delta = baseline_stats["best_cost"] - rl_stats["best_cost"]
        print(f"Baseline best cost: {baseline_stats['best_cost']:.4f}")
        print(f"Best-cost delta (baseline - RL): {delta:+.4f}")

    output_dir = _ensure_output_dir(args.output_dir)
    summary: Dict[str, Any] = {
        "dataset": dataset_rel,
        "seed": args.seed,
        "deterministic": deterministic,
        "rl": rl_stats,
    }
    if baseline_stats:
        summary["baseline"] = baseline_stats
        summary["best_cost_delta"] = baseline_stats["best_cost"] - rl_stats["best_cost"]

    _write_summary(output_dir, summary, dataset_label)

    if not args.skip_baseline:
        compare_model_against_baseline(
            model=model,
            problem_paths=[dataset_rel],
            seeds=[args.seed],
            max_iterations=args.max_iterations,
            output_dir=str(output_dir / "model_vs_baseline"),
            deterministic=deterministic,
        )

if __name__ == "__main__":
    main()
