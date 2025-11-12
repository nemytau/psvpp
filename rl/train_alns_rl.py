"""
Training script for ALNS reinforcement learning using Stable-Baselines3.

This script trains a PPO agent on a pool of ALNS instances, evaluates the
resulting policy on a held-out test split, and benchmarks the learned policy
against a random baseline using shared problem samples.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.vec_env import DummyVecEnv
    print("[OK] Stable-Baselines3 imported successfully")
except ImportError as e:  # pragma: no cover
    print(f"[ERROR] Stable-Baselines3 not found: {e}")
    print("Please install with: pip install stable-baselines3[extra]")
    raise

try:
    import gymnasium as gym  # noqa: F401
    print("[OK] Gymnasium imported successfully")
except ImportError as e:  # pragma: no cover
    print(f"[ERROR] Gymnasium not found: {e}")
    print("Please install with: pip install gymnasium")
    raise

try:
    from rl.rl_alns_environment import ALNSEnvironment
    print("[OK] ALNSEnvironment imported successfully")
except ImportError as e:  # pragma: no cover
    print(f"[ERROR] ALNSEnvironment import failed: {e}")
    raise

from rl.dataset_manager import GeneratedDatasetManager


def _slugify_path(path: str) -> str:
    return Path(path).name.replace(" ", "_")


def _format_dataset_label(path: str) -> str:
    return Path(path).name


class ALNSTrainingCallback(BaseCallback):
    """Callback that captures cost convergence during PPO training."""

    def __init__(self, log_dir: str, verbose: int = 1):
        super().__init__(verbose)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.episode_rewards: List[float] = []
        self.episode_improvements: List[float] = []
        self.episode_lengths: List[int] = []
        self.operator_usage: Dict[str, int] = {}
        self.best_solution_cost: float = float("inf")
        self.training_steps: List[int] = []
        self.training_current_costs: List[float] = []
        self.training_best_costs: List[float] = []
        self.step_counter = 0

    def _on_step(self) -> bool:
        if len(self.locals.get("infos", [])) > 0:
            info = self.locals["infos"][0]

            if "operators_used" in info:
                destroy_idx, repair_idx = info["operators_used"]
                key = f"D{destroy_idx}_R{repair_idx}"
                self.operator_usage[key] = self.operator_usage.get(key, 0) + 1

            step_info = info.get("step_info", {})
            if step_info:
                self.step_counter += 1
                current_cost = float(step_info.get("current_cost", 0.0))
                best_cost = float(step_info.get("best_cost", current_cost))
                self.training_steps.append(self.step_counter)
                self.training_current_costs.append(current_cost)
                self.training_best_costs.append(best_cost)
        return True

    def _on_rollout_end(self) -> None:
        pass

    def _on_training_end(self) -> None:
        summary = {
            "total_episodes": len(self.episode_improvements),
            "avg_improvement": np.mean(self.episode_improvements) if self.episode_improvements else 0.0,
            "best_improvement": max(self.episode_improvements) if self.episode_improvements else 0.0,
            "best_solution_cost": self.best_solution_cost,
            "operator_usage": self.operator_usage,
        }

        summary_file = self.log_dir / "training_summary.txt"
        with summary_file.open("w") as f:
            f.write("ALNS RL Training Summary\n")
            f.write("=" * 30 + "\n")
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")

        if self.verbose > 0:
            print(f"[INFO] Training summary saved to: {summary_file}")

        self._export_training_metrics()

    def _export_training_metrics(self) -> None:
        if not self.training_steps:
            return

        data_file = self.log_dir / "training_costs.csv"
        with data_file.open("w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["step", "current_cost", "best_cost"])
            for step, current_cost, best_cost in zip(
                self.training_steps,
                self.training_current_costs,
                self.training_best_costs,
            ):
                writer.writerow([step, current_cost, best_cost])

        plot_path = self.log_dir / "training_convergence.png"
        plt.figure(figsize=(10, 6))
        plt.plot(self.training_steps, self.training_current_costs, label="Current cost", alpha=0.7)
        plt.plot(self.training_steps, self.training_best_costs, label="Best cost", alpha=0.9)
        plt.xlabel("Training step")
        plt.ylabel("Solution cost")
        plt.title("Training cost convergence")
        plt.legend()
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()


def test_environment_manually(problem_instance: Optional[str] = "SMALL_1") -> bool:
    """Quick smoke test to ensure the ALNS environment is functional."""

    dataset_label = problem_instance or "SMALL_1"
    print("\n[TEST] Testing ALNSEnvironment manually...")
    print(f"   Using problem instance: {dataset_label}")
    path_obj = Path(dataset_label)
    absolute_path = path_obj if path_obj.is_absolute() else PROJECT_ROOT / path_obj
    print(f"   Path exists: {absolute_path.exists()} (dir={absolute_path.is_dir()})")

    try:
        env = ALNSEnvironment(
            problem_instance=dataset_label,
            max_iterations=5,
            seed=42,
            problem_instance_paths=[dataset_label],
            problem_sampling_strategy="round_robin",
        )

        print(f"Action space: {env.action_space}")
        print(f"Observation space: {env.observation_space}")

        print("Checking environment with Stable-Baselines3...")
        check_env(env, warn=True)
        print("[OK] Environment check passed!")

        obs, info = env.reset()
        print(f"Initial observation shape: {obs.shape}")
        print(f"Initial cost: {info.get('initial_cost', 'N/A')}")

        total_reward = 0.0
        for step in range(5):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward

            step_info = info.get("step_info", {})
            print(
                f"Step {step + 1}: reward={reward:.2f}, "
                f"cost={step_info.get('current_cost', 'N/A'):.2f}, "
                f"done={done}"
            )

            if done or truncated:
                break

        print(f"Total reward: {total_reward:.2f}")
        print("[OK] Manual test completed successfully!")

        env.close()
        return True

    except Exception as exc:  # pragma: no cover
        print(f"[ERROR] Manual test failed: {exc}")
        import traceback

        traceback.print_exc()
        return False


def train_ppo_agent(
    total_timesteps: int = 50000,
    learning_rate: float = 3e-4,
    log_dir: str = "logs/ppo_alns",
    model_save_path: str = "models/ppo_alns_model",
    seed: int = 42,
    train_instance_paths: Optional[Sequence[str]] = None,
    max_iterations: int = 100,
    sampling_strategy: str = "random",
    action_module: Optional[str] = None,
    state_module: Optional[str] = None,
    reward_module: Optional[str] = None,
) -> Tuple[PPO, DummyVecEnv]:
    """Train a PPO agent on the ALNS environment using provided datasets."""

    print(f"\n[TRAIN] Training PPO agent for {total_timesteps} timesteps...")

    train_instances = list(train_instance_paths) if train_instance_paths else ["SMALL_1"]
    if not train_instances:
        raise ValueError("train_instance_paths must provide at least one dataset")
    print(
        f"   Training on {len(train_instances)} instances "
        f"(sampling={sampling_strategy})"
    )

    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)

    model_dir = Path(model_save_path).parent
    model_dir.mkdir(parents=True, exist_ok=True)

    def make_env() -> ALNSEnvironment:
        env_kwargs: Dict[str, Any] = {
            "problem_instance": train_instances[0],
            "max_iterations": max_iterations,
            "seed": seed,
            "problem_instance_paths": train_instances,
            "problem_sampling_strategy": sampling_strategy,
        }
        if action_module:
            env_kwargs["action_module"] = action_module
        if state_module:
            env_kwargs["state_module"] = state_module
        if reward_module:
            env_kwargs["reward_module"] = reward_module

        return ALNSEnvironment(**env_kwargs)

    vec_env = DummyVecEnv([make_env])

    logger = configure(str(log_dir_path), ["stdout", "csv", "tensorboard"])

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=learning_rate,
        verbose=1,
        tensorboard_log=str(log_dir_path),
        seed=seed,
    )

    callback = ALNSTrainingCallback(log_dir=str(log_dir_path))
    model.learn(total_timesteps=total_timesteps, callback=callback)

    model.save(model_save_path)
    print(f"[SAVED] Model saved to: {model_save_path}")

    return model, vec_env


def evaluate_trained_model(
    model_path: str,
    problem_paths: Sequence[str],
    n_eval_episodes: Optional[int] = None,
    deterministic: bool = True,
    output_dir: Optional[str] = None,
    max_iterations: int = 100,
) -> Tuple[float, float, List[Dict[str, Any]]]:
    """Evaluate a trained PPO model across a set of problem instances."""

    eval_paths = list(problem_paths) if problem_paths else ["SMALL_1"]
    episodes = n_eval_episodes or len(eval_paths)

    print(f"\n[METRICS] Evaluating trained model across {episodes} episodes...")
    model = PPO.load(model_path)
    print(f"[OK] Model loaded from: {model_path}")

    eval_env = ALNSEnvironment(
        problem_instance=eval_paths[0],
        max_iterations=max_iterations,
        seed=123,
        problem_instance_paths=eval_paths,
        problem_sampling_strategy="round_robin",
    )

    mean_reward_raw, std_reward_raw = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=episodes,
        deterministic=deterministic,
        return_episode_rewards=False,
    )
    mean_reward = cast(float, mean_reward_raw)
    std_reward = cast(float, std_reward_raw)
    mean_reward = float(mean_reward)
    std_reward = float(std_reward)
    eval_env.close()

    details_env = ALNSEnvironment(
        problem_instance=eval_paths[0],
        max_iterations=max_iterations,
        seed=999,
        problem_instance_paths=eval_paths,
        problem_sampling_strategy="round_robin",
    )

    plots_dir: Optional[Path] = None
    if output_dir:
        plots_dir = Path(output_dir)
        plots_dir.mkdir(parents=True, exist_ok=True)

    detailed_stats: List[Dict[str, Any]] = []
    for idx, dataset_path in enumerate(eval_paths):
        obs, _ = details_env.reset(seed=1000 + idx, options={"problem_path": dataset_path})
        episode_reward = 0.0

        for _ in range(max_iterations):
            action_array, _ = model.predict(obs, deterministic=deterministic)
            action = int(action_array.item()) if hasattr(action_array, "item") else int(action_array)
            obs, reward, done, truncated, info = details_env.step(action)
            episode_reward += reward
            if done or truncated:
                break

        stats = details_env.get_episode_statistics()
        record = {
            "dataset_path": dataset_path,
            "dataset": _format_dataset_label(dataset_path),
            "reward": episode_reward,
            "final_cost": stats.get("final_cost"),
            "best_cost": stats.get("best_cost"),
            "iterations": stats.get("total_iterations"),
            "improvement_abs": stats.get("total_improvement", 0.0),
            "improvement_pct": stats.get("total_improvement_pct", 0.0),
        }
        detailed_stats.append(record)

        cost_history = stats.get("cost_history", [])
        best_history = stats.get("best_cost_history", [])
        if plots_dir and cost_history:
            steps = list(range(1, len(cost_history) + 1))
            dataset_slug = _slugify_path(dataset_path)
            plt.figure(figsize=(8, 5))
            plt.plot(steps, cost_history, label="Current cost", alpha=0.7)
            if best_history:
                plt.plot(steps, best_history, label="Best cost", alpha=0.9)
            plt.xlabel("Iteration")
            plt.ylabel("Solution cost")
            plt.title(f"Evaluation convergence - {record['dataset']}")
            plt.legend()
            plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
            plt.tight_layout()
            plot_file = plots_dir / f"evaluation_convergence_{dataset_slug}.png"
            plt.savefig(plot_file)
            plt.close()

    details_env.close()

    print(f"[RESULT] Evaluation: mean {mean_reward:.2f} +/- {std_reward:.2f}")
    return mean_reward, std_reward, detailed_stats


def compare_with_baseline(
    problem_paths: Sequence[str],
    max_iterations: int = 100,
    output_dir: Optional[str] = None,
) -> Tuple[float, float, List[Dict[str, Any]]]:
    """Evaluate a random baseline policy across the same problem instances."""

    baseline_paths = list(problem_paths) if problem_paths else ["SMALL_1"]
    env = ALNSEnvironment(
        problem_instance=baseline_paths[0],
        max_iterations=max_iterations,
        seed=2048,
        problem_instance_paths=baseline_paths,
        problem_sampling_strategy="round_robin",
    )

    plots_dir: Optional[Path] = None
    if output_dir:
        plots_dir = Path(output_dir)
        plots_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    rewards: List[float] = []
    improvements: List[float] = []

    for idx, dataset_path in enumerate(baseline_paths):
        obs, _ = env.reset(seed=3000 + idx, options={"problem_path": dataset_path})
        episode_reward = 0.0

        for _ in range(max_iterations):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            if done or truncated:
                break

        stats = env.get_episode_statistics()
        record = {
            "dataset_path": dataset_path,
            "dataset": _format_dataset_label(dataset_path),
            "reward": episode_reward,
            "final_cost": stats.get("final_cost"),
            "best_cost": stats.get("best_cost"),
            "iterations": stats.get("total_iterations"),
            "improvement_abs": stats.get("total_improvement", 0.0),
            "improvement_pct": stats.get("total_improvement_pct", 0.0),
        }
        results.append(record)
        rewards.append(episode_reward)
        improvements.append(float(stats.get("total_improvement", 0.0)))

        cost_history = stats.get("cost_history", [])
        best_history = stats.get("best_cost_history", [])
        if plots_dir and cost_history:
            steps = list(range(1, len(cost_history) + 1))
            dataset_slug = _slugify_path(dataset_path)
            plt.figure(figsize=(8, 5))
            plt.plot(steps, cost_history, label="Current cost", alpha=0.7)
            if best_history:
                plt.plot(steps, best_history, label="Best cost", alpha=0.9)
            plt.xlabel("Iteration")
            plt.ylabel("Solution cost")
            plt.title(f"Random baseline convergence - {record['dataset']}")
            plt.legend()
            plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
            plt.tight_layout()
            plot_file = plots_dir / f"baseline_convergence_{dataset_slug}.png"
            plt.savefig(plot_file)
            plt.close()

    env.close()

    mean_reward = float(np.mean(rewards)) if rewards else 0.0
    mean_improvement = float(np.mean(improvements)) if improvements else 0.0

    print(
        f"[BASELINE] Random policy: mean reward {mean_reward:.2f}, "
        f"mean improvement {mean_improvement:.2f}"
    )
    return mean_reward, mean_improvement, results


def _pad_history(history: Sequence[float], target_len: int) -> List[float]:
    history_list = list(history)
    if len(history_list) >= target_len:
        return history_list[:target_len]
    if not history_list:
        return [0.0] * target_len
    last_value = history_list[-1]
    return history_list + [last_value] * (target_len - len(history_list))


def run_episode_with_policy(
    policy_model: Optional[PPO],
    problem_path: str,
    seed: int,
    max_iterations: int,
    deterministic: bool = True,
) -> Dict[str, Any]:
    env = ALNSEnvironment(
        problem_instance=problem_path,
        max_iterations=max_iterations,
        seed=seed,
        problem_instance_paths=[problem_path],
        problem_sampling_strategy="round_robin",
    )

    obs, _ = env.reset(seed=seed)
    done = False
    truncated = False
    steps = 0

    while not done and not truncated and steps < max_iterations:
        if policy_model is None:
            action = env.action_space.sample()
        else:
            action_array, _ = policy_model.predict(obs, deterministic=deterministic)
            action = int(action_array.item()) if hasattr(action_array, "item") else int(action_array)
        obs, reward, done, truncated, info = env.step(action)
        steps += 1

    stats = env.get_episode_statistics()
    env.close()

    return {
        "problem_path": problem_path,
        "best_cost": stats.get("best_cost"),
        "final_cost": stats.get("final_cost"),
        "cost_history": stats.get("cost_history", []),
        "best_cost_history": stats.get("best_cost_history", []),
        "iterations": stats.get("total_iterations", steps),
    }


def compare_model_against_baseline(
    model: PPO,
    problem_paths: Sequence[str],
    seeds: Sequence[int] = (42,),
    max_iterations: int = 100,
    output_dir: str = "logs/ppo_alns/model_vs_baseline",
    deterministic: bool = True,
) -> List[Dict[str, Any]]:
    """Run side-by-side comparisons for the PPO model and random baseline."""

    comparison_dir = Path(output_dir)
    comparison_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    rl_current_curves: List[List[float]] = []
    rl_best_curves: List[List[float]] = []
    baseline_current_curves: List[List[float]] = []
    baseline_best_curves: List[List[float]] = []
    combined_length = 0
    dataset_matchups: Dict[str, Dict[str, List[float]]] = {}

    for dataset_path in problem_paths:
        dataset_label = _format_dataset_label(dataset_path)
        for seed in seeds:
            rl_result = run_episode_with_policy(
                policy_model=model,
                problem_path=dataset_path,
                seed=seed,
                max_iterations=max_iterations,
                deterministic=deterministic,
            )

            baseline_result = run_episode_with_policy(
                policy_model=None,
                problem_path=dataset_path,
                seed=seed,
                max_iterations=max_iterations,
                deterministic=deterministic,
            )

            max_len = max(
                len(rl_result["cost_history"]),
                len(baseline_result["cost_history"]),
            )
            steps = list(range(1, max_len + 1))
            rl_current_raw = _pad_history(rl_result["cost_history"], max_len)
            rl_best_raw = _pad_history(rl_result["best_cost_history"], max_len)
            baseline_current_raw = _pad_history(baseline_result["cost_history"], max_len)
            baseline_best_raw = _pad_history(baseline_result["best_cost_history"], max_len)

            candidates = [
                value
                for value in rl_best_raw + baseline_best_raw
                if value is not None and value > 0
            ]
            min_target = min(candidates) if candidates else 1.0

            rl_current = [((value / min_target) - 1.0) * 100.0 for value in rl_current_raw]
            rl_best = [((value / min_target) - 1.0) * 100.0 for value in rl_best_raw]
            baseline_current = [((value / min_target) - 1.0) * 100.0 for value in baseline_current_raw]
            baseline_best = [((value / min_target) - 1.0) * 100.0 for value in baseline_best_raw]

            rl_current_curves.append(rl_current)
            rl_best_curves.append(rl_best)
            baseline_current_curves.append(baseline_current)
            baseline_best_curves.append(baseline_best)
            combined_length = max(combined_length, max_len)

            matchup = dataset_matchups.setdefault(dataset_label, {"rl": [], "baseline": []})
            if rl_result["best_cost"] is not None:
                matchup["rl"].append(float(rl_result["best_cost"]))
            if baseline_result["best_cost"] is not None:
                matchup["baseline"].append(float(baseline_result["best_cost"]))

            dataset_slug = _slugify_path(dataset_path)
            plt.figure(figsize=(10, 6))
            plt.plot(steps, rl_current, label="RL current gap", color="tab:blue")
            plt.plot(steps, rl_best, label="RL best gap", color="tab:blue", linestyle="--")
            plt.plot(steps, baseline_current, label="Baseline current gap", color="tab:orange")
            plt.plot(steps, baseline_best, label="Baseline best gap", color="tab:orange", linestyle="--")
            plt.xlabel("Iteration")
            plt.ylabel("Gap to best-known (%)")
            plt.title(f"Model vs Baseline gap ({dataset_label}, seed {seed})")
            plt.legend()
            plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
            plt.tight_layout()
            plot_file = comparison_dir / f"comparison_{dataset_slug}_seed{seed}.png"
            plt.savefig(plot_file)
            plt.close()

            delta = float(baseline_result["best_cost"] - rl_result["best_cost"])
            results.append(
                {
                    "dataset_path": dataset_path,
                    "dataset": dataset_label,
                    "seed": seed,
                    "rl_best_cost": rl_result["best_cost"],
                    "baseline_best_cost": baseline_result["best_cost"],
                    "best_cost_delta": delta,
                    "rl_iterations": rl_result["iterations"],
                    "baseline_iterations": baseline_result["iterations"],
                    "plot": str(plot_file),
                }
            )

            print(
                f"   {dataset_label} (seed {seed}): RL best={rl_result['best_cost']:.2f}, "
                f"baseline best={baseline_result['best_cost']:.2f}, delta={delta:+.2f}"
            )

    if results:
        csv_path = comparison_dir / "comparison_summary.csv"
        with csv_path.open("w", newline="") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=[
                    "dataset_path",
                    "dataset",
                    "seed",
                    "rl_best_cost",
                    "baseline_best_cost",
                    "best_cost_delta",
                    "rl_iterations",
                    "baseline_iterations",
                    "plot",
                ],
            )
            writer.writeheader()
            writer.writerows(results)
        avg_delta = float(np.mean([row["best_cost_delta"] for row in results]))
        print(f"\n[STATS] Avg best-cost delta (baseline - RL): {avg_delta:+.2f}")
        print(f"[STATS] Detailed comparison saved to: {csv_path}")

    if rl_best_curves and combined_length > 0:
        step_axis = list(range(1, combined_length + 1))

        rl_best_mat = np.array([
            _pad_history(curve, combined_length) for curve in rl_best_curves
        ], dtype=float)
        baseline_best_mat = np.array([
            _pad_history(curve, combined_length) for curve in baseline_best_curves
        ], dtype=float)
        rl_current_mat = np.array([
            _pad_history(curve, combined_length) for curve in rl_current_curves
        ], dtype=float)
        baseline_current_mat = np.array([
            _pad_history(curve, combined_length) for curve in baseline_current_curves
        ], dtype=float)

        def _mean_std(mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            return np.mean(mat, axis=0), np.std(mat, axis=0)

        rl_best_mean, rl_best_std = _mean_std(rl_best_mat)
        baseline_best_mean, baseline_best_std = _mean_std(baseline_best_mat)

        plt.figure(figsize=(10, 6))
        plt.plot(step_axis, baseline_best_mean, label="Baseline best gap", color="tab:orange")
        plt.fill_between(
            step_axis,
            baseline_best_mean - baseline_best_std,
            baseline_best_mean + baseline_best_std,
            color="tab:orange",
            alpha=0.2,
        )
        plt.plot(step_axis, rl_best_mean, label="RL best gap", color="tab:blue")
        plt.fill_between(
            step_axis,
            rl_best_mean - rl_best_std,
            rl_best_mean + rl_best_std,
            color="tab:blue",
            alpha=0.2,
        )
        plt.xlabel("Iteration")
        plt.ylabel("Gap to best-known (%)")
        plt.title("Combined best-cost convergence (mean +/- 1 std)")
        plt.legend()
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        combined_plot = comparison_dir / "combined_best_cost_convergence.png"
        plt.tight_layout()
        plt.savefig(combined_plot)
        plt.close()

        rl_current_mean, rl_current_std = _mean_std(rl_current_mat)
        baseline_current_mean, baseline_current_std = _mean_std(baseline_current_mat)

        plt.figure(figsize=(10, 6))
        plt.plot(step_axis, baseline_current_mean, label="Baseline current gap", color="tab:orange")
        plt.fill_between(
            step_axis,
            baseline_current_mean - baseline_current_std,
            baseline_current_mean + baseline_current_std,
            color="tab:orange",
            alpha=0.2,
        )
        plt.plot(step_axis, rl_current_mean, label="RL current gap", color="tab:blue")
        plt.fill_between(
            step_axis,
            rl_current_mean - rl_current_std,
            rl_current_mean + rl_current_std,
            color="tab:blue",
            alpha=0.2,
        )
        plt.xlabel("Iteration")
        plt.ylabel("Gap to best-known (%)")
        plt.title("Combined current-cost convergence (mean +/- 1 std)")
        plt.legend()
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        combined_current_plot = comparison_dir / "combined_current_cost_convergence.png"
        plt.tight_layout()
        plt.savefig(combined_current_plot)
        plt.close()

    if dataset_matchups:
        dataset_names = sorted(dataset_matchups)
        rl_means = [
            float(np.mean(dataset_matchups[name]["rl"])) if dataset_matchups[name]["rl"] else 0.0
            for name in dataset_names
        ]
        baseline_means = [
            float(np.mean(dataset_matchups[name]["baseline"])) if dataset_matchups[name]["baseline"] else 0.0
            for name in dataset_names
        ]

        x_pos = np.arange(len(dataset_names))
        width = 0.35
        plt.figure(figsize=(10, 6))
        plt.bar(x_pos - width / 2, baseline_means, width=width, label="Baseline", color="tab:orange")
        plt.bar(x_pos + width / 2, rl_means, width=width, label="RL", color="tab:blue")
        plt.xticks(x_pos, dataset_names, rotation=30, ha="right")
        plt.ylabel("Mean best cost")
        plt.title("Best-cost comparison per dataset")
        plt.legend()
        plt.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
        plt.tight_layout()
        bar_plot = comparison_dir / "dataset_best_cost_comparison.png"
        plt.savefig(bar_plot)
        plt.close()

    return results


def prepare_dataset_splits(size: str = "small") -> Dict[str, List[str]]:
    manager = GeneratedDatasetManager()
    splits = manager.prepare_size(size)
    return {
        split: [
            _to_relative_path(Path(path))
            for path in paths
        ]
        for split, paths in splits.items()
    }


def _to_relative_path(path_obj: Path) -> str:
    if not path_obj.is_absolute():
        return str(path_obj)
    try:
        return str(path_obj.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path_obj)


def main() -> None:
    print("[PIPELINE] ALNS Reinforcement Learning Training")
    print("=" * 50)

    dataset_splits = prepare_dataset_splits("small")
    train_paths = dataset_splits.get("train", [])
    test_paths = dataset_splits.get("test", [])

    if not train_paths or not test_paths:
        print("[ERROR] No training or test datasets were found. Aborting.")
        return

    print(
        f"Datasets prepared: {len(train_paths)} train / {len(test_paths)} test instances"
    )

    primary_train_instance = train_paths[0]
    if not test_environment_manually(primary_train_instance):
        print("[ERROR] Manual environment test failed, aborting training")
        return

    max_iterations = 100
    log_dir = "logs/ppo_alns"
    model_path = "models/ppo_alns_model"

    model, vec_env = train_ppo_agent(
        total_timesteps=50000,
        learning_rate=3e-4,
        log_dir=log_dir,
        model_save_path=model_path,
        seed=42,
        train_instance_paths=train_paths,
        max_iterations=max_iterations,
        sampling_strategy="random",
    )

    mean_reward, std_reward, eval_details = evaluate_trained_model(
        model_path=model_path,
        problem_paths=test_paths,
        n_eval_episodes=len(test_paths),
        deterministic=True,
        output_dir=log_dir,
        max_iterations=max_iterations,
    )

    baseline_reward, baseline_improvement, baseline_details = compare_with_baseline(
        problem_paths=test_paths,
        max_iterations=max_iterations,
        output_dir=log_dir,
    )

    comparison_results = compare_model_against_baseline(
        model=model,
        problem_paths=test_paths,
        seeds=[42],
        max_iterations=max_iterations,
        output_dir=f"{log_dir}/model_vs_baseline",
        deterministic=True,
    )

    improvement_vs_baseline = mean_reward - baseline_reward
    pct_improvement = (
        (improvement_vs_baseline / abs(baseline_reward) * 100.0)
        if baseline_reward
        else 0.0
    )

    print("\n[SUMMARY] Final results:")
    print(f"   Trained agent: {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"   Random baseline: {baseline_reward:.2f}")
    print(f"   Reward improvement: {improvement_vs_baseline:+.2f} ({pct_improvement:+.1f}%)")

    if comparison_results:
        rl_wins = sum(1 for row in comparison_results if row["best_cost_delta"] > 0)
        ties = sum(1 for row in comparison_results if row["best_cost_delta"] == 0)
        total = len(comparison_results)
        print(
            f"   Best-cost wins: RL {rl_wins}, ties {ties}, baseline {total - rl_wins - ties}"
        )

    vec_env.close()


if __name__ == "__main__":
    main()
