"""
Training script for ALNS reinforcement learning using Stable-Baselines3.

This script trains a PPO agent on a pool of ALNS instances, evaluates the
resulting policy on a held-out test split, and benchmarks the learned policy
against a random baseline using shared problem samples.
"""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import matplotlib.pyplot as plt
import numpy as np


LOGGER = logging.getLogger("psvpp.rl")
LOG_PREFIX = "[RL]"

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
    LOGGER.info("%s Stable-Baselines3 imported successfully", LOG_PREFIX)
except ImportError as e:  # pragma: no cover
    LOGGER.error("%s Stable-Baselines3 not found: %s", LOG_PREFIX, e)
    LOGGER.error("%s Please install with: pip install stable-baselines3[extra]", LOG_PREFIX)
    raise

try:
    import gymnasium as gym  # noqa: F401
    LOGGER.info("%s Gymnasium imported successfully", LOG_PREFIX)
except ImportError as e:  # pragma: no cover
    LOGGER.error("%s Gymnasium not found: %s", LOG_PREFIX, e)
    LOGGER.error("%s Please install with: pip install gymnasium", LOG_PREFIX)
    raise

try:
    from rl.rl_alns_environment import ALNSEnvironment
    LOGGER.info("%s ALNSEnvironment imported successfully", LOG_PREFIX)
except ImportError as e:  # pragma: no cover
    LOGGER.error("%s ALNSEnvironment import failed: %s", LOG_PREFIX, e)
    raise

from rl.dataset_manager import GeneratedDatasetManager
from rl.registries import (
    DEFAULT_ACTION_KEY,
    DEFAULT_REWARD_KEY,
    DEFAULT_STATE_KEY,
    list_registered_states,
)


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
        self.current_episode_reward: float = 0.0

    def _on_step(self) -> bool:
        if len(self.locals.get("infos", [])) > 0:
            info = self.locals["infos"][0]

            if "operators_used" in info:
                operators = info["operators_used"]
                destroy_idx = operators[0] if len(operators) > 0 else None
                repair_idx = operators[1] if len(operators) > 1 else None
                improvement_idx = operators[2] if len(operators) > 2 else None

                key_parts = []
                if destroy_idx is not None:
                    key_parts.append(f"D{destroy_idx}")
                if repair_idx is not None:
                    key_parts.append(f"R{repair_idx}")
                if improvement_idx is not None and improvement_idx >= 0:
                    key_parts.append(f"I{improvement_idx}")

                if key_parts:
                    key = "_".join(key_parts)
                    self.operator_usage[key] = self.operator_usage.get(key, 0) + 1

            step_info = info.get("step_info", {})
            if step_info:
                self.step_counter += 1
                current_cost = float(step_info.get("current_cost", 0.0))
                best_cost = float(step_info.get("best_cost", current_cost))
                self.training_steps.append(self.step_counter)
                self.training_current_costs.append(current_cost)
                self.training_best_costs.append(best_cost)

            rewards = self.locals.get("rewards")
            reward_value = 0.0
            if rewards is not None:
                try:
                    reward_value = float(np.array(rewards).reshape(-1)[0])
                except Exception:  # pragma: no cover - defensive
                    try:
                        reward_value = float(rewards[0])
                    except Exception:
                        reward_value = 0.0
            self.current_episode_reward += reward_value

            dones = self.locals.get("dones")
            done_flag = False
            if dones is not None:
                try:
                    done_flag = bool(np.array(dones).reshape(-1)[0])
                except Exception:
                    done_flag = bool(dones)

            if done_flag:
                summary = info.get("episode_summary", {})
                improvement_value_raw = summary.get("total_improvement")
                if improvement_value_raw is None:
                    improvement_value_raw = summary.get("best_improvement_abs", 0.0)
                try:
                    improvement_value = float(improvement_value_raw)
                except (TypeError, ValueError):  # pragma: no cover - defensive
                    improvement_value = 0.0

                iterations_completed_raw = summary.get("iterations_completed", self.step_counter)
                try:
                    iterations_completed = int(iterations_completed_raw)
                except (TypeError, ValueError):  # pragma: no cover
                    iterations_completed = self.step_counter

                best_cost_value: Optional[float] = None
                best_cost_raw = summary.get("final_best_cost")
                if best_cost_raw is not None:
                    try:
                        best_cost_value = float(best_cost_raw)
                    except (TypeError, ValueError):  # pragma: no cover
                        best_cost_value = None

                if best_cost_value is not None and best_cost_value < self.best_solution_cost:
                    self.best_solution_cost = best_cost_value

                self.episode_rewards.append(self.current_episode_reward)
                self.episode_improvements.append(improvement_value)
                self.episode_lengths.append(iterations_completed)

                best_cost_display = f"{best_cost_value:.3f}" if best_cost_value is not None else "n/a"
                LOGGER.info(
                    "%s Episode %d finished: reward=%.3f best_cost=%s improvement=%.3f iterations=%d",
                    LOG_PREFIX,
                    len(self.episode_rewards),
                    self.current_episode_reward,
                    best_cost_display,
                    improvement_value,
                    iterations_completed,
                )

                self.current_episode_reward = 0.0
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
            LOGGER.info("%s Training summary saved to: %s", LOG_PREFIX, summary_file)

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
    LOGGER.info("%s Testing ALNSEnvironment manually (instance=%s)", LOG_PREFIX, dataset_label)
    path_obj = Path(dataset_label)
    absolute_path = path_obj if path_obj.is_absolute() else PROJECT_ROOT / path_obj
    LOGGER.info(
        "%s Path exists: %s (dir=%s)",
        LOG_PREFIX,
        absolute_path.exists(),
        absolute_path.is_dir(),
    )

    try:
        env = ALNSEnvironment(
            problem_instance=dataset_label,
            max_iterations=5,
            seed=42,
            problem_instance_paths=[dataset_label],
            problem_sampling_strategy="round_robin",
        )

        LOGGER.info("%s Action space: %s", LOG_PREFIX, env.action_space)
        LOGGER.info("%s Observation space: %s", LOG_PREFIX, env.observation_space)

        LOGGER.info("%s Checking environment with Stable-Baselines3...", LOG_PREFIX)
        check_env(env, warn=True)
        LOGGER.info("%s Environment check passed!", LOG_PREFIX)

        obs, info = env.reset()
        LOGGER.info("%s Initial observation shape: %s", LOG_PREFIX, obs.shape)
        LOGGER.info("%s Initial cost: %s", LOG_PREFIX, info.get("initial_cost", "N/A"))

        total_reward = 0.0
        for step in range(5):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward

            step_info = info.get("step_info", {})
            LOGGER.info(
                "%s Step %d: reward=%.2f cost=%.2f done=%s",
                LOG_PREFIX,
                step + 1,
                reward,
                float(step_info.get("current_cost", 0.0)),
                done,
            )

            if done or truncated:
                break

        LOGGER.info("%s Total reward: %.2f", LOG_PREFIX, total_reward)
        LOGGER.info("%s Manual test completed successfully!", LOG_PREFIX)

        env.close()
        return True

    except Exception as exc:  # pragma: no cover
        LOGGER.exception("%s Manual test failed: %s", LOG_PREFIX, exc)
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
    enable_operator_logging: bool = True,
) -> Tuple[PPO, DummyVecEnv]:
    """Train a PPO agent on the ALNS environment using provided datasets."""

    LOGGER.info("%s Training PPO agent for %d timesteps", LOG_PREFIX, total_timesteps)

    train_instances = list(train_instance_paths) if train_instance_paths else ["SMALL_1"]
    if not train_instances:
        raise ValueError("train_instance_paths must provide at least one dataset")
    LOGGER.info(
        "%s Training on %d instances (sampling=%s)",
        LOG_PREFIX,
        len(train_instances),
        sampling_strategy,
    )

    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)

    model_dir = Path(model_save_path).parent
    model_dir.mkdir(parents=True, exist_ok=True)

    operator_logs_dir = log_dir_path.parent / "operator_usage"
    operator_logging_dir_str = str(operator_logs_dir) if enable_operator_logging else None

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
        env_kwargs["enable_operator_logging"] = enable_operator_logging
        env_kwargs["operator_logging_mode"] = "train"
        if operator_logging_dir_str:
            env_kwargs["operator_logging_dir"] = operator_logging_dir_str

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
    LOGGER.info("%s Model saved to: %s", LOG_PREFIX, model_save_path)

    return model, vec_env


def evaluate_trained_model(
    model_path: str,
    problem_paths: Sequence[str],
    n_eval_episodes: Optional[int] = None,
    deterministic: bool = True,
    output_dir: Optional[str] = None,
    max_iterations: int = 100,
    enable_operator_logging: bool = True,
    action_module: Optional[str] = None,
    state_module: Optional[str] = None,
    reward_module: Optional[str] = None,
) -> Tuple[float, float, List[Dict[str, Any]]]:
    """Evaluate a trained PPO model across a set of problem instances."""

    eval_paths = list(problem_paths) if problem_paths else ["SMALL_1"]
    episodes = n_eval_episodes or len(eval_paths)

    LOGGER.info(
        "%s Evaluating trained model across %d episodes",
        LOG_PREFIX,
        episodes,
    )
    model = PPO.load(model_path)
    LOGGER.info("%s Model loaded from: %s", LOG_PREFIX, model_path)

    resolved_state_module = _resolve_state_module(model, state_module)
    resolved_action_module = action_module
    resolved_reward_module = reward_module

    operator_logs_dir_eval = (
        Path(output_dir) / "operator_usage_eval"
        if output_dir and enable_operator_logging
        else None
    )

    eval_env = ALNSEnvironment(
        problem_instance=eval_paths[0],
        max_iterations=max_iterations,
        seed=123,
        problem_instance_paths=eval_paths,
        problem_sampling_strategy="round_robin",
        enable_operator_logging=enable_operator_logging,
        operator_logging_mode="eval",
        operator_logging_dir=str(operator_logs_dir_eval) if operator_logs_dir_eval else None,
        action_module=resolved_action_module or DEFAULT_ACTION_KEY,
        state_module=resolved_state_module or DEFAULT_STATE_KEY,
        reward_module=resolved_reward_module or DEFAULT_REWARD_KEY,
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
        enable_operator_logging=enable_operator_logging,
        operator_logging_mode="eval_detail",
        operator_logging_dir=str(operator_logs_dir_eval) if operator_logs_dir_eval else None,
        action_module=resolved_action_module or DEFAULT_ACTION_KEY,
        state_module=resolved_state_module or DEFAULT_STATE_KEY,
        reward_module=resolved_reward_module or DEFAULT_REWARD_KEY,
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

    LOGGER.info(
        "%s Evaluation: mean reward %.2f +/- %.2f",
        LOG_PREFIX,
        mean_reward,
        std_reward,
    )
    return mean_reward, std_reward, detailed_stats


def compare_with_baseline(
    problem_paths: Sequence[str],
    max_iterations: int = 100,
    output_dir: Optional[str] = None,
    enable_operator_logging: bool = True,
) -> Tuple[float, float, List[Dict[str, Any]]]:
    """Evaluate a random baseline policy across the same problem instances."""

    baseline_paths = list(problem_paths) if problem_paths else ["SMALL_1"]
    operator_logs_dir_baseline = (
        Path(output_dir) / "operator_usage_baseline"
        if output_dir and enable_operator_logging
        else None
    )

    env = ALNSEnvironment(
        problem_instance=baseline_paths[0],
        max_iterations=max_iterations,
        seed=2048,
        problem_instance_paths=baseline_paths,
        problem_sampling_strategy="round_robin",
        enable_operator_logging=enable_operator_logging,
        operator_logging_mode="baseline",
        operator_logging_dir=str(operator_logs_dir_baseline) if operator_logs_dir_baseline else None,
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

    LOGGER.info(
        "%s Random policy: mean reward %.2f, mean improvement %.2f",
        LOG_PREFIX,
        mean_reward,
        mean_improvement,
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


_STATE_SHAPE_CACHE: Dict[Tuple[int, ...], str] = {}


def _resolve_state_module(
    policy_model: Optional[PPO],
    explicit_module: Optional[str],
) -> Optional[str]:
    if explicit_module:
        return explicit_module
    if policy_model is None:
        return None

    obs_space = getattr(policy_model, "observation_space", None)
    shape = getattr(obs_space, "shape", None)
    if not shape:
        return None

    try:
        target_shape = tuple(int(dim) for dim in shape)
    except TypeError:  # pragma: no cover - defensive
        return None

    if not target_shape:
        return None

    cached = _STATE_SHAPE_CACHE.get(target_shape)
    if cached:
        return cached

    for key, cls in list_registered_states().items():
        try:
            encoder = cls()
        except Exception:  # pragma: no cover - defensive instantiation guard
            continue
        space = encoder.space()
        space_shape = getattr(space, "shape", None)
        if space_shape is None:
            continue
        try:
            encoder_shape = tuple(int(dim) for dim in space_shape)
        except TypeError:  # pragma: no cover - defensive
            continue
        if encoder_shape == target_shape:
            _STATE_SHAPE_CACHE[target_shape] = key
            return key

    return None


def run_episode_with_policy(
    policy_model: Optional[PPO],
    problem_path: str,
    seed: int,
    max_iterations: int,
    deterministic: bool = True,
    enable_operator_logging: bool = True,
    operator_logging_mode: str = "comparison",
    operator_logging_dir: Optional[str] = None,
    shared_snapshot: Optional[Any] = None,
    capture_snapshot: bool = False,
    action_module: Optional[str] = None,
    state_module: Optional[str] = None,
    reward_module: Optional[str] = None,
) -> Dict[str, Any]:
    resolved_state_module = _resolve_state_module(policy_model, state_module)

    env_kwargs: Dict[str, Any] = {
        "problem_instance": problem_path,
        "max_iterations": max_iterations,
        "seed": seed,
        "problem_instance_paths": [problem_path],
        "problem_sampling_strategy": "round_robin",
        "enable_operator_logging": enable_operator_logging,
        "operator_logging_mode": operator_logging_mode,
    }
    if operator_logging_dir:
        env_kwargs["operator_logging_dir"] = operator_logging_dir
    if policy_model is None:
        env_kwargs["force_baseline_improvement"] = True
        env_kwargs["baseline_improvement_idx"] = 0
    if action_module:
        env_kwargs["action_module"] = action_module
    if resolved_state_module:
        env_kwargs["state_module"] = resolved_state_module
    if reward_module:
        env_kwargs["reward_module"] = reward_module

    env = ALNSEnvironment(**env_kwargs)

    reset_options: Dict[str, Any] = {}
    if shared_snapshot is not None:
        reset_options["initial_snapshot"] = shared_snapshot

    obs, _ = env.reset(seed=seed, options=reset_options or None)
    snapshot_obj = env.get_initial_snapshot() if capture_snapshot else None
    done = False
    truncated = False
    steps = 0
    baseline_rng: Optional[np.random.Generator] = None
    if policy_model is None:
        baseline_rng = np.random.default_rng(seed)

    while not done and not truncated and steps < max_iterations:
        if policy_model is None:
            if baseline_rng is None or not hasattr(env.action_impl, "num_pairs"):
                action = env.action_space.sample()
            else:
                num_pairs = getattr(env.action_impl, "num_pairs")()
                if num_pairs <= 0:
                    action = env.action_space.sample()
                else:
                    action = int(baseline_rng.integers(0, num_pairs))
        else:
            action_array, _ = policy_model.predict(obs, deterministic=deterministic)
            action = int(action_array.item()) if hasattr(action_array, "item") else int(action_array)
        obs, reward, done, truncated, info = env.step(action)
        if policy_model is None and info.get("baseline_improvement_triggered"):
            env.log_improvement_usage()
        steps += 1

    stats = env.get_episode_statistics()
    env.close()

    module_info = {
        "action": getattr(env, "action_module_name", action_module),
        "state": getattr(env, "state_module_name", resolved_state_module),
        "reward": getattr(env, "reward_module_name", reward_module),
    }

    result: Dict[str, Any] = {
        "problem_path": problem_path,
        "best_cost": stats.get("best_cost"),
        "final_cost": stats.get("final_cost"),
        "cost_history": stats.get("cost_history", []),
        "best_cost_history": stats.get("best_cost_history", []),
        "iterations": stats.get("total_iterations", steps),
        "initial_cost": stats.get("initial_cost"),
        "elapsed_seconds": stats.get("elapsed_seconds", []),
        "elapsed_ms_steps": stats.get("elapsed_ms_steps", []),
        "elapsed_ms_cumulative": stats.get("elapsed_ms_cumulative", []),
        "modules": module_info,
        "module_versions": getattr(env, "module_versions", {}),
    }
    if snapshot_obj is not None:
        result["initial_snapshot"] = snapshot_obj

    return result


def compare_model_against_baseline(
    model: PPO,
    problem_paths: Sequence[str],
    seeds: Sequence[int] = (42,),
    max_iterations: int = 100,
    output_dir: str = "logs/ppo_alns/model_vs_baseline",
    deterministic: bool = True,
    enable_operator_logging: bool = True,
    action_module: Optional[str] = None,
    state_module: Optional[str] = None,
    reward_module: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Run side-by-side comparisons for the PPO model and random baseline."""

    comparison_dir = Path(output_dir)
    comparison_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    rl_current_series: List[Tuple[List[float], List[float]]] = []
    rl_best_series: List[Tuple[List[float], List[float]]] = []
    baseline_current_series: List[Tuple[List[float], List[float]]] = []
    baseline_best_series: List[Tuple[List[float], List[float]]] = []
    max_elapsed_time = 0.0
    dataset_matchups: Dict[str, Dict[str, List[float]]] = {}

    operator_logs_dir_rl = comparison_dir / "operator_usage_rl"
    operator_logs_dir_baseline = comparison_dir / "operator_usage_random"
    operator_logs_dir_rl_str = str(operator_logs_dir_rl) if enable_operator_logging else None
    operator_logs_dir_baseline_str = str(operator_logs_dir_baseline) if enable_operator_logging else None

    for dataset_path in problem_paths:
        dataset_label = _format_dataset_label(dataset_path)
        for seed in seeds:
            rl_result = run_episode_with_policy(
                policy_model=model,
                problem_path=dataset_path,
                seed=seed,
                max_iterations=max_iterations,
                deterministic=deterministic,
                enable_operator_logging=enable_operator_logging,
                operator_logging_mode="comparison_rl",
                operator_logging_dir=operator_logs_dir_rl_str,
                capture_snapshot=True,
                action_module=action_module,
                state_module=state_module,
                reward_module=reward_module,
            )

            snapshot_for_baseline = rl_result.pop("initial_snapshot", None)
            resolved_modules = rl_result.get("modules", {})
            baseline_action_module = resolved_modules.get("action", action_module)
            baseline_state_module = resolved_modules.get("state", state_module)
            baseline_reward_module = resolved_modules.get("reward", reward_module)

            baseline_result = run_episode_with_policy(
                policy_model=None,
                problem_path=dataset_path,
                seed=seed,
                max_iterations=max_iterations,
                deterministic=deterministic,
                enable_operator_logging=enable_operator_logging,
                operator_logging_mode="comparison_random",
                operator_logging_dir=operator_logs_dir_baseline_str,
                shared_snapshot=snapshot_for_baseline,
                action_module=baseline_action_module,
                state_module=baseline_state_module,
                reward_module=baseline_reward_module,
            )

            rl_cost_history = list(rl_result.get("cost_history", []))
            rl_best_history = list(rl_result.get("best_cost_history", []))
            baseline_cost_history = list(baseline_result.get("cost_history", []))
            baseline_best_history = list(baseline_result.get("best_cost_history", []))

            rl_elapsed_raw = rl_result.get("elapsed_seconds", [])
            baseline_elapsed_raw = baseline_result.get("elapsed_seconds", [])
            rl_elapsed = [float(v) for v in rl_elapsed_raw] if rl_elapsed_raw else []
            baseline_elapsed = [float(v) for v in baseline_elapsed_raw] if baseline_elapsed_raw else []

            rl_len = min(len(rl_cost_history), len(rl_best_history))
            baseline_len = min(len(baseline_cost_history), len(baseline_best_history))

            rl_cost_history = rl_cost_history[:rl_len]
            rl_best_history = rl_best_history[:rl_len]
            if len(rl_elapsed) >= rl_len and rl_len > 0:
                rl_elapsed = rl_elapsed[:rl_len]
            else:
                rl_elapsed = [float(i + 1) for i in range(rl_len)]

            baseline_cost_history = baseline_cost_history[:baseline_len]
            baseline_best_history = baseline_best_history[:baseline_len]
            if len(baseline_elapsed) >= baseline_len and baseline_len > 0:
                baseline_elapsed = baseline_elapsed[:baseline_len]
            else:
                baseline_elapsed = [float(i + 1) for i in range(baseline_len)]

            candidates = [
                value
                for value in rl_best_history + baseline_best_history
                if value is not None and value > 0
            ]
            min_target = min(candidates) if candidates else 1.0

            rl_current = [((value / min_target) - 1.0) * 100.0 for value in rl_cost_history]
            rl_best = [((value / min_target) - 1.0) * 100.0 for value in rl_best_history]
            baseline_current = [((value / min_target) - 1.0) * 100.0 for value in baseline_cost_history]
            baseline_best = [((value / min_target) - 1.0) * 100.0 for value in baseline_best_history]

            rl_current_series.append((rl_elapsed, rl_current))
            rl_best_series.append((rl_elapsed, rl_best))
            baseline_current_series.append((baseline_elapsed, baseline_current))
            baseline_best_series.append((baseline_elapsed, baseline_best))

            if rl_elapsed:
                max_elapsed_time = max(max_elapsed_time, rl_elapsed[-1])
            if baseline_elapsed:
                max_elapsed_time = max(max_elapsed_time, baseline_elapsed[-1])

            matchup = dataset_matchups.setdefault(dataset_label, {"rl": [], "baseline": []})
            rl_best_cost = rl_result.get("best_cost")
            baseline_best_cost = baseline_result.get("best_cost")
            if rl_best_cost is not None:
                matchup["rl"].append(float(rl_best_cost))
            if baseline_best_cost is not None:
                matchup["baseline"].append(float(baseline_best_cost))

            def _prepend_zero(time_vals: List[float], series_vals: List[float]) -> Tuple[List[float], List[float]]:
                if not time_vals or not series_vals:
                    return time_vals, series_vals
                if time_vals[0] <= 0.0:
                    return time_vals, series_vals
                return [0.0] + time_vals, [series_vals[0]] + series_vals

            rl_time_for_plot, rl_current_for_plot = _prepend_zero(rl_elapsed, rl_current)
            _, rl_best_for_plot = _prepend_zero(rl_elapsed, rl_best)
            baseline_time_for_plot, baseline_current_for_plot = _prepend_zero(baseline_elapsed, baseline_current)
            _, baseline_best_for_plot = _prepend_zero(baseline_elapsed, baseline_best)

            dataset_slug = _slugify_path(dataset_path)
            plt.figure(figsize=(10, 6))
            plt.plot(rl_time_for_plot, rl_current_for_plot, label="RL current gap", color="tab:blue")
            plt.plot(rl_time_for_plot, rl_best_for_plot, label="RL best gap", color="tab:blue", linestyle="--")
            plt.plot(
                baseline_time_for_plot,
                baseline_current_for_plot,
                label="Baseline current gap",
                color="tab:orange",
            )
            plt.plot(
                baseline_time_for_plot,
                baseline_best_for_plot,
                label="Baseline best gap",
                color="tab:orange",
                linestyle="--",
            )
            plt.xlabel("Elapsed time (s)")
            plt.ylabel("Gap to best-known (%)")
            plt.title(f"Model vs Baseline gap ({dataset_label}, seed {seed})")
            plt.legend()
            plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
            plt.tight_layout()
            plot_file = comparison_dir / f"comparison_{dataset_slug}_seed{seed}.png"
            plt.savefig(plot_file)
            plt.close()

            if rl_best_cost is None or baseline_best_cost is None:
                LOGGER.warning(
                    "%s Missing best_cost metrics for %s (seed %s); skipping detailed comparison entry.",
                    LOG_PREFIX,
                    dataset_label,
                    seed,
                )
                continue

            delta = float(baseline_best_cost - rl_best_cost)
            results.append(
                {
                    "dataset_path": dataset_path,
                    "dataset": dataset_label,
                    "seed": seed,
                    "rl_best_cost": rl_best_cost,
                    "baseline_best_cost": baseline_best_cost,
                    "best_cost_delta": delta,
                    "rl_iterations": rl_result["iterations"],
                    "baseline_iterations": baseline_result["iterations"],
                    "plot": str(plot_file),
                }
            )

            LOGGER.info(
                "%s %s (seed %s): RL best=%.2f baseline best=%.2f delta=%+0.2f",
                LOG_PREFIX,
                dataset_label,
                seed,
                rl_best_cost,
                baseline_best_cost,
                delta,
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
        LOGGER.info(
            "%s Avg best-cost delta (baseline - RL): %+0.2f",
            LOG_PREFIX,
            avg_delta,
        )
        LOGGER.info(
            "%s Detailed comparison saved to: %s",
            LOG_PREFIX,
            csv_path,
        )

    if rl_best_series:
        max_time = max(max_elapsed_time, 1.0)
        time_grid = np.linspace(0.0, max_time, 200)

        def _resample_series(time_values: List[float], series_values: List[float]) -> np.ndarray:
            if not time_values or not series_values:
                return np.zeros_like(time_grid)
            times = np.array(time_values, dtype=float)
            values = np.array(series_values, dtype=float)
            if times[0] > 0.0:
                times = np.insert(times, 0, 0.0)
                values = np.insert(values, 0, values[0])
            return np.interp(time_grid, times, values, left=values[0], right=values[-1])

        rl_best_mat = np.vstack([
            _resample_series(times, values) for times, values in rl_best_series
        ])
        baseline_best_mat = np.vstack([
            _resample_series(times, values) for times, values in baseline_best_series
        ])
        rl_current_mat = np.vstack([
            _resample_series(times, values) for times, values in rl_current_series
        ])
        baseline_current_mat = np.vstack([
            _resample_series(times, values) for times, values in baseline_current_series
        ])

        def _mean_std(mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            return np.mean(mat, axis=0), np.std(mat, axis=0)

        rl_best_mean, rl_best_std = _mean_std(rl_best_mat)
        baseline_best_mean, baseline_best_std = _mean_std(baseline_best_mat)

        plt.figure(figsize=(10, 6))
        plt.plot(time_grid, baseline_best_mean, label="Baseline best gap", color="tab:orange")
        plt.fill_between(
            time_grid,
            baseline_best_mean - baseline_best_std,
            baseline_best_mean + baseline_best_std,
            color="tab:orange",
            alpha=0.2,
        )
        plt.plot(time_grid, rl_best_mean, label="RL best gap", color="tab:blue")
        plt.fill_between(
            time_grid,
            rl_best_mean - rl_best_std,
            rl_best_mean + rl_best_std,
            color="tab:blue",
            alpha=0.2,
        )
        plt.xlabel("Elapsed time (s)")
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
        plt.plot(time_grid, baseline_current_mean, label="Baseline current gap", color="tab:orange")
        plt.fill_between(
            time_grid,
            baseline_current_mean - baseline_current_std,
            baseline_current_mean + baseline_current_std,
            color="tab:orange",
            alpha=0.2,
        )
        plt.plot(time_grid, rl_current_mean, label="RL current gap", color="tab:blue")
        plt.fill_between(
            time_grid,
            rl_current_mean - rl_current_std,
            rl_current_mean + rl_current_std,
            color="tab:blue",
            alpha=0.2,
        )
        plt.xlabel("Elapsed time (s)")
        plt.ylabel("Gap to best-known (%)")
        plt.title("Combined current-cost convergence (mean +/- 1 std)")
        plt.legend()
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        combined_plot_current = comparison_dir / "combined_current_cost_convergence.png"
        plt.tight_layout()
        plt.savefig(combined_plot_current)
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
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )

    LOGGER.info("%s ALNS Reinforcement Learning Training", LOG_PREFIX)

    dataset_splits = prepare_dataset_splits("small")
    train_paths = dataset_splits.get("train", [])
    test_paths = dataset_splits.get("test", [])

    if not train_paths or not test_paths:
        LOGGER.error("%s No training or test datasets were found. Aborting.", LOG_PREFIX)
        return

    LOGGER.info(
        "%s Datasets prepared: %d train / %d test instances",
        LOG_PREFIX,
        len(train_paths),
        len(test_paths),
    )

    primary_train_instance = train_paths[0]
    if not test_environment_manually(primary_train_instance):
        LOGGER.error("%s Manual environment test failed, aborting training", LOG_PREFIX)
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

    LOGGER.info("%s Summary: trained agent %.2f +/- %.2f", LOG_PREFIX, mean_reward, std_reward)
    LOGGER.info("%s Summary: random baseline %.2f", LOG_PREFIX, baseline_reward)
    LOGGER.info(
        "%s Summary: reward improvement %+0.2f (%+0.1f%%)",
        LOG_PREFIX,
        improvement_vs_baseline,
        pct_improvement,
    )

    if comparison_results:
        rl_wins = sum(1 for row in comparison_results if row["best_cost_delta"] > 0)
        ties = sum(1 for row in comparison_results if row["best_cost_delta"] == 0)
        total = len(comparison_results)
        LOGGER.info(
            "%s Best-cost wins: RL %d, ties %d, baseline %d",
            LOG_PREFIX,
            rl_wins,
            ties,
            total - rl_wins - ties,
        )

    vec_env.close()


if __name__ == "__main__":
    main()
