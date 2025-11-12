"""State encoder implementations for the ALNS RL environment."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import numpy as np
from gymnasium import spaces

from rl.registries import register_state


class BaseStateEncoder:
    """Base class for transforming ALNS metrics into observations."""

    version: str = "base"

    def space(self) -> spaces.Space:  # pragma: no cover - interface only
        raise NotImplementedError

    def encode(self, result: Optional[Dict[str, Any]], env: Any) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


@register_state("features_v2")
class OperatorFeatureStateEncoder(BaseStateEncoder):
    """Match the 30-feature observation vector used by PPO training."""

    version = "2.0"

    def __init__(self) -> None:
        self._space = spaces.Box(low=-np.inf, high=np.inf, shape=(30,), dtype=np.float32)

    def space(self) -> spaces.Box:
        return self._space

    def encode(self, result: Optional[Dict[str, Any]], env: Any) -> np.ndarray:
        metrics = self._resolve_metrics(result, env)
        obs_values: list[float] = []

        total_cost = float(metrics.get("total_cost", 0.0))
        best_cost = float(metrics.get("best_cost", total_cost))
        initial_cost = float(metrics.get("initial_cost", env.initial_cost or 0.0))
        initial_cost = initial_cost if initial_cost else max(total_cost, 1.0)
        denom = max(initial_cost, 1e-9)

        # Solution quality metrics
        obs_values.append(total_cost)
        obs_values.append(best_cost)
        obs_values.append(initial_cost)
        obs_values.append(total_cost / denom)
        obs_values.append(best_cost / denom)
        obs_values.append((initial_cost - best_cost) / denom)

        # Algorithm state metrics
        obs_values.append(float(metrics.get("temperature", 0.0)))
        obs_values.append(float(metrics.get("stagnation_count", 0.0)))
        obs_values.append(float(metrics.get("iteration", env.iteration)))
        progress = env.iteration / max(env.max_iterations, 1)
        obs_values.append(progress)

        # Solution structure metrics
        obs_values.append(float(metrics.get("num_voyages", 0.0)))
        obs_values.append(float(metrics.get("num_empty_voyages", 0.0)))
        obs_values.append(float(metrics.get("num_vessels_used", 0.0)))
        obs_values.append(float(metrics.get("avg_voyage_utilization", 0.0)))
        obs_values.append(1.0 if metrics.get("is_feasible", False) else 0.0)
        obs_values.append(1.0 if metrics.get("is_complete", False) else 0.0)

        # Operator success rates (destroy + repair, padded to len 3 each)
        destroy_rates = list(metrics.get("destroy_success_rates", []))
        repair_rates = list(metrics.get("repair_success_rates", []))
        destroy_rates = (destroy_rates + [0.5] * 3)[:3]
        repair_rates = (repair_rates + [0.5] * 3)[:3]
        obs_values.extend(float(rate) for rate in destroy_rates)
        obs_values.extend(float(rate) for rate in repair_rates)

        # Recent rewards statistics (pad/truncate to 5)
        recent_rewards = list(metrics.get("recent_rewards", []))
        recent_rewards = (recent_rewards + [0.0] * 5)[:5]
        obs_values.extend(float(val) for val in recent_rewards)

        count_recent = len([val for val in recent_rewards if isinstance(val, (int, float))])
        mean_recent = float(np.mean(recent_rewards)) if recent_rewards else 0.0
        std_recent = float(np.std(recent_rewards)) if count_recent > 1 else 0.0
        obs_values.append(float(count_recent))
        obs_values.append(mean_recent)
        obs_values.append(std_recent)

        while len(obs_values) < 30:
            obs_values.append(0.0)

        obs_array = np.array(obs_values[:30], dtype=np.float32)
        return np.nan_to_num(obs_array, nan=0.0, posinf=1e6, neginf=-1e6)

    def _resolve_metrics(self, result: Optional[Dict[str, Any]], env: Any) -> Dict[str, Any]:
        if result is None:
            return dict(env.alns.extract_solution_metrics())

        if "solution_metrics" in result and isinstance(result["solution_metrics"], dict):
            return dict(result["solution_metrics"])

        return dict(result)
