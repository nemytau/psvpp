"""Reward function implementations for the ALNS RL environment."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from rl.registries import register_reward


class BaseRewardFunction:
    """Base reward-function interface."""

    version: str = "base"

    def compute(self, result: Dict[str, Any], env: Any) -> float:  # pragma: no cover - interface only
        raise NotImplementedError


@register_reward("delta_cost_v3")
class DeltaCostReward(BaseRewardFunction):
    """Reward shaped around relative cost improvements and acceptance."""

    version = "3.0"

    def compute(self, result: Dict[str, Any], env: Any) -> float:
        initial_cost = env.initial_cost
        if not initial_cost or initial_cost <= 0:
            return 0.0

        current_cost = float(result.get("current_cost", initial_cost))
        best_cost = float(result.get("best_cost", current_cost))
        accepted = bool(result.get("accepted", False))
        is_new_best = bool(result.get("is_new_best", False))

        cost_improvement = (initial_cost - current_cost) / initial_cost
        best_improvement = (initial_cost - best_cost) / initial_cost

        improvement_reward = cost_improvement * 10.0
        best_reward = best_improvement * 15.0
        acceptance_bonus = 1.0 if accepted else -0.5
        new_best_bonus = 5.0 if is_new_best else 0.0
        progress_bonus = 0.1 if env.iteration > 0 else 0.0

        total_reward = (
            improvement_reward
            + best_reward
            + acceptance_bonus
            + new_best_bonus
            + progress_bonus
        )

        return float(np.clip(total_reward, -20.0, 50.0))
