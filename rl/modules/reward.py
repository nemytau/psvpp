"""Reward function implementations for the ALNS RL environment."""

from __future__ import annotations

from typing import Any, Dict

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


@register_reward("delta_cost_v4")
class DeltaCostStepDeltaReward(BaseRewardFunction):
    """Reward based on step-wise cost deltas (current + best)."""

    version = "4.0"

    def compute(self, result: Dict[str, Any], env: Any) -> float:
        initial = env.initial_cost or result.get("initial_cost", 0.0) or 1.0

        prev_cur = float(getattr(env, "last_current_cost", initial))
        prev_best = float(getattr(env, "last_best_cost", initial))

        cur = float(result.get("current_cost", prev_cur))
        best = float(result.get("best_cost", cur))

        denom_cur = max(abs(prev_cur), 1e-9)
        denom_best = max(abs(prev_best), 1e-9)

        delta_cur = (prev_cur - cur) / denom_cur
        delta_best = (prev_best - best) / denom_best

        if delta_cur < 0.0:
            delta_cur *= 0.3

        reward = 10.0 * delta_cur + 30.0 * delta_best

        acceptance_type = result.get("acceptance_type") or getattr(env, "last_acceptance_type", "unknown")
        is_new_best = bool(result.get("is_new_best", False))

        no_progress = (
            abs(cur - prev_cur) <= 1e-9
            and abs(best - prev_best) <= 1e-9
        )
        stagnation_count = int(getattr(env, "stagnation_count", 0) or 0)
        stagn_penalty = 0.0025 * min(stagnation_count, 200)

        made_progress = not no_progress

        if acceptance_type == "improvement":
            if made_progress:
                reward += 1.0
        elif acceptance_type == "annealing":
            if made_progress:
                reward += 0.2
        else:
            reward -= 0.5
        if is_new_best:
            reward += 5.0

        if no_progress:
            reward = -0.2 - stagn_penalty
        else:
            reward -= stagn_penalty

        if result.get("improvement_only_action") or result.get("improvement_operator_idx") is not None:
            reward -= 0.05

        return float(np.clip(reward, -5.0, 20.0))


@register_reward("delta_cost_v5")
class DeltaCostCandidateReward(BaseRewardFunction):
    """Reward from the candidate solution's own cost, independent of SA's
    stochastic accept/reject decision.

    delta_cost_v3's acceptance_bonus and delta_cost_v4's no_progress
    flattening both make reward magnitude partly a function of SA's random
    accept/reject coin flip rather than purely the proposed move's quality --
    a rejected candidate that nearly matched the current solution gets the
    exact same flat penalty as one that made things drastically worse, so the
    operator gets no gradient telling those two cases apart.

    This variant reads `result["candidate_cost"]` -- the solution's cost
    *before* SA's accept/reject decision runs (engine.rs computed this all
    along but never exposed it; the same key used to be silently aliased to
    the post-decision current_cost, see python_interface.rs) -- and computes
    reward purely from how that candidate compares to the previous current
    and best costs, regardless of what SA actually did with it. No
    accepted/acceptance_type branching at all.
    """

    version = "5.0"

    def compute(self, result: Dict[str, Any], env: Any) -> float:
        initial = env.initial_cost or result.get("initial_cost", 0.0) or 1.0

        prev_cur = float(getattr(env, "last_current_cost", initial))
        prev_best = float(getattr(env, "last_best_cost", initial))

        candidate = float(result.get("candidate_cost", result.get("current_cost", prev_cur)))

        denom_cur = max(abs(prev_cur), 1e-9)
        denom_best = max(abs(prev_best), 1e-9)

        delta_vs_current = (prev_cur - candidate) / denom_cur
        delta_vs_best = (prev_best - candidate) / denom_best

        # Same asymmetric dampening as delta_cost_v4: soften the downside so a
        # single bad candidate doesn't swamp the signal, without hiding it
        # entirely (unlike v4's no_progress flattening).
        if delta_vs_current < 0.0:
            delta_vs_current *= 0.3
        if delta_vs_best < 0.0:
            delta_vs_best *= 0.3

        reward = 10.0 * delta_vs_current + 30.0 * delta_vs_best

        if result.get("improvement_only_action") or result.get("improvement_operator_idx") is not None:
            reward -= 0.05

        return float(np.clip(reward, -5.0, 20.0))
