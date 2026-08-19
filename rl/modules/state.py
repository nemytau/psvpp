"""State encoder implementations for the ALNS RL environment."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from gymnasium import spaces

from rl.registries import register_state

INSTANCE_FEATURE_KEYS = (
    "num_installations",
    "total_visits",
    "num_vessels",
    "total_deck_demand",
    "avg_distance_km",
    "max_distance_km",
)

INSTANCE_NORMALIZATION = {
    "num_installations": 120.0,
    "total_visits": 300.0,
    "num_vessels": 60.0,
    "total_deck_demand": 6000.0,
    "avg_distance_km": 600.0,
    "max_distance_km": 800.0,
}


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
        recent_rewards = (recent_rewards + [0.0] * 3)[:3]
        obs_values.extend(float(val) for val in recent_rewards)

        avg_load_util = float(metrics.get("avg_vessel_load_utilization", 0.0))
        max_load_util = float(metrics.get("max_vessel_load_utilization", avg_load_util))
        min_load_util = float(metrics.get("min_vessel_load_utilization", avg_load_util))
        avg_time_util = float(metrics.get("avg_vessel_time_utilization", 0.0))
        max_time_util = float(metrics.get("max_vessel_time_utilization", avg_time_util))

        obs_values.append(avg_load_util)
        obs_values.append(max_load_util)
        obs_values.append(min_load_util)
        obs_values.append(avg_time_util)
        obs_values.append(max_time_util)

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


@register_state("features_v3")
class OperatorFeatureStateEncoderV3(BaseStateEncoder):
    """Extended features_v2 with simple operator/history signals."""

    version = "3.1"

    def __init__(self) -> None:
        self._base_encoder = OperatorFeatureStateEncoder()
        self._space = spaces.Box(low=-np.inf, high=np.inf, shape=(41,), dtype=np.float32)

    def space(self) -> spaces.Box:
        return self._space

    def encode(self, result: Optional[Dict[str, Any]], env: Any) -> np.ndarray:
        base_obs = self._base_encoder.encode(result, env).tolist()

        last_op_id = float(getattr(env, "last_operator_id", -1.0))
        last_op_type_id = float(getattr(env, "last_operator_type_id", -1.0))
        if result and "fraction_removed" in result:
            try:
                frac_removed = float(result.get("fraction_removed", 0.0))
            except (TypeError, ValueError):  # pragma: no cover - defensive
                frac_removed = 0.0
        else:
            frac_removed = float(getattr(env, "last_fraction_removed", 0.0))
        stagnation = float(getattr(env, "stagnation_count", 0.0))
        last_reward = float(getattr(env, "last_reward", 0.0))

        instance_stats: Dict[str, float] = getattr(env, "current_instance_stats", {}) or {}
        instance_features: list[float] = []
        for key in INSTANCE_FEATURE_KEYS:
            raw_value = float(instance_stats.get(key, 0.0))
            scale = INSTANCE_NORMALIZATION.get(key, 1.0)
            if scale <= 0.0:
                normalized = 0.0
            else:
                normalized = raw_value / scale
            instance_features.append(float(np.clip(normalized, 0.0, 1.0)))

        obs_values = (
            base_obs
            + [last_op_id, last_op_type_id, frac_removed, stagnation, last_reward]
            + instance_features
        )[:41]
        obs_array = np.array(obs_values, dtype=np.float32)
        return np.nan_to_num(obs_array, nan=0.0, posinf=1e6, neginf=-1e6)


@register_state("features_improvement_v1")
class ImprovementOnlyStateEncoderV1(BaseStateEncoder):
    """Improvement-focused state for `improvement_only_v1` action space.

    Keeps the 41-dim shape used by features_v3 for model compatibility, but
    replaces noisy backbone-operator identifiers with normalized signals tied
    to improvement decisions.
    """

    version = "improvement-1.0"

    def __init__(self) -> None:
        self._base_encoder = OperatorFeatureStateEncoder()
        self._space = spaces.Box(low=-np.inf, high=np.inf, shape=(41,), dtype=np.float32)

    def space(self) -> spaces.Box:
        return self._space

    def encode(self, result: Optional[Dict[str, Any]], env: Any) -> np.ndarray:
        base_obs = self._base_encoder.encode(result, env).tolist()

        last_improvement_idx = float(getattr(env, "last_improvement_idx", -1.0))
        num_improvement = float(max(int(getattr(env, "num_improvement", 0)), 0))

        improvement_applied = 1.0 if last_improvement_idx >= 0.0 else 0.0
        if num_improvement > 1.0 and last_improvement_idx >= 0.0:
            normalized_improvement_idx = float(np.clip(last_improvement_idx / (num_improvement - 1.0), 0.0, 1.0))
        elif num_improvement == 1.0 and last_improvement_idx >= 0.0:
            normalized_improvement_idx = 1.0
        else:
            normalized_improvement_idx = 0.0

        last_improvement_type_id = float(getattr(env, "last_improvement_type_id", -1.0))
        if last_improvement_type_id >= 0.0:
            normalized_improvement_type_id = float(np.clip(last_improvement_type_id / 10.0, 0.0, 1.0))
        else:
            normalized_improvement_type_id = 0.0

        stagnation = float(getattr(env, "stagnation_count", 0.0))
        max_iterations = float(max(int(getattr(env, "max_iterations", 1)), 1))
        normalized_stagnation = float(np.clip(stagnation / max_iterations, 0.0, 1.0))

        last_reward = float(getattr(env, "last_reward", 0.0))
        clipped_last_reward = float(np.clip(last_reward / 20.0, -1.0, 1.0))

        instance_stats: Dict[str, float] = getattr(env, "current_instance_stats", {}) or {}
        instance_features: list[float] = []
        for key in INSTANCE_FEATURE_KEYS:
            raw_value = float(instance_stats.get(key, 0.0))
            scale = INSTANCE_NORMALIZATION.get(key, 1.0)
            if scale <= 0.0:
                normalized = 0.0
            else:
                normalized = raw_value / scale
            instance_features.append(float(np.clip(normalized, 0.0, 1.0)))

        obs_values = (
            base_obs
            + [
                improvement_applied,
                normalized_improvement_idx,
                normalized_improvement_type_id,
                normalized_stagnation,
                clipped_last_reward,
            ]
            + instance_features
        )[:41]
        obs_array = np.array(obs_values, dtype=np.float32)
        return np.nan_to_num(obs_array, nan=0.0, posinf=1e6, neginf=-1e6)


@register_state("features_improvement_v2")
class ImprovementOnlyStateEncoderV2(BaseStateEncoder):
    """Improvement-focused state with short-term operator history.

    Adds latest and previous improvement IDs/types plus chain streak signal so
    policy can learn when repeating the same improvement is likely unhelpful.
    """

    version = "improvement-2.0"

    def __init__(self) -> None:
        self._base_encoder = OperatorFeatureStateEncoder()
        self._space = spaces.Box(low=-np.inf, high=np.inf, shape=(43,), dtype=np.float32)

    def space(self) -> spaces.Box:
        return self._space

    def _normalize_improvement_idx(self, idx: float, num_improvement: float) -> float:
        if idx < 0.0:
            return 0.0
        if num_improvement > 1.0:
            return float(np.clip(idx / (num_improvement - 1.0), 0.0, 1.0))
        if num_improvement == 1.0:
            return 1.0
        return 0.0

    def _normalize_type_id(self, type_id: float) -> float:
        if type_id < 0.0:
            return 0.0
        return float(np.clip(type_id / 10.0, 0.0, 1.0))

    def encode(self, result: Optional[Dict[str, Any]], env: Any) -> np.ndarray:
        base_obs = self._base_encoder.encode(result, env).tolist()

        num_improvement = float(max(int(getattr(env, "num_improvement", 0)), 0))
        last_improvement_idx = float(getattr(env, "last_improvement_idx", -1.0))
        last_improvement_type_id = float(getattr(env, "last_improvement_type_id", -1.0))
        prev_improvement_idx = float(getattr(env, "prev_improvement_idx", -1.0))
        prev_improvement_type_id = float(getattr(env, "prev_improvement_type_id", -1.0))

        last_improvement_applied = 1.0 if last_improvement_idx >= 0.0 else 0.0
        normalized_last_idx = self._normalize_improvement_idx(last_improvement_idx, num_improvement)
        normalized_last_type = self._normalize_type_id(last_improvement_type_id)
        normalized_prev_idx = self._normalize_improvement_idx(prev_improvement_idx, num_improvement)
        normalized_prev_type = self._normalize_type_id(prev_improvement_type_id)

        chain_streak = float(max(int(getattr(env, "chain_improvement_streak", 0)), 0))
        max_iterations = float(max(int(getattr(env, "max_iterations", 1)), 1))
        normalized_chain_streak = float(np.clip(chain_streak / max_iterations, 0.0, 1.0))

        last_reward = float(getattr(env, "last_reward", 0.0))
        clipped_last_reward = float(np.clip(last_reward / 20.0, -1.0, 1.0))

        instance_stats: Dict[str, float] = getattr(env, "current_instance_stats", {}) or {}
        instance_features: list[float] = []
        for key in INSTANCE_FEATURE_KEYS:
            raw_value = float(instance_stats.get(key, 0.0))
            scale = INSTANCE_NORMALIZATION.get(key, 1.0)
            if scale <= 0.0:
                normalized = 0.0
            else:
                normalized = raw_value / scale
            instance_features.append(float(np.clip(normalized, 0.0, 1.0)))

        obs_values = (
            base_obs
            + [
                last_improvement_applied,
                normalized_last_idx,
                normalized_last_type,
                normalized_prev_idx,
                normalized_prev_type,
                normalized_chain_streak,
                clipped_last_reward,
            ]
            + instance_features
        )[:43]
        obs_array = np.array(obs_values, dtype=np.float32)
        return np.nan_to_num(obs_array, nan=0.0, posinf=1e6, neginf=-1e6)


@register_state("features_improvement_v3")
class ImprovementOnlyStateEncoderV3(BaseStateEncoder):
    """features_improvement_v1 plus an explicit SA accept/reject signal.

    The reward is partly a function of SA's stochastic accept/reject decision
    (see delta_cost_v3's acceptance_bonus and delta_cost_v4's no_progress
    flattening), but until now the policy had no way to *observe* that
    decision -- temperature and stagnation_count are exposed, but not
    `accepted` or `acceptance_type`, even though both are already computed
    every step (see rl_alns_environment.py's step(), which sets
    result["accepted"]/result["acceptance_type"] before reward/state are
    computed). Without this, the policy can't distinguish "this operator's
    candidate was rejected by SA randomness" from "this operator's candidate
    was actually a bad move" purely from the observation.

    Adds 4 dims on top of features_improvement_v1's 41: `accepted` (bool) and
    a 3-way one-hot for acceptance_type collapsed from the six raw values
    (`improvement`/`annealing`/`rejected` -- `rejected_no_change` and
    `infeasible` fold into `rejected`; `unknown`, seen only before the first
    step, is the implicit all-zero case).
    """

    version = "improvement-3.0"

    _ACCEPTANCE_BUCKETS = ("improvement", "annealing", "rejected")
    _ACCEPTANCE_TYPE_TO_BUCKET = {
        "improvement": "improvement",
        "annealing": "annealing",
        "rejected": "rejected",
        "rejected_no_change": "rejected",
        "infeasible": "rejected",
    }

    def __init__(self) -> None:
        self._base_encoder = ImprovementOnlyStateEncoderV1()
        self._space = spaces.Box(low=-np.inf, high=np.inf, shape=(45,), dtype=np.float32)

    def space(self) -> spaces.Box:
        return self._space

    def encode(self, result: Optional[Dict[str, Any]], env: Any) -> np.ndarray:
        base_obs = self._base_encoder.encode(result, env).tolist()

        accepted = bool(result.get("accepted", False)) if result else False
        raw_acceptance_type = str(result.get("acceptance_type", "unknown")) if result else "unknown"
        bucket = self._ACCEPTANCE_TYPE_TO_BUCKET.get(raw_acceptance_type)
        one_hot = [1.0 if bucket == b else 0.0 for b in self._ACCEPTANCE_BUCKETS]

        obs_values = base_obs + [1.0 if accepted else 0.0] + one_hot
        obs_array = np.array(obs_values[:45], dtype=np.float32)
        return np.nan_to_num(obs_array, nan=0.0, posinf=1e6, neginf=-1e6)
