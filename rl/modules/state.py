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
        # Use progress_horizon (falls back to max_iterations if unset) rather than
        # max_iterations directly -- max_iterations legitimately differs between training
        # episodes and match_kisialiou eval restarts, which would otherwise make the same raw
        # iteration count normalize to different values in each context. See
        # docs/PAPER1_ACTION_SPACE_PLAN.md section 10.
        horizon = getattr(env, "progress_horizon", None) or env.max_iterations
        progress = env.iteration / max(horizon, 1)
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
        horizon = float(max(int(getattr(env, "progress_horizon", None) or getattr(env, "max_iterations", 1)), 1))
        normalized_stagnation = float(np.clip(stagnation / horizon, 0.0, 1.0))

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
        horizon = float(max(int(getattr(env, "progress_horizon", None) or getattr(env, "max_iterations", 1)), 1))
        normalized_chain_streak = float(np.clip(chain_streak / horizon, 0.0, 1.0))

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


@register_state("features_null_v1")
class NullStateEncoder(BaseStateEncoder):
    """Constant observation -- the policy sees nothing about the instance or search.

    State-design ablation (NORM 2026): a PPO policy on this encoder can only learn a
    single context-free action distribution (a bandit over operators), so comparing it
    against the full encoders measures how much of RL's value comes from conditioning
    on state at all versus from simply learning *which* operators to use.
    """

    version = "null-1.0"

    def __init__(self) -> None:
        self._space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

    def space(self) -> spaces.Box:
        return self._space

    def encode(self, result: Optional[Dict[str, Any]], env: Any) -> np.ndarray:
        return np.zeros((1,), dtype=np.float32)


@register_state("features_instance_v1")
class InstanceOnlyStateEncoder(BaseStateEncoder):
    """Static instance descriptors only, no search dynamics (NORM 2026 state ablation).

    Constant within a restart, so the policy can condition its operator choice on the
    instance but cannot react to search progress. Deliberately excludes the vessel list's
    size: it is only an upper bound on the fleet (solutions use far fewer vessels), so the
    fleet requirement is described by weekly deck demand in full loads of the smallest /
    largest available vessel instead. Scaled to the small/medium instance ranges.
    """

    version = "instance-2.0"
    DIM = 11

    def __init__(self) -> None:
        self._space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.DIM,), dtype=np.float32)

    def space(self) -> spaces.Box:
        return self._space

    def encode(self, result: Optional[Dict[str, Any]], env: Any) -> np.ndarray:
        stats: Dict[str, float] = getattr(env, "current_instance_stats", {}) or {}
        installations = float(stats.get("num_installations", 0.0))
        visits = float(stats.get("total_visits", 0.0))
        demand = float(stats.get("total_deck_demand", 0.0))
        min_deck = float(stats.get("min_deck_capacity", 0.0))
        max_deck = float(stats.get("max_deck_capacity", 0.0))
        values = [
            installations / 20.0,
            visits / 50.0,
            demand / 1300.0,
            (demand / min_deck if min_deck > 0 else 0.0) / 15.0,   # weekly loads, smallest vessel
            (demand / max_deck if max_deck > 0 else 0.0) / 15.0,   # weekly loads, largest vessel
            float(stats.get("avg_distance_km", 0.0)) / 150.0,
            float(stats.get("max_distance_km", 0.0)) / 200.0,
            (visits / installations if installations > 0 else 0.0) / 4.0,
            (demand / visits if visits > 0 else 0.0) / 50.0,
            float(stats.get("tw_installation_share", 0.0)),
            float(stats.get("tw_visit_share", 0.0)),
        ]
        assert len(values) == self.DIM, len(values)
        return np.nan_to_num(np.clip(np.array(values, dtype=np.float32), 0.0, 2.0), nan=0.0)


# ---------------------------------------------------------------------------
# NORM 2026 state-design ablation: three disjoint, normalised feature groups
# (instance / search process / solution) plus their concatenation.
# ---------------------------------------------------------------------------

_DESTROY_FAMILIES = ("shaw_removal", "random_visit_removal", "worst_visit_removal")
_REPAIR_FAMILIES = ("deep_greedy", "k_regret_2", "k_regret_3", "first_fit")
_IMPROVEMENT_FAMILIES = ("voyage_number_reduction", "fleet_and_cost_reduction", "deep_relocation", "deep_swap")
# Intensity level (0 = light, 0.5 = medium, 1 = full) of each improvement variant used by
# op_triple_v2 / op_triple_many_v1 / improvement_chain_v2.
_IMPROVEMENT_INTENSITY = {
    "voyage_number_reduction_v1": 0.0,
    "voyage_number_reduction_v3": 0.5,
    "voyage_number_reduction_v5": 1.0,
    "fleet_and_cost_reduction_v1": 0.0,
    "fleet_and_cost_reduction_kisialiou_v1": 0.5,
    "fleet_and_cost_reduction_v10": 1.0,
    "deep_relocation_v1": 0.0,
    "deep_relocation_m10_v1": 0.5,
    "deep_relocation_v10": 1.0,
    "deep_swap_v1": 0.0,
    "deep_swap_m10_v1": 0.5,
    "deep_swap_v10": 1.0,
}


def _family_one_hot(name: Optional[str], families: tuple) -> list:
    name = name or ""
    return [1.0 if name.startswith(family) else 0.0 for family in families]


def _horizon(env: Any) -> float:
    return float(max(int(getattr(env, "progress_horizon", None) or getattr(env, "max_iterations", 1)), 1))


@register_state("features_search_v1")
class SearchProcessStateEncoder(BaseStateEncoder):
    """Search-process features: where the search is, how the last step went, which
    operators were just used, and each operator family's recent track record.

    Operators are encoded by family (plus a parameter level), not by action index, so the
    same encoder works for improvement_chain_v2 (engine-chosen destroy/repair are still
    observed), op_triple_v2 and op_triple_many_v1. Per-restart memory (repeat streak,
    acceptance EMAs, family track record) lives on the encoder, which is created per env,
    and is reset whenever the environment resets.
    """

    version = "search-1.0"
    DIM = 34
    EMA = 0.1           # recent acceptance / new-best rate
    TRACK_EMA = 0.2     # per-family improvement rate, updated only when the family is used

    def __init__(self) -> None:
        self._space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.DIM,), dtype=np.float32)
        self._reset_memory()

    def _reset_memory(self) -> None:
        self._initial_temperature: Optional[float] = None
        self._last_action: Optional[int] = None
        self._streak = 0
        self._accept_rate = 0.5
        self._new_best_rate = 0.0
        self._destroy_track = [0.5] * len(_DESTROY_FAMILIES)
        self._improvement_track = [0.5] * len(_IMPROVEMENT_FAMILIES)

    def space(self) -> spaces.Box:
        return self._space

    def encode(self, result: Optional[Dict[str, Any]], env: Any) -> np.ndarray:
        is_step = bool(result) and "raw_action_index" in result
        if not is_step:
            self._reset_memory()
        result = result or {}
        horizon = _horizon(env)

        temperature = float(result.get("temperature", getattr(env, "temperature", 0.0)) or 0.0)
        if self._initial_temperature is None and temperature > 0.0:
            self._initial_temperature = temperature
        temp_ratio = temperature / self._initial_temperature if self._initial_temperature else 1.0

        acceptance_type = str(result.get("acceptance_type", "unknown"))
        accepted = 1.0 if (is_step and result.get("accepted", False)) else 0.0
        improved = 1.0 if acceptance_type == "improvement" else 0.0
        new_best = 1.0 if (is_step and result.get("is_new_best", False)) else 0.0
        acceptance_one_hot = [
            improved,
            1.0 if acceptance_type == "annealing" else 0.0,
            1.0 if is_step and acceptance_type in ("rejected", "rejected_no_change", "infeasible") else 0.0,
        ]

        destroy_name = result.get("destroy_operator_name") if is_step else None
        repair_name = result.get("repair_operator_name") if is_step else None
        improvement_name = result.get("improvement_operator_name") if is_step else None
        destroy_oh = _family_one_hot(destroy_name, _DESTROY_FAMILIES)
        repair_oh = _family_one_hot(repair_name, _REPAIR_FAMILIES)
        improvement_oh = _family_one_hot(improvement_name, _IMPROVEMENT_FAMILIES)
        improvement_oh.append(1.0 if is_step and not any(improvement_oh) else 0.0)  # no improvement this step
        destroy_high_ratio = 1.0 if destroy_name and "_high_" in destroy_name else 0.0
        improvement_intensity = _IMPROVEMENT_INTENSITY.get(improvement_name or "", 0.0)

        if is_step:
            action = int(result.get("raw_action_index", -1))
            self._streak = self._streak + 1 if action == self._last_action else 1
            self._last_action = action
            self._accept_rate += self.EMA * (accepted - self._accept_rate)
            self._new_best_rate += self.EMA * (new_best - self._new_best_rate)
            for i, hit in enumerate(destroy_oh):
                if hit:
                    self._destroy_track[i] += self.TRACK_EMA * (improved - self._destroy_track[i])
            for i, hit in enumerate(improvement_oh[: len(_IMPROVEMENT_FAMILIES)]):
                if hit:
                    self._improvement_track[i] += self.TRACK_EMA * (improved - self._improvement_track[i])

        elapsed_ms = float(result.get("elapsed_ns", 0.0) or 0.0) / 1e6 if is_step else 0.0
        last_reward = float(getattr(env, "last_reward", 0.0)) if is_step else 0.0

        values = (
            [
                float(np.clip(float(getattr(env, "iteration", 0)) / horizon, 0.0, 1.0)),
                float(np.clip(float(getattr(env, "stagnation_count", 0)) / horizon, 0.0, 1.0)),
                float(np.clip(temp_ratio, 0.0, 1.0)),
                accepted,
                *acceptance_one_hot,
                new_best,
                float(np.clip(last_reward / 20.0, -1.0, 1.0)),
                self._accept_rate,
                self._new_best_rate,
            ]
            + destroy_oh
            + repair_oh
            + improvement_oh
            + [
                destroy_high_ratio,
                improvement_intensity,
                float(np.log1p(self._streak) / np.log1p(horizon)),
                float(np.clip(np.log1p(elapsed_ms) / np.log1p(1000.0), 0.0, 1.0)),
            ]
            + self._destroy_track
            + self._improvement_track
        )
        assert len(values) == self.DIM, len(values)
        return np.nan_to_num(np.array(values, dtype=np.float32), nan=0.0, posinf=1.0, neginf=-1.0)


@register_state("features_solution_v1")
class SolutionStateEncoder(BaseStateEncoder):
    """Properties of the current solution: cost relative to the start and to the best
    found, fleet/voyage structure, and vessel load/time utilisation."""

    version = "solution-1.0"
    DIM = 15

    def __init__(self) -> None:
        self._space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.DIM,), dtype=np.float32)
        self._metrics_helper = OperatorFeatureStateEncoder()

    def space(self) -> spaces.Box:
        return self._space

    def encode(self, result: Optional[Dict[str, Any]], env: Any) -> np.ndarray:
        m = self._metrics_helper._resolve_metrics(result, env)
        total = float(m.get("total_cost", 0.0))
        best = float(m.get("best_cost", total))
        initial = float(m.get("initial_cost", getattr(env, "initial_cost", 0.0) or 0.0)) or max(total, 1.0)
        stats: Dict[str, float] = getattr(env, "current_instance_stats", {}) or {}
        visits = max(float(stats.get("total_visits", 0.0)), 1.0)
        fleet = max(float(stats.get("num_vessels", 0.0)), 1.0)
        voyages = float(m.get("num_voyages", 0.0))

        values = [
            total / initial,
            best / initial,
            (total - best) / max(best, 1e-9),
            float(m.get("fixed_cost", 0.0)) / max(total, 1e-9),
            voyages / visits,
            float(m.get("num_empty_voyages", 0.0)) / max(voyages, 1.0),
            float(m.get("num_vessels_used", 0.0)) / fleet,
            float(m.get("avg_voyage_utilization", 0.0)) / 10.0,
            float(m.get("avg_vessel_load_utilization", 0.0)),
            float(m.get("min_vessel_load_utilization", 0.0)),
            float(m.get("max_vessel_load_utilization", 0.0)),
            float(m.get("avg_vessel_time_utilization", 0.0)),
            float(m.get("min_vessel_time_utilization", 0.0)),
            float(m.get("max_vessel_time_utilization", 0.0)),
            1.0 if m.get("is_feasible", False) else 0.0,
        ]
        assert len(values) == self.DIM, len(values)
        return np.nan_to_num(np.clip(np.array(values, dtype=np.float32), -2.0, 2.0), nan=0.0)


@register_state("features_all_v1")
class AllGroupsStateEncoder(BaseStateEncoder):
    """Concatenation of the instance, search-process and solution feature groups."""

    version = "all-1.0"

    def __init__(self) -> None:
        self._parts = [InstanceOnlyStateEncoder(), SearchProcessStateEncoder(), SolutionStateEncoder()]
        dim = sum(int(p.space().shape[0]) for p in self._parts)
        self._space = spaces.Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32)

    def space(self) -> spaces.Box:
        return self._space

    def encode(self, result: Optional[Dict[str, Any]], env: Any) -> np.ndarray:
        return np.concatenate([p.encode(result, env) for p in self._parts]).astype(np.float32)
