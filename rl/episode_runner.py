"""Shared baseline/Kisialiou episode-running logic.

Both `scripts/run_experiment_rl_improvement_mvp.py` (RL vs. baseline vs.
Kisialiou comparison) and `scripts/alns_mode_benchmark.py` (standalone
baseline/Kisialiou benchmarking) need to run a baseline or Kisialiou episode
with restart support, operator logging, and cache-compatible output. This
used to be implemented twice, independently, in both scripts -- they silently
drifted apart (one script picked up the fix decoupling Kisialiou's
destroy/repair selection from whatever RL action module it was compared
against, the other didn't, for a long time) until that divergence was found
and fixed. Keeping exactly one implementation here prevents that class of bug
from recurring.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from rl.rl_alns_environment import ALNSEnvironment


@dataclass
class EpisodeSummary:
    initial_cost: float
    best_cost: float
    solver_runtime_seconds: float
    wall_runtime_seconds: float
    # Engine CPU time actually spent searching (constructive solutions + every step). All time
    # budgets use this clock since 2026-10-09; solver/wall times are kept for reference.
    search_cpu_seconds: float = 0.0

    @property
    def improvement_pct(self) -> float:
        if self.initial_cost <= 1e-12:
            return 0.0
        return max(0.0, (self.initial_cost - self.best_cost) / self.initial_cost * 100.0)

    @property
    def runtime_seconds(self) -> float:
        return self.wall_runtime_seconds


def _restart_seed(base_seed: int, restart_idx: int) -> int:
    return int(base_seed + restart_idx * 1009)


def _time_to_best(stats: Dict) -> tuple:
    """Return (solver_seconds_to_best, iterations_to_best) within a single restart.

    Uses the cumulative elapsed_seconds from episode_history. If the episode
    produced no usable data both values are 0.
    """
    best_cost_history = stats.get("best_cost_history") or []
    elapsed_seconds = stats.get("elapsed_seconds") or []
    if not best_cost_history:
        return 0.0, 0
    global_best = min(best_cost_history)
    for i, cost in enumerate(best_cost_history):
        if cost <= global_best + 1e-9:
            t = float(elapsed_seconds[i]) if i < len(elapsed_seconds) else 0.0
            return t, i
    return 0.0, 0


def _time_to_best_search(stats: Dict) -> float:
    """Search CPU seconds within a single restart until its best cost was first reached."""
    best_cost_history = stats.get("best_cost_history") or []
    search_seconds = stats.get("search_cpu_seconds") or []
    if not best_cost_history or not search_seconds:
        return 0.0
    global_best = min(best_cost_history)
    for i, cost in enumerate(best_cost_history):
        if cost <= global_best + 1e-9:
            return float(search_seconds[i]) if i < len(search_seconds) else 0.0
    return 0.0


def _episode_time_to_best_search(restart_records: list) -> float:
    """Cumulative search CPU seconds until the episode's global best was first found."""
    if not restart_records:
        return 0.0
    global_best = min(r["best_cost"] for r in restart_records)
    cumulative = 0.0
    for rec in restart_records:
        if rec["best_cost"] <= global_best + 1e-9:
            return cumulative + float(rec.get("time_to_best_search_seconds", 0.0))
        cumulative += float(rec.get("search_cpu_seconds", 0.0))
    return cumulative


def _budget_reached(spent_seconds: float, target_seconds: Optional[float]) -> bool:
    return target_seconds is not None and spent_seconds >= target_seconds


def _episode_time_to_best(restart_records: list) -> float:
    """Compute cumulative solver seconds until the global best was first found.

    Sums full solver time of restarts that completed before the best-finding
    restart, then adds the time-to-best within that restart.
    """
    if not restart_records:
        return 0.0
    global_best = min(r["best_cost"] for r in restart_records)
    cumulative_solver = 0.0
    for rec in restart_records:
        if rec["best_cost"] <= global_best + 1e-9:
            return cumulative_solver + float(rec.get("time_to_best_solver_seconds", 0.0))
        cumulative_solver += float(rec.get("solver_runtime_seconds", 0.0))
    return cumulative_solver


def _episode_vessel_stats(restart_records: list) -> tuple:
    """Return (initial_num_vessels_used, best_num_vessels_used) for an episode.

    initial is taken from the first restart (representative of the constructive
    heuristic's fleet size before search); best is the vessel count of whichever
    restart achieved the episode's global-best cost.
    """
    if not restart_records:
        return None, None
    initial_vessels = restart_records[0].get("initial_num_vessels_used")
    best_record = min(restart_records, key=lambda r: r["best_cost"])
    return initial_vessels, best_record.get("best_num_vessels_used")


def _episode_structure_stats(restart_records: list) -> Dict[str, Any]:
    """Return initial/best num_voyages and avg_vessel_load_utilization for an episode.

    Same aggregation convention as _episode_vessel_stats: initial from the first
    restart, best from whichever restart hit the episode's global-best cost.
    """
    if not restart_records:
        return {
            "initial_num_voyages": None,
            "best_num_voyages": None,
            "initial_avg_vessel_load_utilization": None,
            "best_avg_vessel_load_utilization": None,
        }
    first = restart_records[0]
    best_record = min(restart_records, key=lambda r: r["best_cost"])
    return {
        "initial_num_voyages": first.get("initial_num_voyages"),
        "best_num_voyages": best_record.get("best_num_voyages"),
        "initial_avg_vessel_load_utilization": first.get("initial_avg_vessel_load_utilization"),
        "best_avg_vessel_load_utilization": best_record.get("best_avg_vessel_load_utilization"),
    }


def _cache_key(dataset_path: str, seed: int) -> str:
    return f"{Path(dataset_path).name}__seed{seed}"


def _episode_summary_to_cache(stats: "EpisodeSummary", restart_rows: list) -> Dict[str, Any]:
    return {
        "initial_cost": stats.initial_cost,
        "best_cost": stats.best_cost,
        "solver_runtime_seconds": stats.solver_runtime_seconds,
        "wall_runtime_seconds": stats.wall_runtime_seconds,
        "search_cpu_seconds": stats.search_cpu_seconds,
        "restart_rows": restart_rows,
    }


def _episode_summary_from_cache(entry: Dict[str, Any]) -> tuple:
    stats = EpisodeSummary(
        initial_cost=entry["initial_cost"],
        best_cost=entry["best_cost"],
        solver_runtime_seconds=entry["solver_runtime_seconds"],
        wall_runtime_seconds=entry["wall_runtime_seconds"],
        search_cpu_seconds=float(entry.get("search_cpu_seconds") or 0.0),
    )
    return stats, entry["restart_rows"]


def run_baseline_episode(
    *,
    dataset_path: str,
    seed: int,
    max_iterations: int,
    log_dir: Path,
    mode_label: str,
    size_label: str,
    initial_snapshots: Optional[List[object]],
    action_module: str,
    state_module: str,
    reward_module: str,
    restart_max_iterations: int,
    target_runtime_seconds: Optional[float] = None,
    restart_limit: Optional[int] = None,
    progress_horizon: Optional[int] = None,
    fixed_pair_idx: Optional[int] = None,
) -> tuple[EpisodeSummary, List[Dict[str, object]]]:
    restart_cap = max(1, min(int(restart_max_iterations), int(max_iterations)))
    effective_restart_limit = max(1, int(restart_limit)) if restart_limit is not None else int(max_iterations)
    solver_elapsed_ms_total = 0.0
    total_solver_runtime_seconds = 0.0
    total_wall_runtime_seconds = 0.0
    total_search_seconds = 0.0
    initial_cost: Optional[float] = None
    best_candidates: List[float] = []
    restart_idx = 0
    restart_records: List[Dict[str, object]] = []

    while restart_idx < effective_restart_limit:
        if restart_idx > 0 and _budget_reached(total_search_seconds, target_runtime_seconds):
            break

        restart_seed = _restart_seed(seed, restart_idx)
        restart_wall_start = time.perf_counter()

        env = ALNSEnvironment(
            problem_instance=dataset_path,
            max_iterations=restart_cap,
            progress_horizon=progress_horizon,
            seed=restart_seed,
            problem_instance_paths=[dataset_path],
            problem_sampling_strategy="round_robin",
            enable_operator_logging=True,
            operator_logging_mode="baseline",
            operator_logging_mode_label=mode_label,
            operator_logging_size=size_label,
            operator_logging_dataset_name=f"{Path(dataset_path).name}__seed{seed}",
            operator_logging_restart_id=restart_idx,
            operator_logging_stable_filename=True,
            operator_logging_dir=str(log_dir),
            action_module=action_module,
            state_module=state_module,
            reward_module=reward_module,
            force_baseline_improvement=False,
            baseline_improvement_idx=None,
            chain_max_consecutive_improvements=None,
            anti_repeat_penalty=0.0,
            algorithm_mode="baseline",
            size_bucket="small" if size_label == "small" else "large",
        )

        reset_options: Dict[str, Any] = {"problem_path": dataset_path}
        if initial_snapshots and restart_idx < len(initial_snapshots):
            reset_options["initial_snapshot"] = initial_snapshots[restart_idx]

        obs, _ = env.reset(seed=restart_seed, options=reset_options)

        done = False
        truncated = False
        steps = 0
        rng = np.random.default_rng(restart_seed)
        restart_elapsed_ms = 0.0

        if not hasattr(env.action_impl, "num_pairs"):
            raise RuntimeError(
                f"Baseline action module '{action_module}' does not expose pair-only actions; "
                "use 'op_pair_v1' for baseline isolation"
            )

        if fixed_pair_idx is not None:
            num_pairs = int(env.action_impl.num_pairs())
            if not (0 <= fixed_pair_idx < num_pairs):
                raise ValueError(f"fixed_pair_idx {fixed_pair_idx} out of range [0, {num_pairs})")

        while not done and not truncated and steps < restart_cap:
            if fixed_pair_idx is not None:
                action = int(fixed_pair_idx)
            else:
                num_pairs = int(env.action_impl.num_pairs())
                action = int(rng.integers(0, num_pairs)) if num_pairs > 0 else int(env.action_space.sample())
            obs, _, done, truncated, info = env.step(action)
            step_info = info.get("step_info", {}) if isinstance(info, dict) else {}
            step_ms = float(step_info.get("operator_elapsed_ms", step_info.get("elapsed_ms", 0.0)) or 0.0)
            restart_elapsed_ms += step_ms
            solver_elapsed_ms_total += step_ms
            steps += 1
            if _budget_reached(total_search_seconds + env.search_cpu_seconds, target_runtime_seconds):
                break

        stats = env.get_episode_statistics()
        restart_search_seconds = env.search_cpu_seconds
        env.close()

        restart_initial = float(stats.get("initial_cost") or 0.0)
        restart_best = float(stats.get("best_cost") or restart_initial)
        restart_initial_vessels = stats.get("initial_num_vessels_used")
        restart_best_vessels = stats.get("best_num_vessels_used")
        restart_initial_voyages = stats.get("initial_num_voyages")
        restart_best_voyages = stats.get("best_num_voyages")
        restart_initial_load_util = stats.get("initial_avg_vessel_load_utilization")
        restart_best_load_util = stats.get("best_avg_vessel_load_utilization")
        if initial_cost is None:
            initial_cost = restart_initial
        best_candidates.append(restart_best)

        elapsed_seq = stats.get("elapsed_seconds") or []
        restart_solver_runtime = float(elapsed_seq[-1]) if elapsed_seq else (restart_elapsed_ms / 1000.0)
        restart_wall_runtime = time.perf_counter() - restart_wall_start
        total_solver_runtime_seconds += restart_solver_runtime
        total_wall_runtime_seconds += restart_wall_runtime
        total_search_seconds += restart_search_seconds
        ttb_solver, ttb_iters = _time_to_best(stats)
        restart_records.append(
            {
                "restart_id": restart_idx,
                "seed": seed,
                "restart_seed": restart_seed,
                "solver_runtime_seconds": restart_solver_runtime,
                "wall_runtime_seconds": restart_wall_runtime,
                "search_cpu_seconds": restart_search_seconds,
                "time_to_best_search_seconds": _time_to_best_search(stats),
                "time_to_best_solver_seconds": ttb_solver,
                "iterations_to_best": ttb_iters,
                "initial_cost": restart_initial,
                "best_cost": restart_best,
                "initial_num_vessels_used": restart_initial_vessels,
                "best_num_vessels_used": restart_best_vessels,
                "initial_num_voyages": restart_initial_voyages,
                "best_num_voyages": restart_best_voyages,
                "initial_avg_vessel_load_utilization": restart_initial_load_util,
                "best_avg_vessel_load_utilization": restart_best_load_util,
                "iterations": int(stats.get("total_iterations", steps)),
                "dataset_path": dataset_path,
                "mode_label": mode_label,
                "size_label": size_label,
                "algorithm_mode": "baseline",
            }
        )

        restart_idx += 1
        if target_runtime_seconds is None:
            break

    if initial_cost is None:
        initial_cost = 0.0
    global_best = min(best_candidates) if best_candidates else 0.0

    return EpisodeSummary(
        initial_cost=float(initial_cost),
        best_cost=float(global_best),
        solver_runtime_seconds=float(total_solver_runtime_seconds),
        wall_runtime_seconds=float(total_wall_runtime_seconds),
        search_cpu_seconds=float(total_search_seconds),
    ), restart_records


def run_policy_and_baseline_episodes(
    *,
    model: Any,
    dataset_path: str,
    seed: int,
    max_iterations: int,
    restart_max_iterations: int,
    log_dir: Path,
    size_label: str,
    action_module: str,
    state_module: str,
    reward_module: str,
    baseline_action_module: str,
    baseline_state_module: str,
    chain_max_consecutive_improvements: Optional[int] = None,
    anti_repeat_penalty: float = 0.0,
    policy_deterministic: bool = True,
    target_runtime_seconds: Optional[float] = None,
    restart_limit: Optional[int] = None,
    progress_horizon: Optional[int] = None,
    decision_cost_seconds: float = 0.0,
) -> tuple[EpisodeSummary, List[Dict[str, object]], EpisodeSummary, List[Dict[str, object]]]:
    """Run RL-policy and baseline restarts interleaved, one restart index at a time.

    Baseline restart i needs to start from RL restart i's *initial* snapshot (same
    starting solution) for a fair per-restart comparison -- see the `initial_snapshot`
    reset option below, which mirrors `run_baseline_episode`'s `initial_snapshots[i]`
    lookup. Running the two restart loops back to back (RL sweep fully first, then
    baseline consumes the snapshots) means every one of those duplicated engine
    snapshots -- up to `effective_restart_limit`, i.e. up to `max_iterations` restarts
    when no fixed restart count is given -- has to stay alive in memory for the whole
    RL sweep. On large instances that's what OOM-killed the 2026-09-06 eval run. Doing
    one RL restart then immediately its matched baseline restart, in the same loop,
    means at most one snapshot is ever live at a time, with no change to either mode's
    own restart budget/target-runtime bookkeeping (each side still stops independently
    based on its own accumulated wall time vs. `target_runtime_seconds`).
    """
    restart_cap = max(1, min(int(restart_max_iterations), int(max_iterations)))
    effective_restart_limit = max(1, int(restart_limit)) if restart_limit is not None else int(max_iterations)

    rl_solver_total = 0.0
    rl_wall_total = 0.0
    rl_search_total = 0.0
    rl_initial_cost: Optional[float] = None
    rl_best_candidates: List[float] = []
    rl_records: List[Dict[str, object]] = []
    rl_done = False

    bl_solver_total = 0.0
    bl_wall_total = 0.0
    bl_search_total = 0.0
    bl_initial_cost: Optional[float] = None
    bl_best_candidates: List[float] = []
    bl_records: List[Dict[str, object]] = []
    bl_done = False

    restart_idx = 0
    while restart_idx < effective_restart_limit and not (rl_done and bl_done):
        restart_seed = _restart_seed(seed, restart_idx)
        snapshot: Optional[object] = None

        if not rl_done:
            if restart_idx > 0 and _budget_reached(rl_search_total, target_runtime_seconds):
                rl_done = True
            else:
                restart_wall_start = time.perf_counter()
                env = ALNSEnvironment(
                    problem_instance=dataset_path,
                    max_iterations=restart_cap,
                    progress_horizon=progress_horizon,
                    seed=restart_seed,
                    problem_instance_paths=[dataset_path],
                    problem_sampling_strategy="round_robin",
                    enable_operator_logging=True,
                    operator_logging_mode="reinforcement_learning",
                    operator_logging_mode_label="rl_improvement_mvp",
                    operator_logging_size=size_label,
                    operator_logging_dataset_name=f"{Path(dataset_path).name}__seed{seed}",
                    operator_logging_restart_id=restart_idx,
                    operator_logging_stable_filename=True,
                    operator_logging_dir=str(log_dir),
                    action_module=action_module,
                    state_module=state_module,
                    reward_module=reward_module,
                    chain_max_consecutive_improvements=chain_max_consecutive_improvements,
                    anti_repeat_penalty=anti_repeat_penalty,
                    algorithm_mode="baseline",
                    size_bucket="small" if size_label == "small" else "large",
                )

                obs, _ = env.reset(seed=restart_seed, options={"problem_path": dataset_path})
                snapshot = env.get_initial_snapshot()

                done = False
                truncated = False
                steps = 0
                restart_elapsed_ms = 0.0
                while not done and not truncated and steps < restart_cap:
                    action_raw, _ = model.predict(obs, deterministic=policy_deterministic)
                    action = int(action_raw.item()) if hasattr(action_raw, "item") else int(action_raw)
                    obs, _, done, truncated, info = env.step(action)
                    step_info = info.get("step_info", {}) if isinstance(info, dict) else {}
                    step_ms = float(step_info.get("operator_elapsed_ms", step_info.get("elapsed_ms", 0.0)) or 0.0)
                    restart_elapsed_ms += step_ms
                    steps += 1
                    if _budget_reached(
                        rl_search_total + env.search_cpu_seconds + steps * decision_cost_seconds,
                        target_runtime_seconds,
                    ):
                        break

                stats = env.get_episode_statistics()
                restart_search_seconds = env.search_cpu_seconds + steps * decision_cost_seconds
                env.close()

                restart_initial = float(stats.get("initial_cost") or 0.0)
                restart_best = float(stats.get("best_cost") or restart_initial)
                if rl_initial_cost is None:
                    rl_initial_cost = restart_initial
                rl_best_candidates.append(restart_best)

                elapsed_seq = stats.get("elapsed_seconds") or []
                restart_solver_runtime = float(elapsed_seq[-1]) if elapsed_seq else (restart_elapsed_ms / 1000.0)
                restart_wall_runtime = time.perf_counter() - restart_wall_start
                rl_solver_total += restart_solver_runtime
                rl_wall_total += restart_wall_runtime
                rl_search_total += restart_search_seconds
                ttb_solver, ttb_iters = _time_to_best(stats)
                rl_records.append(
                    {
                        "restart_id": restart_idx,
                        "seed": seed,
                        "restart_seed": restart_seed,
                        "solver_runtime_seconds": restart_solver_runtime,
                        "wall_runtime_seconds": restart_wall_runtime,
                        "search_cpu_seconds": restart_search_seconds,
                        "time_to_best_search_seconds": _time_to_best_search(stats),
                        "time_to_best_solver_seconds": ttb_solver,
                        "iterations_to_best": ttb_iters,
                        "initial_cost": restart_initial,
                        "best_cost": restart_best,
                        "initial_num_vessels_used": stats.get("initial_num_vessels_used"),
                        "best_num_vessels_used": stats.get("best_num_vessels_used"),
                        "initial_num_voyages": stats.get("initial_num_voyages"),
                        "best_num_voyages": stats.get("best_num_voyages"),
                        "initial_avg_vessel_load_utilization": stats.get("initial_avg_vessel_load_utilization"),
                        "best_avg_vessel_load_utilization": stats.get("best_avg_vessel_load_utilization"),
                        "iterations": int(stats.get("total_iterations", steps)),
                        "dataset_path": dataset_path,
                        "mode_label": "rl_improvement_mvp",
                        "size_label": size_label,
                        "algorithm_mode": "baseline",
                    }
                )

                if target_runtime_seconds is None:
                    rl_done = True

        if not bl_done:
            if restart_idx > 0 and _budget_reached(bl_search_total, target_runtime_seconds):
                bl_done = True
            else:
                restart_wall_start = time.perf_counter()
                env = ALNSEnvironment(
                    problem_instance=dataset_path,
                    max_iterations=restart_cap,
                    progress_horizon=progress_horizon,
                    seed=restart_seed,
                    problem_instance_paths=[dataset_path],
                    problem_sampling_strategy="round_robin",
                    enable_operator_logging=True,
                    operator_logging_mode="baseline",
                    operator_logging_mode_label="baseline",
                    operator_logging_size=size_label,
                    operator_logging_dataset_name=f"{Path(dataset_path).name}__seed{seed}",
                    operator_logging_restart_id=restart_idx,
                    operator_logging_stable_filename=True,
                    operator_logging_dir=str(log_dir),
                    action_module=baseline_action_module,
                    state_module=baseline_state_module,
                    reward_module=reward_module,
                    force_baseline_improvement=False,
                    baseline_improvement_idx=None,
                    chain_max_consecutive_improvements=None,
                    anti_repeat_penalty=0.0,
                    algorithm_mode="baseline",
                    size_bucket="small" if size_label == "small" else "large",
                )

                reset_options: Dict[str, Any] = {"problem_path": dataset_path}
                if snapshot is not None:
                    reset_options["initial_snapshot"] = snapshot
                obs, _ = env.reset(seed=restart_seed, options=reset_options)

                if not hasattr(env.action_impl, "num_pairs"):
                    raise RuntimeError(
                        f"Baseline action module '{baseline_action_module}' does not expose pair-only actions; "
                        "use 'op_pair_v1' for baseline isolation"
                    )

                done = False
                truncated = False
                steps = 0
                rng = np.random.default_rng(restart_seed)
                restart_elapsed_ms = 0.0
                while not done and not truncated and steps < restart_cap:
                    num_pairs = int(env.action_impl.num_pairs())
                    action = int(rng.integers(0, num_pairs)) if num_pairs > 0 else int(env.action_space.sample())
                    obs, _, done, truncated, info = env.step(action)
                    step_info = info.get("step_info", {}) if isinstance(info, dict) else {}
                    step_ms = float(step_info.get("operator_elapsed_ms", step_info.get("elapsed_ms", 0.0)) or 0.0)
                    restart_elapsed_ms += step_ms
                    steps += 1
                    if _budget_reached(bl_search_total + env.search_cpu_seconds, target_runtime_seconds):
                        break

                stats = env.get_episode_statistics()
                restart_search_seconds = env.search_cpu_seconds
                env.close()

                restart_initial = float(stats.get("initial_cost") or 0.0)
                restart_best = float(stats.get("best_cost") or restart_initial)
                if bl_initial_cost is None:
                    bl_initial_cost = restart_initial
                bl_best_candidates.append(restart_best)

                elapsed_seq = stats.get("elapsed_seconds") or []
                restart_solver_runtime = float(elapsed_seq[-1]) if elapsed_seq else (restart_elapsed_ms / 1000.0)
                restart_wall_runtime = time.perf_counter() - restart_wall_start
                bl_solver_total += restart_solver_runtime
                bl_wall_total += restart_wall_runtime
                bl_search_total += restart_search_seconds
                ttb_solver, ttb_iters = _time_to_best(stats)
                bl_records.append(
                    {
                        "restart_id": restart_idx,
                        "seed": seed,
                        "restart_seed": restart_seed,
                        "solver_runtime_seconds": restart_solver_runtime,
                        "wall_runtime_seconds": restart_wall_runtime,
                        "search_cpu_seconds": restart_search_seconds,
                        "time_to_best_search_seconds": _time_to_best_search(stats),
                        "time_to_best_solver_seconds": ttb_solver,
                        "iterations_to_best": ttb_iters,
                        "initial_cost": restart_initial,
                        "best_cost": restart_best,
                        "initial_num_vessels_used": stats.get("initial_num_vessels_used"),
                        "best_num_vessels_used": stats.get("best_num_vessels_used"),
                        "initial_num_voyages": stats.get("initial_num_voyages"),
                        "best_num_voyages": stats.get("best_num_voyages"),
                        "initial_avg_vessel_load_utilization": stats.get("initial_avg_vessel_load_utilization"),
                        "best_avg_vessel_load_utilization": stats.get("best_avg_vessel_load_utilization"),
                        "iterations": int(stats.get("total_iterations", steps)),
                        "dataset_path": dataset_path,
                        "mode_label": "baseline",
                        "size_label": size_label,
                        "algorithm_mode": "baseline",
                    }
                )

                if target_runtime_seconds is None:
                    bl_done = True

        restart_idx += 1

    if rl_initial_cost is None:
        rl_initial_cost = 0.0
    if bl_initial_cost is None:
        bl_initial_cost = 0.0

    rl_summary = EpisodeSummary(
        initial_cost=float(rl_initial_cost),
        best_cost=float(min(rl_best_candidates) if rl_best_candidates else 0.0),
        solver_runtime_seconds=float(rl_solver_total),
        wall_runtime_seconds=float(rl_wall_total),
        search_cpu_seconds=float(rl_search_total),
    )
    bl_summary = EpisodeSummary(
        initial_cost=float(bl_initial_cost),
        best_cost=float(min(bl_best_candidates) if bl_best_candidates else 0.0),
        solver_runtime_seconds=float(bl_solver_total),
        wall_runtime_seconds=float(bl_wall_total),
        search_cpu_seconds=float(bl_search_total),
    )
    return rl_summary, rl_records, bl_summary, bl_records


def run_kisialiou_episode(
    *,
    dataset_path: str,
    seed: int,
    max_iterations: int,
    log_dir: Path,
    mode_label: str,
    size_label: str,
    action_module: str,
    state_module: str,
    reward_module: str,
    restart_max_iterations: int,
    target_runtime_seconds: Optional[float] = None,
    chain_max_consecutive_improvements: Optional[int] = None,
    reference_restarts: Optional[int] = None,
    progress_horizon: Optional[int] = None,
    kisialiou_improvement_sequence: Optional[List[str]] = None,
) -> tuple[EpisodeSummary, List[Dict[str, object]]]:
    """
    Run Kisialiou deterministic ALNS algorithm with operator logging.

     IMPORTANT: We use step-by-step execution (env.reset() + env.step() loop) rather
     than env.run_with_restarts() to ensure operator usage is logged for each iteration.
     The run_with_restarts() method executes entirely in Rust without calling Python's
     step function, which bypasses the operator logging pipeline.

     By using Gym-compatible step-by-step execution with algorithm_mode="kisialiou",
     the Rust backend still implements the deterministic Kisialiou operator selection
     strategy, but each iteration is recorded in the operators logs (CSV files).

     This ensures consistency with other evaluation modes (baseline, RL) which also
     use step-by-step execution and produce detailed operator logs for analysis.

     `action_module`/`state_module` here should always be the *baseline* action/state
     modules (e.g. args.baseline_action_module / args.baseline_state_module), not
     whatever RL model this episode is being compared against -- Kisialiou's own
     destroy/repair selection is handled entirely by the Rust engine's
     algorithm_mode="kisialiou" branch, but the action space still determines which
     concrete operator set index 0 (the dummy action always passed below) decodes to
     if the engine ever fell back to it. Passing an RL-specific action module here
     previously coupled Kisialiou's results to whichever model it was being compared
     against -- see docs/PAPER1_ACTION_SPACE_PLAN.md.
     """
    restart_cap = max(1, min(int(restart_max_iterations), int(max_iterations)))
    solver_elapsed_ms_total = 0.0
    total_solver_runtime_seconds = 0.0
    total_wall_runtime_seconds = 0.0
    total_search_seconds = 0.0
    initial_cost: Optional[float] = None
    best_candidates: List[float] = []
    restart_idx = 0
    restart_limit = max(1, int(reference_restarts)) if reference_restarts is not None else int(max_iterations)
    restart_records: List[Dict[str, object]] = []

    while restart_idx < restart_limit:
        if restart_idx > 0 and _budget_reached(total_search_seconds, target_runtime_seconds):
            break

        restart_seed = _restart_seed(seed, restart_idx)
        restart_wall_start = time.perf_counter()

        env = ALNSEnvironment(
            problem_instance=dataset_path,
            max_iterations=restart_cap,
            progress_horizon=progress_horizon,
            seed=restart_seed,
            problem_instance_paths=[dataset_path],
            problem_sampling_strategy="round_robin",
            enable_operator_logging=True,
            operator_logging_mode="kisialiou",
            operator_logging_mode_label=mode_label,
            operator_logging_size=size_label,
            operator_logging_dataset_name=f"{Path(dataset_path).name}__seed{seed}",
            operator_logging_restart_id=restart_idx,
            operator_logging_stable_filename=True,
            operator_logging_dir=str(log_dir),
            action_module=action_module,
            state_module=state_module,
            reward_module=reward_module,
            algorithm_mode="kisialiou",
            chain_max_consecutive_improvements=chain_max_consecutive_improvements,
            kisialiou_improvement_sequence=kisialiou_improvement_sequence,
            size_bucket="small" if size_label == "small" else "large",
        )

        obs, _ = env.reset(seed=restart_seed, options={"problem_path": dataset_path})

        done = False
        truncated = False
        steps = 0
        restart_elapsed_ms = 0.0

        # Run step-by-step to enable operator logging
        while not done and not truncated and steps < restart_cap:
            # For Kisialiou, let the environment handle action selection via the algorithm_mode
            # We use a dummy action (0) since Kisialiou is deterministic and handled by Rust
            action = 0
            obs, _, done, truncated, info = env.step(action)
            step_info = info.get("step_info", {}) if isinstance(info, dict) else {}
            step_ms = float(step_info.get("operator_elapsed_ms", step_info.get("elapsed_ms", 0.0)) or 0.0)
            restart_elapsed_ms += step_ms
            solver_elapsed_ms_total += step_ms
            steps += 1
            if _budget_reached(total_search_seconds + env.search_cpu_seconds, target_runtime_seconds):
                break

        stats = env.get_episode_statistics()
        restart_search_seconds = env.search_cpu_seconds
        env.close()

        restart_initial = float(stats.get("initial_cost") or 0.0)
        restart_best = float(stats.get("best_cost") or restart_initial)
        restart_initial_vessels = stats.get("initial_num_vessels_used")
        restart_best_vessels = stats.get("best_num_vessels_used")
        restart_initial_voyages = stats.get("initial_num_voyages")
        restart_best_voyages = stats.get("best_num_voyages")
        restart_initial_load_util = stats.get("initial_avg_vessel_load_utilization")
        restart_best_load_util = stats.get("best_avg_vessel_load_utilization")
        if initial_cost is None:
            initial_cost = restart_initial
        best_candidates.append(restart_best)

        elapsed_seq = stats.get("elapsed_seconds") or []
        restart_solver_runtime = float(elapsed_seq[-1]) if elapsed_seq else (restart_elapsed_ms / 1000.0)
        restart_wall_runtime = time.perf_counter() - restart_wall_start
        total_solver_runtime_seconds += restart_solver_runtime
        total_wall_runtime_seconds += restart_wall_runtime
        total_search_seconds += restart_search_seconds
        ttb_solver, ttb_iters = _time_to_best(stats)
        restart_records.append(
            {
                "restart_id": restart_idx,
                "seed": seed,
                "restart_seed": restart_seed,
                "solver_runtime_seconds": restart_solver_runtime,
                "wall_runtime_seconds": restart_wall_runtime,
                "search_cpu_seconds": restart_search_seconds,
                "time_to_best_search_seconds": _time_to_best_search(stats),
                "time_to_best_solver_seconds": ttb_solver,
                "iterations_to_best": ttb_iters,
                "initial_cost": restart_initial,
                "best_cost": restart_best,
                "initial_num_vessels_used": restart_initial_vessels,
                "best_num_vessels_used": restart_best_vessels,
                "initial_num_voyages": restart_initial_voyages,
                "best_num_voyages": restart_best_voyages,
                "initial_avg_vessel_load_utilization": restart_initial_load_util,
                "best_avg_vessel_load_utilization": restart_best_load_util,
                "iterations": int(stats.get("total_iterations", steps)),
                "dataset_path": dataset_path,
                "mode_label": mode_label,
                "size_label": size_label,
                "algorithm_mode": "kisialiou",
            }
        )

        restart_idx += 1
        if target_runtime_seconds is None and reference_restarts is None:
            break

    if initial_cost is None:
        initial_cost = 0.0
    global_best = min(best_candidates) if best_candidates else 0.0

    return EpisodeSummary(
        initial_cost=float(initial_cost),
        best_cost=float(global_best),
        solver_runtime_seconds=float(total_solver_runtime_seconds),
        wall_runtime_seconds=float(total_wall_runtime_seconds),
        search_cpu_seconds=float(total_search_seconds),
    ), restart_records
