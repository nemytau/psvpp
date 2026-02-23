"""
Reinforcement Learning Environment for ALNS using Gymnasium API.

This module provides a Gymnasium-compatible environment that wraps the Rust ALNS
library, allowing RL agents to learn optimal operator selection strategies.
"""

import logging
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
import rust_alns_py
from gymnasium import spaces

from rl.instance_stats import get_instance_statistics
from rl.registries import (
    DEFAULT_ACTION_KEY,
    DEFAULT_REWARD_KEY,
    DEFAULT_STATE_KEY,
    create_action_space,
    create_reward_function,
    create_state_encoder,
)
from rl.operator_usage_logger import OperatorUsageLogger


LOGGER = logging.getLogger("psvpp.alns")
LOG_PREFIX = "[ALNS]"


class ALNSEnvironment(gym.Env):
    """
    Gymnasium environment for ALNS operator selection learning.
    
    The environment allows an RL agent to select destroy and repair operators
    at each iteration of the ALNS algorithm. The agent receives observations
    about the current solution state and gets rewards based on solution quality
    improvements and acceptance decisions.
    
    Action Space:
        Discrete space representing combinations of destroy/repair operators with an
        optional improvement operator (including a "no improvement" choice).
        Each action index deterministically maps to (destroy_idx, repair_idx, improvement_idx).
        
    Observation Space:
        Box space with shape (41,) when using the default features_v3 encoder,
        containing solution metrics, operator stats, and instance-level features.
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 4}
    
    def __init__(
        self,
        problem_instance: str = "SMALL_1",
        seed: int = 42,
        max_iterations: int = 100,
        temperature: float = 500.0,
        theta: float = 0.9,
        weight_update_interval: int = 10,
        aggressive_search_factor: float = 0.85,
        num_episodes: int = 1,
        problem_instance_paths: Optional[Sequence[str]] = None,
        problem_sampling_strategy: str = "round_robin",
        action_module: str = DEFAULT_ACTION_KEY,
        state_module: str = DEFAULT_STATE_KEY,
        reward_module: str = DEFAULT_REWARD_KEY,
        algorithm_mode: str = "baseline",
        enable_operator_logging: bool = True,
        operator_logging_mode: str = "train",
        operator_logging_format: str = "csv",
        operator_logging_dir: Optional[str] = None,
        operator_logging_future_window: int = 5,
        force_baseline_improvement: bool = False,
        baseline_improvement_idx: Optional[int] = None,
    ):
        """
        Initialize the ALNS environment.
        
        Args:
            problem_instance: Name of the problem instance to solve
            seed: Random seed for reproducibility
            max_iterations: Maximum number of ALNS iterations per episode
            temperature: Initial simulated annealing temperature
            theta: Temperature cooling factor
            weight_update_interval: Iterations between operator weight updates
            num_episodes: Number of restarts (each with different seed for diverse initial solutions)
            enable_operator_logging: Whether to persist per-iteration operator usage details
            operator_logging_mode: Mode label embedded in output filenames (e.g. "train")
            operator_logging_format: Log file format, either "csv" or "jsonl"
            operator_logging_dir: Target directory for operator usage logs (defaults to logs/)
            operator_logging_future_window: Number of future iterations considered when
                computing delayed best-cost deltas per operator record
            algorithm_mode: High-level ALNS variant to execute (baseline, kisialiou, reinforcement_learning)
            force_baseline_improvement: Automatically execute a configured improvement after each destroy/repair pair (baseline mode)
            baseline_improvement_idx: Optional improvement operator index to use when ``force_baseline_improvement`` is enabled
        """
        super().__init__()
        
        # Store parameters
        self.default_problem_instance = problem_instance
        self.problem_paths = self._init_problem_pool(problem_instance, problem_instance_paths)
        self.problem_sampling_strategy = problem_sampling_strategy.lower()
        if self.problem_sampling_strategy not in {"round_robin", "random", "seed"}:
            LOGGER.warning(
                "%s Unknown sampling strategy '%s'; falling back to 'round_robin'",
                LOG_PREFIX,
                problem_sampling_strategy,
            )
            self.problem_sampling_strategy = "round_robin"
        self._pool_index = 0
        self._problem_rng = np.random.default_rng(seed)
        self.problem_instance = self.problem_paths[0] if self.problem_paths else problem_instance
        self.current_problem_instance = self.problem_instance
        self.seed = seed
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.theta = theta
        self.weight_update_interval = weight_update_interval
        self.aggressive_search_factor = float(aggressive_search_factor)
        self.num_episodes = max(1, int(num_episodes))
        self.current_restart = 0
        self.action_module_name = action_module or DEFAULT_ACTION_KEY
        self.state_module_name = state_module or DEFAULT_STATE_KEY
        self.reward_module_name = reward_module or DEFAULT_REWARD_KEY
        self.operator_logging_enabled = bool(enable_operator_logging)
        self.algorithm_mode = (algorithm_mode or "baseline").strip().lower()
        if self.algorithm_mode not in {"baseline", "kisialiou", "reinforcement_learning", "rl"}:
            LOGGER.warning(
                "%s Unknown algorithm_mode '%s'; defaulting to 'baseline'",
                LOG_PREFIX,
                algorithm_mode,
            )
            self.algorithm_mode = "baseline"
        if self.algorithm_mode == "rl":
            self.algorithm_mode = "reinforcement_learning"
        self.force_baseline_improvement = bool(force_baseline_improvement)
        self.baseline_improvement_idx = baseline_improvement_idx if baseline_improvement_idx is None else int(baseline_improvement_idx)
        
        if self.force_baseline_improvement and self.algorithm_mode != "baseline":
            LOGGER.warning(
                "%s force_baseline_improvement is only compatible with baseline mode; disabling the flag.",
                LOG_PREFIX,
            )
            self.force_baseline_improvement = False
            self.baseline_improvement_idx = None

        # Initialize ALNS interface
        self.alns = rust_alns_py.RustALNSInterface() # type: ignore
        
        # Get operator information
        operator_info = self.alns.get_operator_info()
        self.destroy_operators = operator_info["destroy_operators"]
        self.repair_operators = operator_info["repair_operators"]
        self.improvement_operators = operator_info.get("improvement_operators", [])
        self.num_destroy = len(self.destroy_operators)
        self.num_repair = len(self.repair_operators)
        self.num_improvement = len(self.improvement_operators)

        if self.force_baseline_improvement and self.num_improvement <= 0:
            LOGGER.warning(
                "%s force_baseline_improvement requested but no improvement operators are available; disabling the flag.",
                LOG_PREFIX,
            )
            self.force_baseline_improvement = False
            self.baseline_improvement_idx = None
        elif self.baseline_improvement_idx is not None and (
            self.baseline_improvement_idx < 0
            or self.baseline_improvement_idx >= self.num_improvement
        ):
            LOGGER.warning(
                "%s baseline_improvement_idx out of range; defaulting to the first improvement operator.",
                LOG_PREFIX,
            )
            self.baseline_improvement_idx = 0 if self.num_improvement > 0 else None

        self.action_impl = create_action_space(
            self.action_module_name,
            destroy_operators=self.destroy_operators,
            repair_operators=self.repair_operators,
            improvement_operators=self.improvement_operators,
        )
        self.action_space = self.action_impl.gym_space()

        self.state_encoder = create_state_encoder(self.state_module_name)
        self.observation_space = self.state_encoder.space()
        self.reward_function = create_reward_function(self.reward_module_name)

        self.module_versions = {
            "action": getattr(self.action_impl, "version", "unknown"),
            "state": getattr(self.state_encoder, "version", "unknown"),
            "reward": getattr(self.reward_function, "version", "unknown"),
        }

        logging_dir = Path(operator_logging_dir) if operator_logging_dir else Path("logs")
        self.operator_logger = OperatorUsageLogger(
            mode=operator_logging_mode,
            fmt=operator_logging_format,
            output_dir=logging_dir,
            enabled=self.operator_logging_enabled,
        )
        self.operator_logging_future_window = max(0, int(operator_logging_future_window))
        self._operator_log_pending: Deque[Dict[str, Any]] = deque()
        self._episode_counter = 0
        self._current_episode_id: Optional[int] = None
        self._latest_policy_entropy: Optional[float] = None
        self._initial_snapshot = None
        self._instance_stats_cache: Dict[str, Dict[str, float]] = {}
        self.current_instance_stats: Dict[str, float] = {}
        self._last_prev_current_cost: float = 0.0
        self._last_prev_best_cost: float = 0.0
        self._last_step_current_cost: float = 0.0
        self._last_step_best_cost: float = 0.0
        
        # Initialize state variables
        self.iteration = 0
        self.initial_cost = None
        self.current_episode_best = None
        self.episode_history = []
        self.best_improvement_pct = 0.0
        self.total_improvement_pct = 0.0
        self.best_improvement_abs = 0.0
        self.total_improvement_abs = 0.0
        self.stagnation_count = 0
        self.last_current_cost: Optional[float] = None
        self.last_best_cost: Optional[float] = None
        self.last_reward: float = 0.0
        self.last_operator_id: float = -1.0
        self.last_operator_type_id: float = -1.0
        self.last_improvement_idx: float = -1.0
        self.last_improvement_type_id: float = -1.0
        self.last_fraction_removed: float = 0.0
        self.last_action_type: str = "none"
        self.last_acceptance_type: str = "unknown"
        
        action_space_size = self.action_impl.n()
        LOGGER.info(
            "%s Environment initialized for problem '%s' (destroy=%d, repair=%d, improvement=%d, action_space=%d, max_iterations=%d)",
            LOG_PREFIX,
            self.problem_instance,
            self.num_destroy,
            self.num_repair,
            self.num_improvement,
            action_space_size,
            max_iterations,
        )
        if len(self.problem_paths) > 1:
            LOGGER.info(
                "%s Problem pool configured with %d instances (sampling=%s)",
                LOG_PREFIX,
                len(self.problem_paths),
                self.problem_sampling_strategy,
            )

    def _init_problem_pool(
        self,
        default_instance: Optional[str],
        problem_instance_paths: Optional[Sequence[str]]
    ) -> List[str]:
        pool: List[str] = []
        if problem_instance_paths:
            pool.extend(str(path) for path in problem_instance_paths if path)
        if default_instance:
            if not pool or default_instance not in pool:
                pool.insert(0, default_instance)
        if not pool:
            raise ValueError("At least one problem instance or dataset path must be provided.")
        return pool

    def _resolve_instance_stats(self, dataset_identifier: str) -> Dict[str, float]:
        cache_key = dataset_identifier
        candidate_path = Path(dataset_identifier)
        if candidate_path.exists():
            try:
                cache_key = str(candidate_path.resolve())
            except OSError:
                cache_key = str(candidate_path)

        if cache_key not in self._instance_stats_cache:
            try:
                stats = get_instance_statistics(cache_key)
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.warning(
                    "%s Failed to load instance statistics for '%s': %s",
                    LOG_PREFIX,
                    dataset_identifier,
                    exc,
                )
                stats = {}
            self._instance_stats_cache[cache_key] = stats

        return dict(self._instance_stats_cache.get(cache_key, {}))

    def _choose_problem_instance(
        self,
        episode_seed: Optional[int],
        options: Optional[Dict[str, Any]]
    ) -> str:
        if options and options.get("problem_path"):
            return str(options["problem_path"])
        if not self.problem_paths:
            return self.default_problem_instance

        strategy = self.problem_sampling_strategy
        if strategy == "random":
            idx = int(self._problem_rng.integers(0, len(self.problem_paths)))
        elif strategy == "seed" and episode_seed is not None:
            idx = int(episode_seed % len(self.problem_paths))
        else:  # round robin default
            idx = self._pool_index % len(self.problem_paths)
            self._pool_index += 1

        return self.problem_paths[idx]
    
    def _encode_action(self, action_idx: int) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        """
        Decode action index into destroy, repair, and optional improvement indices.
        
        Args:
            action_idx: Action index from the discrete action space
            
        Returns:
            Tuple of (destroy_operator_idx, repair_operator_idx, improvement_operator_idx)
        """
        destroy_idx, repair_idx, improvement_idx = self.action_impl.id_to_action(action_idx)
        return destroy_idx, repair_idx, improvement_idx

    def _resolve_baseline_improvement_idx(self) -> Optional[int]:
        if self.num_improvement <= 0:
            return None
        if self.baseline_improvement_idx is None:
            return 0
        if 0 <= self.baseline_improvement_idx < self.num_improvement:
            return self.baseline_improvement_idx
        return 0
    
    def reset(
        self, 
        *, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to start a new episode.
        
        Args:
            seed: Optional seed for this episode
            options: Optional reset options (can include 'restart_index' to set restart number)
            
        Returns:
            Tuple of (initial_observation, info_dict)
        """
        # Check if we should increment restart counter
        if options and "restart_index" in options:
            self.current_restart = int(options["restart_index"])
        elif self.current_restart > 0 or self._episode_counter > 0:
            # Only increment if we've had at least one episode
            # (first reset initializes to 0, subsequent ones increment)
            pass  # Keep current restart number if explicitly set, otherwise it gets incremented in step()
        
        # Use provided seed or compute from base seed + restart number
        if seed is not None:
            episode_seed = seed
        else:
            # Derive unique seed for each restart to get different initial solutions
            episode_seed = self.seed + (self.current_restart * 10000)
        
        selected_problem = self._choose_problem_instance(episode_seed, options)
        self.current_problem_instance = selected_problem
        self.problem_instance = selected_problem
        self._latest_policy_entropy = None

        if self.operator_logging_enabled:
            # Flush any buffered logs from previous episode before starting anew
            self._update_operator_log_future_metrics(self.last_best_cost or 0.0, finalize=True)
            self.operator_logger.flush()
            self.operator_logger.start_episode()
        self._operator_log_pending.clear()

        self._episode_counter += 1
        self._current_episode_id = self._episode_counter
        self._initial_snapshot = None

        LOGGER.info(
            "%s Starting ALNS episode %d (restart %d/%d) for problem '%s' (seed=%d)",
            LOG_PREFIX,
            self._current_episode_id,
            self.current_restart + 1,
            self.num_episodes,
            self.current_problem_instance,
            episode_seed,
        )
        
        # Reset episode state
        self.iteration = 0
        self.episode_history = []
        self.best_improvement_pct = 0.0
        self.total_improvement_pct = 0.0
        self.best_improvement_abs = 0.0
        self.total_improvement_abs = 0.0
        self.stagnation_count = 0
        self.last_current_cost = None
        self.last_best_cost = None
        self.last_reward = 0.0
        self.last_operator_id = -1.0
        self.last_operator_type_id = -1.0
        self.last_improvement_idx = -1.0
        self.last_improvement_type_id = -1.0
        self.last_fraction_removed = 0.0
        self._last_step_result = None
        self._last_step_result: Optional[Dict[str, Any]] = None
        self._last_prev_current_cost = 0.0
        self._last_prev_best_cost = 0.0
        self._last_step_current_cost = 0.0
        self._last_step_best_cost = 0.0

        snapshot_override = options.get("initial_snapshot") if options else None
        shared_snapshot = snapshot_override is not None
        
        # Initialize ALNS with Rust engine
        try:
            init_result = self.alns.initialize_alns(
                problem_instance=self.current_problem_instance,
                seed=episode_seed,
                temperature=self.temperature,
                theta=self.theta,
                weight_update_interval=self.weight_update_interval,
                aggressive_search_factor=self.aggressive_search_factor,
                algorithm_mode=self.algorithm_mode,
            )

            if snapshot_override is not None:
                init_result = self.alns.apply_snapshot(snapshot_override)

            self._initial_snapshot = self.alns.create_snapshot()
            
            # Store initial cost for reward computation
            self.initial_cost = init_result["total_cost"]
            self.current_episode_best = self.initial_cost
            self.last_current_cost = float(self.initial_cost)
            self.last_best_cost = float(self.initial_cost)
            self.last_action_type = "none"
            self.last_acceptance_type = "unknown"
            self.current_instance_stats = self._resolve_instance_stats(self.current_problem_instance)
            
            # Log initial solution
            if self.operator_logging_enabled:
                initial_record = {
                    "episode_id": self._current_episode_id,
                    "iteration": 0,
                    "instance_id": self.current_problem_instance,
                    "reward": None,
                    "cost_current": float(self.initial_cost),
                    "cost_best": float(self.initial_cost),
                    "iterations_since_last_best": 0,
                    "policy_entropy": None,
                    "accepted": None,
                    "is_new_best": None,
                    "elapsed_ms": 0.0,
                    "temperature": None,
                    "destroy_idx": None,
                    "repair_idx": None,
                    "improvement_idx": None,
                    "cost_before": float(self.initial_cost),
                    "cost_after": float(self.initial_cost),
                    "cost_delta": 0.0,
                    "best_cost_delta": 0.0,
                    "best_cost_delta_future": None,
                    "lookahead_window": None,
                    "sequence_position": None,
                    "sequence_call_id": None,
                    "operator_name": "initial_solution",
                    "operator_type": "initial",
                    "operator_index": None,
                    "num_removed_requests": None,
                    "num_inserted_requests": None,
                }
                self._queue_operator_log_record(initial_record, self.initial_cost)
            
            # Get initial observation
            obs = self._get_observation(init_result)
            
            info = {
                "initial_cost": self.initial_cost,
                "problem_instance": self.current_problem_instance,
                "episode_seed": episode_seed,
                "shared_initial_snapshot": shared_snapshot,
                "instance_stats": dict(self.current_instance_stats),
            }
            
            return obs, info
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize ALNS: {e}")
    
    def get_initial_snapshot(self):
        if self._initial_snapshot is None:
            raise RuntimeError("No initial snapshot available; reset() must be called first")
        return self._initial_snapshot.duplicate()

    def run_with_restarts(
        self,
        restarts: Optional[int] = None,
        *,
        problem_instance: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run the Rust ALNS engine in full-pass restart mode.

        This is a convenience wrapper around ``RustALNSInterface.run_with_restarts``
        for evaluation/benchmark scenarios where the full search is executed inside
        Rust rather than step-by-step via Gym actions.
        """
        restart_count = int(restarts) if restarts is not None else int(self.num_episodes)
        restart_count = max(1, restart_count)

        selected_problem = problem_instance or self.current_problem_instance or self.problem_instance
        run_seed = int(seed) if seed is not None else int(self.seed)

        self.alns.initialize_alns(
            problem_instance=selected_problem,
            seed=run_seed,
            temperature=self.temperature,
            theta=self.theta,
            weight_update_interval=self.weight_update_interval,
            aggressive_search_factor=self.aggressive_search_factor,
            algorithm_mode=self.algorithm_mode,
        )

        return self.alns.run_with_restarts(restarts=restart_count)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action index representing operator selection
            
        Returns:
            Tuple of (observation, reward, done, truncated, info)
        """
        if self.initial_cost is None:
            raise RuntimeError("Environment must be reset before calling step()")
        
        # Decode action into operator indices
        destroy_idx, repair_idx, improvement_idx = self._encode_action(action)
        improvement_only = (
            improvement_idx is not None
            and destroy_idx is None
            and repair_idx is None
        )

        selected_improvement_idx: Optional[int] = (
            int(improvement_idx) if improvement_idx is not None else None
        )
        baseline_improvement_triggered = False
        action_type = "improvement_only" if improvement_only else "destroy_repair"
        acceptance_type = "unknown"

        iteration_index = self.iteration + 1

        try:
            # Execute ALNS iteration with selected operators
            if improvement_only:
                if selected_improvement_idx is None:
                    raise ValueError("Improvement action must provide an improvement index")
                result = self.alns.execute_improvement_only(
                    iteration=self.iteration,
                    improvement_operator_idx=selected_improvement_idx,
                )
            else:
                if destroy_idx is None or repair_idx is None:
                    raise ValueError(
                        "Destroy and repair indices are required for paired actions"
                    )

                destroy_arg = int(destroy_idx)
                repair_arg = int(repair_idx)

                if (
                    selected_improvement_idx is None
                    and self.force_baseline_improvement
                ):
                    selected_improvement_idx = self._resolve_baseline_improvement_idx()
                    baseline_improvement_triggered = selected_improvement_idx is not None

                result = self.alns.execute_iteration(
                    iteration=self.iteration,
                    destroy_operator_idx=destroy_arg,
                    repair_operator_idx=repair_arg,
                    improvement_operator_idx=selected_improvement_idx,
                    mode="explicit",
                )

            if improvement_only:
                action_type = "improvement_only"
            elif baseline_improvement_triggered:
                action_type = "paired_with_baseline"
            elif selected_improvement_idx is not None:
                action_type = "paired_with_improvement"
            else:
                action_type = "destroy_repair"

            if isinstance(result, dict):
                result["action_type"] = action_type

            feasible_flag = True
            infeasible_reason: Optional[str] = None
            infeasible_flagged = False
            if isinstance(result, dict):
                feasible_raw = result.get("feasible")
                if feasible_raw is not None:
                    feasible_flag = bool(feasible_raw)
                status_value = result.get("status")
                if isinstance(status_value, str) and "infeas" in status_value.lower():
                    feasible_flag = False
                    infeasible_reason = status_value
                reject_reason = result.get("reject_reason")
                if isinstance(reject_reason, str) and "infeas" in reject_reason.lower():
                    feasible_flag = False
                    infeasible_reason = reject_reason
                extra_reason = result.get("infeasible_reason")
                if isinstance(extra_reason, str) and extra_reason:
                    feasible_flag = False
                    infeasible_reason = extra_reason
                if not feasible_flag and infeasible_reason is None:
                    infeasible_reason = "unspecified"

            if not feasible_flag:
                infeasible_flagged = True
                if isinstance(result, dict):
                    result["accepted"] = False

            self.last_action_type = action_type
            self._last_step_result = result

            # Capture previous cost references (before this iteration's updates)
            prev_current_cost = (
                float(self.last_current_cost)
                if self.last_current_cost is not None
                else float(self.initial_cost or result.get("current_cost", 0.0))
            )
            prev_best_cost = (
                float(self.last_best_cost)
                if self.last_best_cost is not None
                else float(self.initial_cost or result.get("best_cost", prev_current_cost))
            )

            current_cost_raw = float(result.get("current_cost", prev_current_cost))
            improved = current_cost_raw < prev_current_cost - 1e-9
            worse = current_cost_raw > prev_current_cost + 1e-9
            accepted_flag = bool(result.get("accepted", False))
            if not feasible_flag:
                acceptance_type = "infeasible"
                if isinstance(result, dict):
                    result["accepted"] = False
                accepted_flag = False
            elif accepted_flag:
                if improved:
                    acceptance_type = "improvement"
                elif worse:
                    acceptance_type = "annealing"
                else:
                    acceptance_type = "rejected_no_change"
                    if isinstance(result, dict):
                        result["accepted"] = False
                    accepted_flag = False
            else:
                acceptance_type = "rejected"
            result["acceptance_type"] = acceptance_type

            # Ensure reward function can access the previous values via env attributes
            if self.last_current_cost is None:
                self.last_current_cost = prev_current_cost
            if self.last_best_cost is None:
                self.last_best_cost = prev_best_cost

            # Compute reward using the pre-update references
            reward = self._compute_reward(result)
            self.last_reward = float(reward)
            self.last_acceptance_type = acceptance_type

            current_cost = float(result.get("current_cost", prev_current_cost))
            best_cost = float(result.get("best_cost", prev_best_cost))
            self.last_current_cost = current_cost
            self.last_best_cost = best_cost
            self.stagnation_count = int(result.get("stagnation_count", self.stagnation_count))
            self._last_prev_current_cost = prev_current_cost
            self._last_prev_best_cost = prev_best_cost
            self._last_step_current_cost = current_cost
            self._last_step_best_cost = best_cost

            # TODO: emit real `fraction_removed` from Rust ALNS and remove 0.0 fallback
            fraction_removed = result.get("fraction_removed")
            if fraction_removed is None:
                fraction_removed = result.get("destroy_removed_requests")
            try:
                self.last_fraction_removed = float(fraction_removed) if fraction_removed is not None else 0.0
            except (TypeError, ValueError):
                self.last_fraction_removed = 0.0

            destroy_idx_result_raw = result.get("destroy_operator_idx")
            repair_idx_result_raw = result.get("repair_operator_idx")

            destroy_idx_result: Optional[int]
            if destroy_idx_result_raw is None:
                destroy_idx_result = destroy_idx if (destroy_idx is not None and destroy_idx >= 0) else None
            else:
                try:
                    destroy_idx_result = int(destroy_idx_result_raw)
                except (TypeError, ValueError):
                    destroy_idx_result = None

            repair_idx_result: Optional[int]
            if repair_idx_result_raw is None:
                repair_idx_result = repair_idx if (repair_idx is not None and repair_idx >= 0) else None
            else:
                try:
                    repair_idx_result = int(repair_idx_result_raw)
                except (TypeError, ValueError):
                    repair_idx_result = None

            repair_operator_type_id_raw = result.get("repair_operator_type_id")
            repair_operator_type_id = -1
            if repair_idx_result is not None and repair_operator_type_id_raw is not None:
                try:
                    repair_operator_type_id = int(repair_operator_type_id_raw)
                except (TypeError, ValueError):
                    repair_operator_type_id = -1

            self.last_operator_id = float(repair_idx_result) if repair_idx_result is not None else -1.0
            self.last_operator_type_id = float(repair_operator_type_id)

            improvement_idx_result_raw = result.get("improvement_operator_idx")
            if improvement_idx_result_raw is None and selected_improvement_idx is not None:
                improvement_idx_result_raw = selected_improvement_idx

            improvement_idx_result: Optional[int]
            if improvement_idx_result_raw is None:
                improvement_idx_result = None
            else:
                try:
                    improvement_idx_result = int(improvement_idx_result_raw)
                except (TypeError, ValueError):
                    improvement_idx_result = None

            improvement_type_id_raw = result.get("improvement_operator_type_id")
            improvement_type_id = None
            if improvement_type_id_raw is not None:
                try:
                    improvement_type_id = int(improvement_type_id_raw)
                except (TypeError, ValueError):
                    improvement_type_id = None

            self.last_improvement_idx = (
                float(improvement_idx_result) if improvement_idx_result is not None else -1.0
            )
            self.last_improvement_type_id = (
                float(improvement_type_id)
                if improvement_type_id is not None
                else -1.0
            )

            improvement_name = result.get("improvement_operator_name")
            if improvement_name is None and improvement_idx_result is not None:
                if 0 <= improvement_idx_result < self.num_improvement:
                    improvement_name = self.improvement_operators[improvement_idx_result]

            # Attach optional fields for downstream consumers
            if isinstance(result, dict):
                result.setdefault("fraction_removed", self.last_fraction_removed)
                result.setdefault("improvement_operator_idx", improvement_idx_result)
                result.setdefault("improvement_operator_name", improvement_name)
                result.setdefault("action_type", action_type)

            if isinstance(result, dict):
                result["improvement_only_action"] = improvement_only
                result["raw_action_index"] = action

            # Extract metrics and compute next observation
            obs = self._get_observation(result)
            
            # Track episode progress
            self.iteration += 1
            current_cost = result["current_cost"]
            best_cost = result["best_cost"]
            
            # Update episode best
            if best_cost < self.current_episode_best:
                self.current_episode_best = best_cost

            # Update improvement tracking (cumulative delta only)
            if self.initial_cost and self.initial_cost > 1e-9:
                improvement_abs = max(0.0, self.initial_cost - best_cost)
                improvement_pct = (improvement_abs / self.initial_cost) * 100.0

                delta_pct = improvement_pct - self.best_improvement_pct
                delta_abs = improvement_abs - self.best_improvement_abs

                if delta_pct > 1e-6:
                    self.total_improvement_pct += delta_pct
                    self.best_improvement_pct = improvement_pct

                if delta_abs > 1e-9:
                    self.total_improvement_abs += delta_abs
                    self.best_improvement_abs = improvement_abs
            
            # Store step information

            destroy_name: Optional[str] = None
            if destroy_idx_result is not None and 0 <= destroy_idx_result < self.num_destroy:
                destroy_name = self.destroy_operators[destroy_idx_result]

            repair_name: Optional[str] = None
            if repair_idx_result is not None and 0 <= repair_idx_result < self.num_repair:
                repair_name = self.repair_operators[repair_idx_result]

            destroy_display = destroy_name or "none"
            repair_display = repair_name or "none"
            improvement_display = improvement_name or "none"

            if infeasible_flagged:
                LOGGER.error(
                    "%s Iteration %d produced infeasible solution (destroy=%s, repair=%s, improvement=%s); reason=%s",
                    LOG_PREFIX,
                    iteration_index,
                    destroy_display,
                    repair_display,
                    improvement_display,
                    infeasible_reason,
                )

            try:
                current_cost_value = (
                    float(self.last_current_cost)
                    if self.last_current_cost is not None
                    else float(current_cost)
                )
            except (TypeError, ValueError):
                current_cost_value = prev_current_cost

            if acceptance_type == "improvement":
                LOGGER.info(
                    "%s Iteration %d improvement: cost %.4f -> %.4f (destroy=%s, repair=%s, improvement=%s)",
                    LOG_PREFIX,
                    iteration_index,
                    prev_current_cost,
                    current_cost_value,
                    destroy_display,
                    repair_display,
                    improvement_display,
                )
            elif acceptance_type == "annealing":
                LOGGER.info(
                    "%s Iteration %d annealing accepted worse solution: cost %.4f -> %.4f (destroy=%s, repair=%s, improvement=%s)",
                    LOG_PREFIX,
                    iteration_index,
                    prev_current_cost,
                    current_cost_value,
                    destroy_display,
                    repair_display,
                    improvement_display,
                )
            elif worse and acceptance_type == "rejected":
                LOGGER.debug(
                    "%s Iteration %d annealing rejected worse solution: cost %.4f -> %.4f (destroy=%s, repair=%s, improvement=%s)",
                    LOG_PREFIX,
                    iteration_index,
                    prev_current_cost,
                    current_cost_value,
                    destroy_display,
                    repair_display,
                    improvement_display,
                )

            step_info = {
                "iteration": self.iteration,
                "destroy_operator": destroy_name,
                "repair_operator": repair_name,
                "improvement_operator": improvement_name,
                "improvement_operator_idx": improvement_idx_result,
                "improvement_operator_type": result.get("improvement_operator_type"),
                "improvement_operator_type_id": result.get("improvement_operator_type_id"),
                "current_cost": current_cost,
                "best_cost": best_cost,
                "accepted": result.get("accepted", False),
                "temperature": result["temperature"],
                "elapsed_ms": result.get("elapsed_ms", 0),
                "baseline_improvement_triggered": baseline_improvement_triggered,
                "raw_action_index": action,
                "improvement_only_action": improvement_only,
                "action_type": action_type,
                "acceptance_type": acceptance_type,
            }
            self.episode_history.append(step_info)

            # Log per-operator usage for downstream analysis
            self._log_operator_usage(
                result=result,
                reward=reward,
                destroy_idx=destroy_idx_result if destroy_idx_result is not None else -1,
                repair_idx=repair_idx_result if repair_idx_result is not None else -1,
                improvement_idx=improvement_idx_result,
                log_destroy=True,
                log_repair=True,
                log_improvement=(
                    improvement_only
                    or self.algorithm_mode == "kisialiou"
                ),
                prev_current_cost=prev_current_cost,
                prev_best_cost=prev_best_cost,
                current_cost=current_cost,
                current_best_cost=best_cost,
            )
            
            # Check if episode is done
            done = self.iteration >= self.max_iterations
            truncated = False  # We don't truncate episodes early
            
            # Prepare info dict
            info = {
                "step_info": step_info,
                "episode_progress": self.iteration / self.max_iterations,
                "best_improvement_abs": self.best_improvement_abs,
                "best_improvement_pct": self.best_improvement_pct,
                "total_improvement_pct": self.total_improvement_pct,
                "total_improvement": self.total_improvement_abs,
                "operators_used": (
                    destroy_idx_result,
                    repair_idx_result,
                    improvement_idx_result,
                ),
                "improvement_operator_type": result.get("improvement_operator_type"),
                "improvement_operator_type_id": result.get("improvement_operator_type_id"),
                "current_cost": current_cost,
                "best_cost": best_cost,
                "baseline_improvement_triggered": baseline_improvement_triggered,
                "raw_action_index": action,
                "improvement_only_action": improvement_only,
                "action_type": action_type,
                "acceptance_type": acceptance_type,
                "instance_stats": dict(self.current_instance_stats),
            }
            
            if done:
                info["episode_summary"] = {
                    "initial_cost": self.initial_cost,
                    "final_best_cost": best_cost,
                    "best_improvement_abs": self.best_improvement_abs,
                    "best_improvement_pct": self.best_improvement_pct,
                    "total_improvement_pct": self.total_improvement_pct,
                    "total_improvement": self.total_improvement_abs,
                    "iterations_completed": self.iteration,
                    "final_temperature": result["temperature"],
                    "restart_index": self.current_restart,
                    "total_restarts": self.num_episodes,
                }
                # Increment restart counter for next episode
                if self.current_restart < self.num_episodes - 1:
                    self.current_restart += 1
                    info["has_more_restarts"] = True
                else:
                    info["has_more_restarts"] = False
                    
                if self.operator_logging_enabled:
                    self._update_operator_log_future_metrics(best_cost, finalize=True)
                    log_path = self.operator_logger.flush()
                    if log_path:
                        info["operator_usage_log"] = str(log_path)
            
            return obs, reward, done, truncated, info
            
        except Exception as e:
            LOGGER.exception(
                "%s Iteration %d failed: %s",
                LOG_PREFIX,
                iteration_index,
                e,
            )
            # Return a penalty and mark episode as done on error
            obs_shape = getattr(self.observation_space, "shape", (30,)) or (30,)
            obs = np.zeros(obs_shape, dtype=np.float32)
            reward = -10.0  # Large penalty for errors
            done = True
            truncated = False
            
            info = {
                "error": str(e),
                "iteration": self.iteration,
                "operators_used": (destroy_idx, repair_idx),
                "instance_stats": dict(self.current_instance_stats),
            }
            self.last_action_type = "error"
            self.last_acceptance_type = "error"

            self._latest_policy_entropy = None

            if self.operator_logging_enabled:
                final_best = self.last_best_cost if self.last_best_cost is not None else prev_best_cost
                self._update_operator_log_future_metrics(final_best, finalize=True)
            
            return obs, reward, done, truncated, info
    
    def _get_observation(self, result: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Delegate observation encoding to the configured StateEncoder."""

        try:
            return self.state_encoder.encode(result, env=self)
        except Exception as exc:  # pragma: no cover - safety net
            LOGGER.exception("%s Failed to extract observation: %s", LOG_PREFIX, exc)
            shape = getattr(self.observation_space, "shape", (30,))
            return np.zeros(shape, dtype=np.float32)

    def _compute_reward(self, result: Dict[str, Any]) -> float:
        """Delegate reward computation to the configured RewardFn."""

        try:
            return float(self.reward_function.compute(result, env=self))
        except Exception as exc:  # pragma: no cover - safety net
            LOGGER.exception("%s Failed to compute reward: %s", LOG_PREFIX, exc)
            return 0.0
    
    def render(self, mode: str = "human") -> Optional[Any]:
        """
        Render the environment state.
        
        Args:
            mode: Rendering mode
            
        Returns:
            Rendered output (implementation specific)
        """
        if mode == "human" and self.episode_history:
            latest = self.episode_history[-1]
            LOGGER.info(
                "%s Render iteration %d: cost=%.2f best=%.2f accepted=%s temp=%.1f",
                LOG_PREFIX,
                latest["iteration"],
                float(latest.get("current_cost", 0.0)),
                float(latest.get("best_cost", 0.0)),
                latest.get("accepted"),
                float(latest.get("temperature", 0.0)),
            )
        return None
    
    def close(self) -> None:
        """Clean up environment resources."""
        if self.operator_logging_enabled:
            final_best = self.last_best_cost if self.last_best_cost is not None else 0.0
            self._update_operator_log_future_metrics(final_best, finalize=True)
            log_path = self.operator_logger.flush()
            if log_path:
                LOGGER.debug("%s Operator usage log flushed to %s", LOG_PREFIX, log_path)
        LOGGER.debug("%s Environment closed", LOG_PREFIX)
    
    def get_episode_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics for the current episode.
        
        Returns:
            Dictionary with episode statistics
        """
        if not self.episode_history:
            return {}
        
        costs = [step["current_cost"] for step in self.episode_history]
        best_costs = [step["best_cost"] for step in self.episode_history]
        acceptances = [step["accepted"] for step in self.episode_history]

        elapsed_ms_steps: List[int] = []
        elapsed_ms_cumulative: List[int] = []
        cumulative_ms = 0
        for step in self.episode_history:
            ms_raw = step.get("elapsed_ms", 0)
            try:
                ms_value = int(ms_raw)
            except (TypeError, ValueError):
                ms_value = 0
            elapsed_ms_steps.append(ms_value)
            cumulative_ms += ms_value
            elapsed_ms_cumulative.append(cumulative_ms)

        elapsed_seconds = [ms / 1000.0 for ms in elapsed_ms_cumulative]
        
        return {
            "total_iterations": len(self.episode_history),
            "initial_cost": self.initial_cost,
            "final_cost": costs[-1] if costs else self.initial_cost,
            "best_cost": min(best_costs) if best_costs else self.initial_cost,
            "best_improvement_abs": self.best_improvement_abs,
            "best_improvement_pct": self.best_improvement_pct,
            "total_improvement_pct": self.total_improvement_pct,
            "total_improvement": self.total_improvement_abs,
            "acceptance_rate": np.mean(acceptances) if acceptances else 0.0,
            "cost_history": costs,
            "best_cost_history": best_costs,
            "problem_instance": self.current_problem_instance,
            "elapsed_ms_steps": elapsed_ms_steps,
            "elapsed_ms_cumulative": elapsed_ms_cumulative,
            "elapsed_seconds": elapsed_seconds,
        }

    def set_policy_statistics(self, *, entropy: Optional[float] = None) -> None:
        """Optionally record policy diagnostics for the next logged step."""
        self._latest_policy_entropy = entropy

    def log_improvement_usage(self) -> None:
        """Record an improvement-only operator entry for baseline runs."""
        if not self.operator_logging_enabled or self._current_episode_id is None:
            return
        if self._last_step_result is None:
            return

        result = self._last_step_result
        improvement_idx_raw = result.get("improvement_operator_idx")
        if improvement_idx_raw is None:
            return

        try:
            improvement_idx = int(improvement_idx_raw)
        except (TypeError, ValueError):
            return

        destroy_idx_raw = result.get("destroy_operator_idx")
        repair_idx_raw = result.get("repair_operator_idx")

        destroy_idx = -1
        if destroy_idx_raw is not None:
            try:
                destroy_idx = int(destroy_idx_raw)
            except (TypeError, ValueError):
                destroy_idx = -1

        repair_idx = -1
        if repair_idx_raw is not None:
            try:
                repair_idx = int(repair_idx_raw)
            except (TypeError, ValueError):
                repair_idx = -1

        self._log_operator_usage(
            result=result,
            reward=self.last_reward,
            destroy_idx=destroy_idx,
            repair_idx=repair_idx,
            improvement_idx=improvement_idx,
            log_destroy=False,
            log_repair=False,
            log_improvement=True,
            prev_current_cost=self._last_prev_current_cost,
            prev_best_cost=self._last_prev_best_cost,
            current_cost=self._last_step_current_cost,
            current_best_cost=self._last_step_best_cost,
        )

    def _update_operator_log_future_metrics(self, latest_best_cost: float, finalize: bool = False) -> None:
        if not self.operator_logging_enabled:
            self._operator_log_pending.clear()
            return
        if not self._operator_log_pending:
            return

        matured: List[Dict[str, Any]] = []
        latest_best = float(latest_best_cost) if latest_best_cost is not None else float("inf")
        for entry in list(self._operator_log_pending):
            entry["best_seen"] = min(entry["best_seen"], latest_best)
            if finalize or self.iteration >= entry["deadline"]:
                record = entry["record"]
                record["best_cost_delta_future"] = entry["best_seen"] - entry["initial_best"]
                matured.append(record)
                self._operator_log_pending.remove(entry)

        for record in matured:
            self.operator_logger.append(record)

    def _queue_operator_log_record(self, record: Dict[str, Any], current_best_cost: float) -> None:
        if not self.operator_logging_enabled:
            return
        if self.operator_logging_future_window <= 0:
            record["best_cost_delta_future"] = 0.0
            self.operator_logger.append(record)
            return

        iteration_index = int(record.get("iteration", 0))
        best_cost_value = record.get("cost_best")
        if best_cost_value is None:
            best_cost_value = current_best_cost
        entry = {
            "record": record,
            "initial_best": float(best_cost_value),
            "best_seen": float(current_best_cost),
            "deadline": iteration_index + self.operator_logging_future_window,
        }
        self._operator_log_pending.append(entry)

    def _log_operator_usage(
        self,
        *,
        result: Dict[str, Any],
        reward: float,
        destroy_idx: int,
        repair_idx: int,
        improvement_idx: Optional[int],
        log_destroy: bool = True,
        log_repair: bool = True,
        log_improvement: bool = False,
        prev_current_cost: float,
        prev_best_cost: float,
        current_cost: float,
        current_best_cost: float,
    ) -> None:
        if not self.operator_logging_enabled or self._current_episode_id is None:
            self._latest_policy_entropy = None
            return

        self._update_operator_log_future_metrics(current_best_cost)

        destroy_idx_result = result.get("destroy_operator_idx")
        if destroy_idx_result is None:
            destroy_idx_result = destroy_idx if destroy_idx >= 0 else None
        else:
            try:
                destroy_idx_result = int(destroy_idx_result)
            except (TypeError, ValueError):
                destroy_idx_result = None

        repair_idx_result = result.get("repair_operator_idx")
        if repair_idx_result is None:
            repair_idx_result = repair_idx if repair_idx >= 0 else None
        else:
            try:
                repair_idx_result = int(repair_idx_result)
            except (TypeError, ValueError):
                repair_idx_result = None

        improvement_idx_result = result.get("improvement_operator_idx")
        if improvement_idx_result is None:
            improvement_idx_result = improvement_idx if improvement_idx is not None else None
        else:
            try:
                improvement_idx_result = int(improvement_idx_result)
            except (TypeError, ValueError):
                improvement_idx_result = None

        cost_delta = float(current_cost) - float(prev_current_cost)
        best_cost_delta = float(current_best_cost) - float(prev_best_cost)

        def _to_float_or_none(value: Any) -> Optional[float]:
            try:
                if value is None:
                    return None
                return float(value)
            except (TypeError, ValueError):
                return None

        cost_before_destroy = _to_float_or_none(result.get("cost_before_destroy"))
        cost_after_destroy = _to_float_or_none(result.get("cost_after_destroy"))
        cost_after_repair = _to_float_or_none(result.get("cost_after_repair"))

        if cost_before_destroy is None:
            cost_before_destroy = float(prev_current_cost)
        if cost_after_destroy is None:
            cost_after_destroy = float(prev_current_cost)
        if cost_after_repair is None:
            cost_after_repair = float(current_cost)

        running_best_cost_for_steps = float(prev_best_cost)

        def _consume_step_best_delta(step_cost_after: Optional[float]) -> Optional[float]:
            nonlocal running_best_cost_for_steps
            if step_cost_after is None:
                return None
            new_best = min(running_best_cost_for_steps, float(step_cost_after))
            delta = new_best - running_best_cost_for_steps
            running_best_cost_for_steps = new_best
            return float(delta)

        base_record = {
            "episode_id": self._current_episode_id,
            "iteration": self.iteration,
            "instance_id": str(self.current_problem_instance),
            "reward": float(reward),
            "cost_current": float(result.get("current_cost", 0.0)),
            "cost_best": float(result.get("best_cost", 0.0)),
            "iterations_since_last_best": int(result.get("stagnation_count", 0)),
            "policy_entropy": self._latest_policy_entropy,
            "accepted": bool(result.get("accepted", False)),
            "is_new_best": bool(result.get("is_new_best", False)),
            "elapsed_ms": int(result.get("elapsed_ms", 0)),
            "temperature": float(result.get("temperature", 0.0)),
            "destroy_idx": destroy_idx_result if destroy_idx_result is not None else -1,
            "repair_idx": repair_idx_result if repair_idx_result is not None else -1,
            "improvement_idx": improvement_idx_result if improvement_idx_result is not None else -1,
            "cost_before": float(prev_current_cost),
            "cost_after": float(current_cost),
            "cost_delta": cost_delta,
            "best_cost_delta": best_cost_delta,
            "best_cost_delta_future": None,
            "lookahead_window": self.operator_logging_future_window,
            "sequence_position": None,
            "sequence_call_id": None,
        }

        if (
            log_destroy
            and destroy_idx_result is not None
            and 0 <= destroy_idx_result < self.num_destroy
        ):
            destroy_name = result.get("destroy_operator_name")
            if destroy_name is None:
                destroy_name = self.destroy_operators[destroy_idx_result]
            destroy_record = {
                **base_record,
                "operator_name": destroy_name,
                "operator_type": result.get("destroy_operator_type", "destroy"),
                "operator_index": destroy_idx_result,
                "num_removed_requests": result.get("destroy_removed_requests"),
                "num_inserted_requests": None,
                "cost_before": float(cost_before_destroy),
                "cost_after": None,
                "cost_delta": None,
                "best_cost_delta": None,
            }
            self._queue_operator_log_record(destroy_record, current_best_cost)

        if (
            log_repair
            and repair_idx_result is not None
            and 0 <= repair_idx_result < self.num_repair
        ):
            repair_name = result.get("repair_operator_name")
            if repair_name is None:
                repair_name = self.repair_operators[repair_idx_result]
            repair_record = {
                **base_record,
                "operator_name": repair_name,
                "operator_type": result.get("repair_operator_type", "repair"),
                "operator_index": repair_idx_result,
                "num_removed_requests": None,
                "num_inserted_requests": result.get("repair_inserted_requests"),
                "cost_before": float(cost_after_destroy),
                "cost_after": float(cost_after_repair),
                "cost_delta": float(cost_after_repair - cost_after_destroy),
                "best_cost_delta": _consume_step_best_delta(float(cost_after_repair)),
            }
            self._queue_operator_log_record(repair_record, current_best_cost)

        if log_improvement:
            sequence_call_prefix = f"{self._current_episode_id}:{self.iteration}"

            improvement_step_metrics_raw = result.get("improvement_step_metrics")
            improvement_step_metrics: List[Dict[str, Any]] = []
            if isinstance(improvement_step_metrics_raw, (list, tuple)):
                for item in improvement_step_metrics_raw:
                    if not isinstance(item, dict):
                        continue
                    try:
                        idx = int(item.get("operator_idx"))
                    except (TypeError, ValueError):
                        continue
                    if not (0 <= idx < self.num_improvement):
                        continue

                    name_raw = item.get("operator_name")
                    operator_name = (
                        str(name_raw)
                        if isinstance(name_raw, str) and name_raw.strip()
                        else self.improvement_operators[idx]
                    )

                    try:
                        seq_pos = int(item.get("sequence_position"))
                    except (TypeError, ValueError):
                        seq_pos = len(improvement_step_metrics)

                    cost_before: Optional[float]
                    cost_after: Optional[float]
                    try:
                        cost_before = float(item.get("cost_before"))
                    except (TypeError, ValueError):
                        cost_before = None
                    try:
                        cost_after = float(item.get("cost_after"))
                    except (TypeError, ValueError):
                        cost_after = None

                    if cost_before is not None and cost_after is not None:
                        individual_cost_delta = cost_after - cost_before
                    else:
                        try:
                            individual_cost_delta = float(item.get("cost_delta"))
                        except (TypeError, ValueError):
                            individual_cost_delta = float(current_cost) - float(prev_current_cost)

                    improvement_step_metrics.append(
                        {
                            "operator_index": idx,
                            "operator_name": operator_name,
                            "sequence_position": seq_pos,
                            "cost_before": cost_before,
                            "cost_after": cost_after,
                            "cost_delta": individual_cost_delta,
                            "best_cost_delta": _consume_step_best_delta(cost_after),
                        }
                    )

            if improvement_step_metrics:
                for step in improvement_step_metrics:
                    seq_pos = int(step["sequence_position"])
                    improvement_record = {
                        **base_record,
                        "operator_name": step["operator_name"],
                        "operator_type": "improvement",
                        "operator_index": int(step["operator_index"]),
                        "improvement_idx": int(step["operator_index"]),
                        "num_removed_requests": None,
                        "num_inserted_requests": None,
                        "cost_before": _to_float_or_none(step.get("cost_before")),
                        "cost_after": _to_float_or_none(step.get("cost_after")),
                        "cost_delta": float(step["cost_delta"]),
                        "best_cost_delta": float(step["best_cost_delta"]),
                        "sequence_position": seq_pos,
                        "sequence_call_id": f"{sequence_call_prefix}:{seq_pos}",
                    }
                    self._queue_operator_log_record(improvement_record, current_best_cost)

                self._latest_policy_entropy = None
                return

            improvement_indices: List[int] = []
            sequence_raw = result.get("improvement_sequence")
            if isinstance(sequence_raw, (list, tuple)):
                for candidate in sequence_raw:
                    try:
                        idx = int(candidate)
                    except (TypeError, ValueError):
                        continue
                    if 0 <= idx < self.num_improvement:
                        improvement_indices.append(idx)

            if not improvement_indices and improvement_idx_result is not None:
                if 0 <= improvement_idx_result < self.num_improvement:
                    improvement_indices.append(improvement_idx_result)

            # Get per-improvement costs from Rust if available
            improvement_costs_raw = result.get("improvement_costs", [])
            improvement_costs: List[float] = []
            if isinstance(improvement_costs_raw, (list, tuple)):
                for cost_val in improvement_costs_raw:
                    try:
                        improvement_costs.append(float(cost_val))
                    except (TypeError, ValueError):
                        pass

            # Cost before improvements (after destroy+repair)
            cost_before_improvements = float(cost_after_repair)

            for seq_pos, idx in enumerate(improvement_indices):
                improvement_name = None
                if 0 <= idx < self.num_improvement:
                    improvement_name = self.improvement_operators[idx]
                if seq_pos == len(improvement_indices) - 1:
                    reported_name = result.get("improvement_operator_name")
                    if isinstance(reported_name, str) and reported_name.strip():
                        improvement_name = reported_name

                # Calculate individual cost delta for this improvement
                if seq_pos < len(improvement_costs):
                    # Use the cost before this improvement and after
                    cost_before = cost_before_improvements if seq_pos == 0 else improvement_costs[seq_pos - 1]
                    cost_after = improvement_costs[seq_pos]
                    individual_cost_delta = cost_after - cost_before
                    individual_best_cost_delta = _consume_step_best_delta(cost_after)
                else:
                    # Fallback: use the combined cost_delta (old behavior)
                    cost_before = None
                    cost_after = None
                    individual_cost_delta = float(current_cost) - float(prev_current_cost)
                    individual_best_cost_delta = None

                improvement_record = {
                    **base_record,
                    "operator_name": improvement_name or f"improvement_{idx}",
                    "operator_type": result.get("improvement_operator_type", "improvement"),
                    "operator_index": idx,
                    "improvement_idx": idx,
                    "num_removed_requests": None,
                    "num_inserted_requests": None,
                    "cost_before": cost_before,
                    "cost_after": cost_after,
                    "cost_delta": individual_cost_delta,
                    "best_cost_delta": individual_best_cost_delta,
                    "sequence_position": seq_pos,
                    "sequence_call_id": f"{sequence_call_prefix}:{seq_pos}",
                }
                self._queue_operator_log_record(improvement_record, current_best_cost)

        self._latest_policy_entropy = None


def test_environment():
    """Test the ALNS environment manually."""
    LOGGER.info("%s Testing ALNSEnvironment...", LOG_PREFIX)
    
    # Create environment
    env = ALNSEnvironment(
        problem_instance="SMALL_1",
        max_iterations=10,
        seed=42
    )
    
    LOGGER.info("%s Action space: %s", LOG_PREFIX, env.action_space)
    LOGGER.info("%s Observation space: %s", LOG_PREFIX, env.observation_space)
    
    # Test reset
    obs, info = env.reset()
    LOGGER.info("%s Initial observation shape: %s", LOG_PREFIX, obs.shape)
    LOGGER.info("%s Initial info: %s", LOG_PREFIX, info)
    
    # Test some steps
    total_reward = 0.0
    for step_num in range(10):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        destroy_idx, repair_idx, improvement_idx = env._encode_action(action)
        step_info = info.get("step_info", {})
        
        LOGGER.info(
            "%s Step %d: action=%s (destroy=%s, repair=%s, improvement=%s) reward=%.2f cost=%.2f accepted=%s",
            LOG_PREFIX,
            step_num + 1,
            action,
            destroy_idx,
            repair_idx,
            improvement_idx if improvement_idx is not None else "none",
            reward,
            float(step_info.get("current_cost", float("nan"))),
            step_info.get("accepted", "N/A"),
        )
        
        if done:
            LOGGER.info("%s Episode finished!", LOG_PREFIX)
            break
    
    LOGGER.info("%s Total reward: %.2f", LOG_PREFIX, total_reward)
    
    # Get episode statistics
    stats = env.get_episode_statistics()
    LOGGER.info("%s Episode stats: %s", LOG_PREFIX, stats)
    
    env.close()
    LOGGER.info("%s Environment test completed!", LOG_PREFIX)


if __name__ == "__main__":
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    test_environment()