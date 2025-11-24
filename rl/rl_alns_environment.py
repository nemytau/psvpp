"""
Reinforcement Learning Environment for ALNS using Gymnasium API.

This module provides a Gymnasium-compatible environment that wraps the Rust ALNS
library, allowing RL agents to learn optimal operator selection strategies.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
        problem_instance_paths: Optional[Sequence[str]] = None,
        problem_sampling_strategy: str = "round_robin",
        action_module: str = DEFAULT_ACTION_KEY,
        state_module: str = DEFAULT_STATE_KEY,
        reward_module: str = DEFAULT_REWARD_KEY,
        enable_operator_logging: bool = True,
        operator_logging_mode: str = "train",
        operator_logging_format: str = "csv",
        operator_logging_dir: Optional[str] = None,
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
            enable_operator_logging: Whether to persist per-iteration operator usage details
            operator_logging_mode: Mode label embedded in output filenames (e.g. "train")
            operator_logging_format: Log file format, either "csv" or "jsonl"
            operator_logging_dir: Target directory for operator usage logs (defaults to logs/)
            force_baseline_improvement: Automatically execute a configured improvement after each destroy/repair pair (baseline mode)
            baseline_improvement_idx: Optional improvement operator index to use when ``force_baseline_improvement`` is enabled
        """
        super().__init__()
        
        # Store parameters
        self.default_problem_instance = problem_instance
        self.problem_paths = self._init_problem_pool(problem_instance, problem_instance_paths)
        self.problem_sampling_strategy = problem_sampling_strategy.lower()
        if self.problem_sampling_strategy not in {"round_robin", "random", "seed"}:
            print(
                f"Warning: Unknown sampling strategy '{problem_sampling_strategy}',"
                " falling back to 'round_robin'"
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
        self.action_module_name = action_module or DEFAULT_ACTION_KEY
        self.state_module_name = state_module or DEFAULT_STATE_KEY
        self.reward_module_name = reward_module or DEFAULT_REWARD_KEY
        self.operator_logging_enabled = bool(enable_operator_logging)
        self.force_baseline_improvement = bool(force_baseline_improvement)
        self.baseline_improvement_idx = baseline_improvement_idx if baseline_improvement_idx is None else int(baseline_improvement_idx)
        
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
            print(
                "Warning: force_baseline_improvement requested but no improvement operators are available; disabling the flag."
            )
            self.force_baseline_improvement = False
            self.baseline_improvement_idx = None
        elif self.baseline_improvement_idx is not None and (
            self.baseline_improvement_idx < 0
            or self.baseline_improvement_idx >= self.num_improvement
        ):
            print(
                "Warning: baseline_improvement_idx out of range; defaulting to the first improvement operator."
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
        self._episode_counter = 0
        self._current_episode_id: Optional[int] = None
        self._latest_policy_entropy: Optional[float] = None
        self._initial_snapshot = None
        self._instance_stats_cache: Dict[str, Dict[str, float]] = {}
        self.current_instance_stats: Dict[str, float] = {}
        
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
        
        print(f"ALNSEnvironment initialized:")
        print(f"  Problem: {self.problem_instance}")
        if len(self.problem_paths) > 1:
            print(f"  Problem pool size: {len(self.problem_paths)}")
            print(f"  Sampling strategy: {self.problem_sampling_strategy}")
        print(f"  Destroy operators: {self.num_destroy}")
        print(f"  Repair operators: {self.num_repair}")
        print(f"  Improvement operators: {self.num_improvement}")
        print(f"  Action space size: {self.action_impl.n()}")
        print(f"  Max iterations: {max_iterations}")

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
                print(f"Warning: Failed to load instance statistics for '{dataset_identifier}': {exc}")
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
            options: Optional reset options
            
        Returns:
            Tuple of (initial_observation, info_dict)
        """
        # Use provided seed or fall back to instance seed
        episode_seed = seed if seed is not None else self.seed
        selected_problem = self._choose_problem_instance(episode_seed, options)
        self.current_problem_instance = selected_problem
        self.problem_instance = selected_problem
        self._latest_policy_entropy = None

        if self.operator_logging_enabled:
            # Flush any buffered logs from previous episode before starting anew
            self.operator_logger.flush()
            self.operator_logger.start_episode()

        self._episode_counter += 1
        self._current_episode_id = self._episode_counter
        self._initial_snapshot = None
        
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

        snapshot_override = options.get("initial_snapshot") if options else None
        shared_snapshot = snapshot_override is not None
        
        # Initialize ALNS with Rust engine
        try:
            init_result = self.alns.initialize_alns(
                problem_instance=self.current_problem_instance,
                seed=episode_seed,
                temperature=self.temperature,
                theta=self.theta,
                weight_update_interval=self.weight_update_interval
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
            if accepted_flag:
                if improved:
                    acceptance_type = "improvement"
                elif worse:
                    acceptance_type = "annealing"
                else:
                    acceptance_type = "rejected_no_change"
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
                log_improvement=improvement_only,
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
                    "final_temperature": result["temperature"]
                }
                if self.operator_logging_enabled:
                    log_path = self.operator_logger.flush()
                    if log_path:
                        info["operator_usage_log"] = str(log_path)
            
            return obs, reward, done, truncated, info
            
        except Exception as e:
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
            
            return obs, reward, done, truncated, info
    
    def _get_observation(self, result: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Delegate observation encoding to the configured StateEncoder."""

        try:
            return self.state_encoder.encode(result, env=self)
        except Exception as exc:  # pragma: no cover - safety net
            print(f"Warning: Failed to extract observation: {exc}")
            shape = getattr(self.observation_space, "shape", (30,))
            return np.zeros(shape, dtype=np.float32)

    def _compute_reward(self, result: Dict[str, Any]) -> float:
        """Delegate reward computation to the configured RewardFn."""

        try:
            return float(self.reward_function.compute(result, env=self))
        except Exception as exc:  # pragma: no cover - safety net
            print(f"Warning: Failed to compute reward: {exc}")
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
            print(f"Iteration {latest['iteration']}: "
                  f"Cost={latest['current_cost']:.2f}, "
                  f"Best={latest['best_cost']:.2f}, "
                  f"Accepted={latest['accepted']}, "
                  f"Temp={latest['temperature']:.1f}")
        return None
    
    def close(self) -> None:
        """Clean up environment resources."""
        if self.operator_logging_enabled:
            self.operator_logger.flush()
    
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
        )

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
    ) -> None:
        if not self.operator_logging_enabled or self._current_episode_id is None:
            self._latest_policy_entropy = None
            return

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
            }
            self.operator_logger.append(destroy_record)

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
            }
            self.operator_logger.append(repair_record)

        if log_improvement and improvement_idx_result is not None:
            improvement_name = result.get("improvement_operator_name")
            if (
                improvement_name is None
                and 0 <= improvement_idx_result < self.num_improvement
            ):
                improvement_name = self.improvement_operators[improvement_idx_result]

            improvement_record = {
                **base_record,
                "operator_name": improvement_name or "none",
                "operator_type": result.get("improvement_operator_type", "improvement"),
                "operator_index": improvement_idx_result,
                "num_removed_requests": None,
                "num_inserted_requests": None,
            }
            self.operator_logger.append(improvement_record)

        self._latest_policy_entropy = None


def test_environment():
    """Test the ALNS environment manually."""
    print("Testing ALNSEnvironment...")
    
    # Create environment
    env = ALNSEnvironment(
        problem_instance="SMALL_1",
        max_iterations=10,
        seed=42
    )
    
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Test reset
    obs, info = env.reset()
    print(f"\nInitial observation shape: {obs.shape}")
    print(f"Initial info: {info}")
    
    # Test some steps
    total_reward = 0.0
    for step_num in range(10):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        destroy_idx, repair_idx, improvement_idx = env._encode_action(action)
        step_info = info.get("step_info", {})
        
        print(
            "Step {step}: action={action_id} (destroy={d}, repair={r}, improvement={impr}), "
            "reward={reward:.2f}, cost={cost:.2f}, accepted={accepted}".format(
                step=step_num + 1,
                action_id=action,
                d=destroy_idx,
                r=repair_idx,
                impr=improvement_idx if improvement_idx is not None else "none",
                reward=reward,
                cost=step_info.get("current_cost", float("nan")),
                accepted=step_info.get("accepted", "N/A"),
            )
        )
        
        if done:
            print("Episode finished!")
            break
    
    print(f"\nTotal reward: {total_reward:.2f}")
    
    # Get episode statistics
    stats = env.get_episode_statistics()
    print(f"Episode stats: {stats}")
    
    env.close()
    print("Environment test completed!")


if __name__ == "__main__":
    test_environment()