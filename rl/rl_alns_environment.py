"""
Reinforcement Learning Environment for ALNS using Gymnasium API.

This module provides a Gymnasium-compatible environment that wraps the Rust ALNS 
library, allowing RL agents to learn optimal operator selection strategies.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
import rust_alns_py
from gymnasium import spaces

from rl.registries import (
    DEFAULT_ACTION_KEY,
    DEFAULT_REWARD_KEY,
    DEFAULT_STATE_KEY,
    create_action_space,
    create_reward_function,
    create_state_encoder,
)


class ALNSEnvironment(gym.Env):
    """
    Gymnasium environment for ALNS operator selection learning.
    
    The environment allows an RL agent to select destroy and repair operators
    at each iteration of the ALNS algorithm. The agent receives observations
    about the current solution state and gets rewards based on solution quality
    improvements and acceptance decisions.
    
    Action Space:
        Discrete space representing combinations of destroy and repair operators.
        action_idx encodes both operator selections as: destroy_idx * num_repair + repair_idx
        
    Observation Space:
        Box space with shape (30,) containing solution metrics like costs,
        temperature, utilization rates, operator success rates, etc.
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
        
        # Initialize ALNS interface
        self.alns = rust_alns_py.RustALNSInterface() # type: ignore
        
        # Get operator information
        operator_info = self.alns.get_operator_info()
        self.destroy_operators = operator_info["destroy_operators"]
        self.repair_operators = operator_info["repair_operators"]
        self.num_destroy = len(self.destroy_operators)
        self.num_repair = len(self.repair_operators)

        self.action_impl = create_action_space(
            self.action_module_name,
            destroy_operators=self.destroy_operators,
            repair_operators=self.repair_operators,
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
        
        # Initialize state variables
        self.iteration = 0
        self.initial_cost = None
        self.current_episode_best = None
        self.episode_history = []
        self.best_improvement_pct = 0.0
        self.total_improvement_pct = 0.0
        self.best_improvement_abs = 0.0
        self.total_improvement_abs = 0.0
        
        print(f"ALNSEnvironment initialized:")
        print(f"  Problem: {self.problem_instance}")
        if len(self.problem_paths) > 1:
            print(f"  Problem pool size: {len(self.problem_paths)}")
            print(f"  Sampling strategy: {self.problem_sampling_strategy}")
        print(f"  Destroy operators: {self.num_destroy}")
        print(f"  Repair operators: {self.num_repair}")
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
    
    def _encode_action(self, action_idx: int) -> Tuple[int, int]:
        """
        Decode action index into destroy and repair operator indices.
        
        Args:
            action_idx: Action index from the discrete action space
            
        Returns:
            Tuple of (destroy_operator_idx, repair_operator_idx)
        """
        destroy_idx, repair_idx = self.action_impl.id_to_action(action_idx)
        return destroy_idx, repair_idx
    
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
        
        # Reset episode state
        self.iteration = 0
        self.episode_history = []
        self.best_improvement_pct = 0.0
        self.total_improvement_pct = 0.0
        self.best_improvement_abs = 0.0
        self.total_improvement_abs = 0.0
        
        # Initialize ALNS with Rust engine
        try:
            init_result = self.alns.initialize_alns(
                problem_instance=self.current_problem_instance,
                seed=episode_seed,
                temperature=self.temperature,
                theta=self.theta,
                weight_update_interval=self.weight_update_interval
            )
            
            # Store initial cost for reward computation
            self.initial_cost = init_result["total_cost"]
            self.current_episode_best = self.initial_cost
            
            # Get initial observation
            obs = self._get_observation(init_result)
            
            info = {
                "initial_cost": self.initial_cost,
                "problem_instance": self.current_problem_instance,
                "episode_seed": episode_seed
            }
            
            return obs, info
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize ALNS: {e}")
    
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
        destroy_idx, repair_idx = self._encode_action(action)
        
        try:
            # Execute ALNS iteration with selected operators
            result = self.alns.execute_iteration(
                iteration=self.iteration,
                destroy_operator_idx=destroy_idx,
                repair_operator_idx=repair_idx,
                mode="explicit"
            )
            
            # Extract metrics and compute reward
            obs = self._get_observation(result)
            reward = self._compute_reward(result)
            
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
            step_info = {
                "iteration": self.iteration,
                "destroy_operator": self.destroy_operators[destroy_idx],
                "repair_operator": self.repair_operators[repair_idx],
                "current_cost": current_cost,
                "best_cost": best_cost,
                "accepted": result["accepted"],
                "temperature": result["temperature"],
                "elapsed_ms": result.get("elapsed_ms", 0)
            }
            self.episode_history.append(step_info)
            
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
                "operators_used": (destroy_idx, repair_idx)
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
            
            return obs, reward, done, truncated, info
            
        except Exception as e:
            # Return a penalty and mark episode as done on error
            obs = np.zeros(30, dtype=np.float32)
            reward = -10.0  # Large penalty for errors
            done = True
            truncated = False
            
            info = {
                "error": str(e),
                "iteration": self.iteration,
                "operators_used": (destroy_idx, repair_idx)
            }
            
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
        # No specific cleanup needed for our environment
        pass
    
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
            "problem_instance": self.current_problem_instance
        }


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
        
        destroy_idx, repair_idx = env._encode_action(action)
        step_info = info.get("step_info", {})
        
        print(f"Step {step_num + 1}: action={action} (destroy={destroy_idx}, repair={repair_idx}), "
              f"reward={reward:.2f}, cost={step_info.get('current_cost', 'N/A'):.2f}, "
              f"accepted={step_info.get('accepted', 'N/A')}")
        
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