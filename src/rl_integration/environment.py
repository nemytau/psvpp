"""
RL-ALNS Environment Design

This file contains the architectural design for integrating Reinforcement Learning
with the ALNS system. The RL agent learns optimal operator selection strategies
instead of using traditional adaptive weight mechanisms.

Architecture Overview:
    Rust ALNS Engine ←→ PyO3 Bindings ←→ Python RL Environment ←→ RL Agent
"""

try:
    import gymnasium as gym  # type: ignore
except ImportError:
    print("❌ gymnasium not installed. Please install with: pip install gymnasium")
    # For now, create a dummy gym module to avoid import errors
    class DummyGym:
        class Env:
            def __init__(self):
                self.action_space = None
                self.observation_space = None
            def reset(self, seed=None, options=None): return np.array([0.0]), {}
            def step(self, action): return np.array([0.0]), 0.0, False, False, {}
            def render(self, mode="human"): pass
        class spaces:
            @staticmethod
            def Discrete(n): return None
            @staticmethod 
            def Box(low, high, shape, dtype): return None
    gym = DummyGym()

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json

# ============================================================================
# STATE REPRESENTATION DESIGN
# ============================================================================

@dataclass
class SolutionMetrics:
    """Core solution metrics that form the RL state representation"""
    
    # Primary objectives
    total_cost: float
    cost_normalized: float  # Normalized by initial solution cost
    
    # Solution quality metrics
    is_complete: bool
    is_feasible: bool
    feasibility_violations: int  # Number of constraint violations
    
    # Solution structure metrics
    num_voyages: int
    num_empty_voyages: int
    num_vessels_used: int
    avg_voyage_utilization: float  # Average visits per voyage
    
    # Solution diversity/exploration metrics
    cost_improvement_ratio: float  # (best_cost - current_cost) / best_cost
    stagnation_count: int  # Iterations since last improvement
    
    # Search progression metrics
    iteration: int
    iteration_normalized: float  # iteration / max_iterations
    temperature: float
    temperature_normalized: float
    
    # Operator performance history (rolling averages)
    destroy_operator_success_rates: List[float]  # Success rate per destroy operator
    repair_operator_success_rates: List[float]   # Success rate per repair operator
    recent_operator_rewards: List[float]         # Last N operator rewards
    
    def to_feature_vector(self) -> np.ndarray:
        """Convert metrics to normalized feature vector for RL agent"""
        features = [
            self.cost_normalized,
            float(self.is_complete),
            float(self.is_feasible),
            self.feasibility_violations / 10.0,  # Normalize violations
            self.num_voyages / 50.0,  # Assume max ~50 voyages
            self.num_empty_voyages / 20.0,
            self.num_vessels_used / 10.0,  # Assume max ~10 vessels
            self.avg_voyage_utilization,
            self.cost_improvement_ratio,
            min(self.stagnation_count / 20.0, 1.0),  # Cap at 20 iterations
            self.iteration_normalized,
            self.temperature_normalized,
        ]
        
        # Add operator success rates (flatten lists)
        features.extend(self.destroy_operator_success_rates)
        features.extend(self.repair_operator_success_rates)
        features.extend(self.recent_operator_rewards[-5:])  # Last 5 rewards
        
        return np.array(features, dtype=np.float32)

# ============================================================================
# ACTION SPACE DESIGN
# ============================================================================

class OperatorSelectionAction:
    """Discrete action space for operator selection"""
    
    def __init__(self, destroy_operators: List[str], repair_operators: List[str]):
        self.destroy_operators = destroy_operators
        self.repair_operators = repair_operators
        
        # Create combined action space: each action is a (destroy_idx, repair_idx) pair
        self.action_combinations = []
        for d_idx, d_name in enumerate(destroy_operators):
            for r_idx, r_name in enumerate(repair_operators):
                self.action_combinations.append((d_idx, r_idx, f"{d_name}+{r_name}"))
        
        self.n_actions = len(self.action_combinations)
    
    def action_to_operators(self, action_id: int) -> Tuple[int, int]:
        """Convert discrete action ID to (destroy_idx, repair_idx)"""
        return self.action_combinations[action_id][:2]
    
    def get_action_description(self, action_id: int) -> str:
        """Get human-readable description of action"""
        return self.action_combinations[action_id][2]

# ============================================================================
# REWARD FUNCTION DESIGN
# ============================================================================

class RewardFunction:
    """Sophisticated reward function balancing multiple objectives"""
    
    def __init__(self, 
                 cost_weight: float = 1.0,
                 feasibility_weight: float = 0.5,
                 exploration_weight: float = 0.2,
                 convergence_weight: float = 0.1):
        self.cost_weight = cost_weight
        self.feasibility_weight = feasibility_weight
        self.exploration_weight = exploration_weight
        self.convergence_weight = convergence_weight
    
    def calculate_reward(self, 
                        prev_metrics: SolutionMetrics,
                        current_metrics: SolutionMetrics,
                        action_accepted: bool,
                        is_new_best: bool,
                        is_better_than_current: bool) -> float:
        """
        Calculate comprehensive reward considering multiple factors:
        
        1. Cost Improvement: Reward for better solutions
        2. Feasibility: Reward for maintaining/achieving feasibility
        3. Exploration: Reward for accepting worse solutions (controlled exploration)
        4. Convergence: Reward for consistent improvements
        """
        
        reward = 0.0
        
        # 1. Cost improvement reward
        if is_new_best:
            cost_improvement = (prev_metrics.total_cost - current_metrics.total_cost) / prev_metrics.total_cost
            reward += self.cost_weight * cost_improvement * 100  # Scale up
        elif is_better_than_current:
            cost_improvement = (prev_metrics.total_cost - current_metrics.total_cost) / prev_metrics.total_cost
            reward += self.cost_weight * cost_improvement * 50  # Smaller reward
        elif action_accepted:
            # Small reward for exploration (accepting worse solutions)
            reward += self.exploration_weight * 5
        
        # 2. Feasibility reward/penalty
        if current_metrics.is_complete and current_metrics.is_feasible:
            reward += self.feasibility_weight * 10
        elif current_metrics.is_feasible:
            reward += self.feasibility_weight * 5
        else:
            reward -= self.feasibility_weight * 20  # Penalty for infeasibility
        
        # 3. Stagnation penalty
        if current_metrics.stagnation_count > 10:
            reward -= self.convergence_weight * current_metrics.stagnation_count
        
        # 4. Progress reward (moving towards end of search)
        if current_metrics.iteration_normalized > 0.8:  # Late in search
            if is_new_best:
                reward += self.convergence_weight * 20  # Bonus for late improvements
        
        return reward

# ============================================================================
# RL ENVIRONMENT CLASS
# ============================================================================

class ALNSRLEnvironment(gym.Env):  # type: ignore
    """
    Gymnasium environment for RL-guided ALNS operator selection
    
    State: SolutionMetrics converted to feature vector
    Action: Discrete selection of (destroy_op, repair_op) combination
    Reward: Multi-objective reward balancing cost, feasibility, exploration
    """
    
    def __init__(self, 
                 rust_alns_interface,  # PyO3 interface to Rust ALNS
                 max_iterations: int = 1000,
                 problem_instance: str = "SMALL_1"):
        
        super().__init__()
        
        self.rust_interface = rust_alns_interface
        self.max_iterations = max_iterations
        self.problem_instance = problem_instance
        
        # Initialize operator lists (these should come from Rust interface)
        self.destroy_operators = ["shaw_removal", "random_removal", "worst_removal"]
        self.repair_operators = ["deep_greedy", "k_regret_2", "k_regret_3"]
        
        # Action space: discrete selection of operator combinations
        self.action_selector = OperatorSelectionAction(self.destroy_operators, self.repair_operators)
        self.action_space = gym.spaces.Discrete(self.action_selector.n_actions)
        
        # State space: will be set after first reset when we know actual size
        # Start with estimated size, will be corrected on first reset
        estimated_state_dim = 12 + len(self.destroy_operators) + len(self.repair_operators) + 5
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(estimated_state_dim,), dtype=np.float32
        )
        
        # Reward function
        self.reward_function = RewardFunction()
        
        # Episode state
        self.current_iteration = 0
        self.current_metrics: Optional[SolutionMetrics] = None
        self.best_cost = float('inf')
        self.initial_cost = None
        self.episode_history = []
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment and start new ALNS episode"""
        
        super().reset(seed=seed)
        
        # Initialize new ALNS run through Rust interface
        initial_solution_data = self.rust_interface.initialize_alns(
            problem_instance=self.problem_instance,
            seed=seed or 42
        )
        
        # Extract initial metrics
        self.current_metrics = self._extract_metrics(initial_solution_data)
        self.initial_cost = self.current_metrics.total_cost
        self.best_cost = self.initial_cost
        self.current_iteration = 0
        self.episode_history = []
        
        obs = self.current_metrics.to_feature_vector()
        
        # Correct observation space if needed (on first reset)
        if self.observation_space is not None and hasattr(self.observation_space, 'shape') and self.observation_space.shape[0] != len(obs):
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(len(obs),), dtype=np.float32
            )
        
        info = {"initial_cost": self.initial_cost}
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one ALNS iteration with RL-selected operators"""
        
        # Convert action to operator indices
        destroy_idx, repair_idx = self.action_selector.action_to_operators(action)
        
        # Execute ALNS iteration through Rust interface
        iteration_result = self.rust_interface.execute_iteration(
            destroy_operator_idx=destroy_idx,
            repair_operator_idx=repair_idx,
            iteration=self.current_iteration
        )
        
        # Extract new solution metrics
        prev_metrics = self.current_metrics
        self.current_metrics = self._extract_metrics(iteration_result)
        
        # Update best cost tracking
        is_new_best = self.current_metrics.total_cost < self.best_cost
        if is_new_best:
            self.best_cost = self.current_metrics.total_cost
        
        is_better_than_current = prev_metrics is not None and self.current_metrics.total_cost < prev_metrics.total_cost
        action_accepted = iteration_result.get("accepted", False)
        
        # Calculate reward (only if prev_metrics exists)
        if prev_metrics is not None:
            reward = self.reward_function.calculate_reward(
                prev_metrics, self.current_metrics, 
                action_accepted, is_new_best, is_better_than_current
            )
        else:
            reward = 0.0  # No reward for first step
        
        # Check termination conditions
        self.current_iteration += 1
        terminated = self.current_iteration >= self.max_iterations
        truncated = False  # Could add early stopping conditions
        
        # Observation for next state
        obs = self.current_metrics.to_feature_vector()
        
        # Info dictionary
        info = {
            "iteration": self.current_iteration,
            "current_cost": self.current_metrics.total_cost,
            "best_cost": self.best_cost,
            "action_accepted": action_accepted,
            "is_new_best": is_new_best,
            "operators_used": self.action_selector.get_action_description(action),
            "feasible": self.current_metrics.is_feasible,
            "complete": self.current_metrics.is_complete
        }
        
        return obs, reward, terminated, truncated, info
    
    def _extract_metrics(self, solution_data: Dict) -> SolutionMetrics:
        """Extract SolutionMetrics from Rust interface response"""
        # Check if this is iteration data (has 'solution_metrics' key) or direct metrics
        if 'solution_metrics' in solution_data:
            metrics = solution_data['solution_metrics']
        else:
            metrics = solution_data
        
        return SolutionMetrics(
            total_cost=metrics["total_cost"],
            cost_normalized=metrics["total_cost"] / self.initial_cost if self.initial_cost else 1.0,
            is_complete=metrics["is_complete"],
            is_feasible=metrics["is_feasible"],
            feasibility_violations=metrics.get("violations", 0),
            num_voyages=metrics["num_voyages"],
            num_empty_voyages=metrics["num_empty_voyages"],
            num_vessels_used=metrics["num_vessels_used"],
            avg_voyage_utilization=metrics["avg_voyage_utilization"],
            cost_improvement_ratio=(self.best_cost - metrics["total_cost"]) / self.best_cost if self.best_cost > 0 else 0.0,
            stagnation_count=metrics.get("stagnation_count", 0),
            iteration=self.current_iteration,
            iteration_normalized=self.current_iteration / self.max_iterations,
            temperature=metrics["temperature"],
            temperature_normalized=metrics["temperature"] / 1000.0,  # Assuming initial temp = 1000
            destroy_operator_success_rates=metrics.get("destroy_success_rates", [0.5] * len(self.destroy_operators)),
            repair_operator_success_rates=metrics.get("repair_success_rates", [0.5] * len(self.repair_operators)),
            recent_operator_rewards=metrics.get("recent_rewards", [0.0] * 5)
        )
    
    def render(self, mode: str = "human"):
        """Render current state (could output solution visualization)"""
        if mode == "human" and self.current_metrics is not None:
            print(f"Iteration {self.current_iteration}: Cost={self.current_metrics.total_cost:.2f}, "
                  f"Best={self.best_cost:.2f}, Feasible={self.current_metrics.is_feasible}")
        
        return None

# ============================================================================
# TRAINING INTEGRATION EXAMPLE
# ============================================================================

def example_training_setup():
    """Example of how the environment would be used with stable-baselines3"""
    
    # This is pseudocode showing the integration pattern
    try:
        from stable_baselines3 import PPO  # type: ignore
        from stable_baselines3.common.vec_env import DummyVecEnv  # type: ignore
    except ImportError:
        print("❌ stable_baselines3 not installed. Please install with: pip install stable-baselines3")
        return
    
    # Initialize Rust interface (to be implemented via PyO3)
    # rust_interface = RustALNSInterface()
    
    # Create environment
    # env = ALNSRLEnvironment(rust_interface, max_iterations=500)
    # env = DummyVecEnv([lambda: env])
    
    # Create RL agent
    # model = PPO("MlpPolicy", env, verbose=1, 
    #           learning_rate=3e-4, n_steps=2048, batch_size=64)
    
    # Train the agent
    # model.learn(total_timesteps=100000)
    
    # Save the trained model
    # model.save("alns_rl_agent")
    
    print("Training integration example - implementation needed")

if __name__ == "__main__":
    print("RL-ALNS Environment Design")
    print("=" * 50)
    print("\nKey Design Decisions:")
    print("1. State: Rich solution metrics + operator performance history")
    print("2. Action: Discrete operator pair selection")
    print("3. Reward: Multi-objective balancing cost, feasibility, exploration")
    print("4. Episodes: Full ALNS runs (500-1000 iterations)")
    print("\nNext Steps:")
    print("- Implement PyO3 bindings for Rust ALNS")
    print("- Test environment with random policy")
    print("- Train RL agent with PPO/A2C")
    print("- Compare RL vs traditional adaptive weights")
    
    example_training_setup()