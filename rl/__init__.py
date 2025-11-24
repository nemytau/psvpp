"""
ALNS Reinforcement Learning Package

This package provides a Gymnasium-compatible environment for training RL agents
on the ALNS (Adaptive Large Neighborhood Search) algorithm for vehicle routing
and scheduling optimization problems.

Main Components:
- ALNSEnvironment: Gymnasium environment for ALNS operator selection
- Training utilities: PPO training with Stable-Baselines3
- Evaluation tools: Performance comparison and analysis

Example usage:
    from rl.rl_alns_environment import ALNSEnvironment
    
    env = ALNSEnvironment(problem_instance="SMALL_1")
    obs, info = env.reset()
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
"""

from . import modules as _modules  # noqa: F401  # Ensure default modules register themselves
from .rl_alns_environment import ALNSEnvironment
from .operator_usage_logger import OperatorUsageLogger

__version__ = "1.0.0"
__author__ = "PSVPP Project"
__all__ = ["ALNSEnvironment", "OperatorUsageLogger"]