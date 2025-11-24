# API Documentation

This document provides comprehensive API documentation for the PSVPP project components.

## 📚 Table of Contents

1. [RL Environment API](#-rl-environment-api)
2. [PyO3 Rust Interface](#-pyo3-rust-interface)
3. [ALNS Components](#-alns-components)
4. [Configuration System](#-configuration-system)
5. [Data Structures](#-data-structures)
6. [Utilities](#-utilities)

## 🤖 RL Environment API

### ALNSRLEnvironment

**Location**: `src/rl_integration/environment.py`

The main Gymnasium-compatible RL environment for training RL agents on ALNS operator selection.

#### Constructor

```python
ALNSRLEnvironment(
    rust_alns_interface,
    max_iterations: int = 1000,
    problem_instance: str = "SMALL_1"
)
```

**Parameters:**
- `rust_alns_interface`: PyO3 interface to Rust ALNS implementation
- `max_iterations`: Maximum number of ALNS iterations per episode
- `problem_instance`: Problem instance identifier ("SMALL_1", "SMALL_2", etc.)

#### Properties

```python
# Gymnasium standard properties
action_space: gym.spaces.Discrete        # |destroy| × |repair| × (|improvement| + 1) combinations
observation_space: gym.spaces.Box        # 18-dimensional continuous state space

# Environment-specific properties  
destroy_operators: List[str]             # e.g. ["shaw_removal", "random_removal", "worst_removal"]
repair_operators: List[str]              # e.g. ["deep_greedy", "k_regret_2", "k_regret_3"]
improvement_operators: List[str]         # e.g. ["voyage_number_reduction"]
current_iteration: int                   # Current ALNS iteration
best_cost: float                         # Best solution cost found so far
```

#### Methods

##### reset()
```python
def reset(
    seed: Optional[int] = None, 
    options: Optional[dict] = None
) -> Tuple[np.ndarray, Dict]
```

Reset environment for new episode.

**Returns:**
- `observation`: 18-dimensional numpy array representing initial state
- `info`: Dictionary with initial solution information

**Example:**
```python
env = ALNSRLEnvironment(rust_interface, max_iterations=500)
obs, info = env.reset(seed=42)
print(f"Initial cost: {info['initial_cost']}")
```

##### step()
```python
def step(action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]
```

Execute one ALNS iteration with selected operator combination.

**Parameters:**
- `action`: Integer 0-8 representing operator combination

**Returns:**
- `observation`: New state after applying operators
- `reward`: Reward for this action
- `terminated`: Whether episode is finished (max iterations reached)
- `truncated`: Whether episode was truncated early
- `info`: Detailed information about the step

**Action Encoding:** Actions span all destroy/repair pairs and optionally apply an improvement
operator. The policy can select "no improvement" by choosing the dedicated slot (index -1):

```python
destroy_idx, repair_idx, improvement_idx = env._encode_action(action)

if improvement_idx is None:
    # Skip improvement for this iteration
else:
    improvement_name = env.improvement_operators[improvement_idx]
```

**Example:**
```python
obs, reward, terminated, truncated, info = env.step(action=4)
print(f"Reward: {reward}, Cost: {info['current_cost']}")
```

##### render()
```python
def render(mode: str = "human")
```

Display current environment state.

### SolutionMetrics

**Location**: `src/rl_integration/environment.py`

Data class representing solution state for RL observation.

#### Properties

```python
@dataclass
class SolutionMetrics:
    # Primary metrics
    total_cost: float                           # Current solution cost
    cost_normalized: float                      # Cost normalized by initial solution
    
    # Feasibility metrics  
    is_complete: bool                          # Solution visits all installations
    is_feasible: bool                          # Solution satisfies all constraints
    feasibility_violations: int               # Number of constraint violations
    
    # Structure metrics
    num_voyages: int                           # Total number of voyages
    num_empty_voyages: int                     # Number of empty voyages
    num_vessels_used: int                      # Number of vessels in use
    avg_voyage_utilization: float              # Average visits per voyage
    avg_vessel_load_utilization: float         # Mean load ratio across active vessels
    max_vessel_load_utilization: float         # Peak load ratio across active vessels
    min_vessel_load_utilization: float         # Lowest load ratio across active vessels
    avg_vessel_time_utilization: float         # Mean sailing/service share of period
    max_vessel_time_utilization: float         # Peak time-in-use share across vessels
    min_vessel_time_utilization: float         # Lowest time-in-use share across vessels
    
    # Performance metrics
    cost_improvement_ratio: float              # Improvement ratio vs best
    stagnation_count: int                      # Iterations since last improvement
    
    # Search metrics
    iteration: int                             # Current iteration
    iteration_normalized: float                # Normalized iteration progress
    temperature: float                         # Current simulated annealing temperature
    temperature_normalized: float              # Normalized temperature
    
    # Operator performance
    destroy_operator_success_rates: List[float]  # Success rate per destroy op
    repair_operator_success_rates: List[float]   # Success rate per repair op
    recent_operator_rewards: List[float]         # Recent rewards history
```

#### Methods

##### to_feature_vector()
```python
def to_feature_vector() -> np.ndarray
```

Convert metrics to a normalized feature vector for RL agent, including vessel utilization signals.

**Returns:**
- Numpy array with normalized features ready for RL consumption

### RewardFunction

**Location**: `src/rl_integration/environment.py`

Configurable reward function for multi-objective optimization.

#### Constructor

```python
RewardFunction(
    cost_weight: float = 1.0,
    feasibility_weight: float = 0.5, 
    exploration_weight: float = 0.2,
    convergence_weight: float = 0.1
)
```

#### Methods

##### calculate_reward()
```python
def calculate_reward(
    prev_metrics: SolutionMetrics,
    current_metrics: SolutionMetrics,
    action_accepted: bool,
    is_new_best: bool,
    is_better_than_current: bool
) -> float
```

Calculate comprehensive reward based on solution improvement.

**Returns:**
- Float reward value (can be positive or negative)

## 🦀 PyO3 Rust Interface

### rust_alns_py Module

**Location**: Built from `src/rust_alns/`

PyO3 bindings providing Python access to high-performance Rust ALNS implementation.

#### Functions

##### initialize_alns()
```python
def initialize_alns(
    problem_instance: str,
    seed: int = 42
) -> Dict[str, Any]
```

Initialize ALNS solver with problem instance.

**Parameters:**
- `problem_instance`: Instance identifier ("SMALL_1", "SMALL_2", etc.)
- `seed`: Random seed for reproducibility

**Returns:**
- Dictionary with initial solution metrics

##### execute_iteration()
```python
def execute_iteration(
    destroy_operator_idx: int,
    repair_operator_idx: int,
    iteration: int
) -> Dict[str, Any]
```

Execute one ALNS iteration with specified operators.

**Parameters:**
- `destroy_operator_idx`: Index of destroy operator (0-2)
- `repair_operator_idx`: Index of repair operator (0-2)  
- `iteration`: Current iteration number

**Returns:**
- Dictionary with iteration results and solution metrics

##### get_current_solution()
```python
def get_current_solution() -> Dict[str, Any]
```

Get current solution in detailed format.

**Returns:**
- Dictionary with complete solution representation

## 🧠 ALNS Components

### Python ALNS (src/py_alns/)

#### ALNSAlgorithm

**Location**: `src/py_alns/alns.py`

Pure Python implementation of ALNS algorithm.

```python
class ALNSAlgorithm:
    def __init__(self, 
                 problem_instance: str,
                 max_iterations: int = 1000,
                 initial_temperature: float = 1000.0,
                 cooling_rate: float = 0.995):
        pass
    
    def solve(self) -> Solution:
        """Run complete ALNS algorithm"""
        pass
    
    def step(self, destroy_op: str, repair_op: str) -> IterationResult:
        """Execute single iteration"""
        pass
```

#### Operators

**Location**: `src/py_alns/operators/`

##### Destroy Operators
```python
def shaw_removal(solution: Solution, num_remove: int) -> List[Visit]:
    """Remove related visits based on similarity"""
    pass

def random_removal(solution: Solution, num_remove: int) -> List[Visit]:
    """Remove random visits"""
    pass

def worst_removal(solution: Solution, num_remove: int) -> List[Visit]:
    """Remove visits with highest cost impact"""
    pass
```

##### Repair Operators
```python
def deep_greedy(partial_solution: Solution, removed_visits: List[Visit]) -> Solution:
    """Greedy insertion with deep evaluation"""
    pass

def k_regret_2(partial_solution: Solution, removed_visits: List[Visit]) -> Solution:
    """2-regret insertion heuristic"""
    pass

def k_regret_3(partial_solution: Solution, removed_visits: List[Visit]) -> Solution:
    """3-regret insertion heuristic"""
    pass
```

## ⚙️ Configuration System

### ConfigManager

**Location**: `config/config_utils.py`

Central configuration management system.

```python
class ConfigManager:
    def __init__(self, config_file: str = "config/settings.ini"):
        pass
    
    def get_alns_params(self) -> Dict[str, Any]:
        """Get ALNS algorithm parameters"""
        pass
    
    def get_rl_params(self) -> Dict[str, Any]:
        """Get RL training parameters"""
        pass
    
    def get_instance_params(self, instance: str) -> Dict[str, Any]:
        """Get problem instance parameters"""
        pass
```

### Configuration File Format

**Location**: `config/settings.ini`

```ini
[ALNS]
max_iterations = 1000
initial_temperature = 1000.0
cooling_rate = 0.995
destroy_rate = 0.3

[RL]
learning_rate = 3e-4
batch_size = 64
n_steps = 2048
total_timesteps = 100000

[INSTANCE_SMALL_1]
vessels = 2
installations = 5
planning_horizon = 7
```

## 📊 Data Structures

### Core Classes

#### Solution
```python
@dataclass
class Solution:
    voyages: List[Voyage]
    vessels: List[Vessel]
    total_cost: float
    is_feasible: bool
    violations: List[str]
```

#### Voyage
```python
@dataclass
class Voyage:
    vessel_id: int
    visits: List[Visit]
    departure_time: float
    arrival_time: float
    total_cost: float
```

#### Visit
```python
@dataclass  
class Visit:
    installation_id: int
    arrival_time: float
    departure_time: float
    demand_pickup: Dict[str, float]
    demand_delivery: Dict[str, float]
```

#### Vessel
```python
@dataclass
class Vessel:
    id: int
    capacity: Dict[str, float]
    base_location: int
    operating_cost: float
    available_from: float
    available_until: float
```

## 🛠️ Utilities

### Distance Manager

**Location**: `src/py_alns/utils/distance_manager.py`

```python
class DistanceManager:
    def __init__(self, distance_matrix: np.ndarray):
        pass
    
    def get_distance(self, from_id: int, to_id: int) -> float:
        """Get distance between two locations"""
        pass
    
    def get_travel_time(self, from_id: int, to_id: int, vessel_speed: float) -> float:
        """Calculate travel time between locations"""
        pass
```

### Solution Validator

**Location**: `src/py_alns/utils/validator.py`

```python
class SolutionValidator:
    def validate(self, solution: Solution) -> ValidationResult:
        """Comprehensive solution validation"""
        pass
    
    def check_capacity_constraints(self, voyage: Voyage) -> List[str]:
        """Check vessel capacity constraints"""
        pass
    
    def check_time_windows(self, solution: Solution) -> List[str]:
        """Check time window constraints"""
        pass
```

### Visualization Tools

**Location**: `src/py_alns/visualization/`

```python
def plot_solution(solution: Solution, save_path: str = None):
    """Create solution visualization plot"""
    pass

def plot_convergence(cost_history: List[float], save_path: str = None):
    """Plot algorithm convergence"""
    pass

def create_interactive_dashboard(solution: Solution) -> dash.Dash:
    """Create interactive Dash dashboard"""
    pass
```

## 📋 Usage Examples

### Basic RL Training

```python
from src.rl_integration.environment import ALNSRLEnvironment
from stable_baselines3 import PPO
import rust_alns_py

# Initialize Rust interface
rust_interface = rust_alns_py

# Create environment
env = ALNSRLEnvironment(rust_interface, max_iterations=500, problem_instance="SMALL_1")

# Create and train RL agent
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50000)

# Save trained model
model.save("alns_rl_agent")
```

### Custom Reward Function

```python
from src.rl_integration.environment import RewardFunction, ALNSRLEnvironment

# Create custom reward function
custom_reward = RewardFunction(
    cost_weight=2.0,      # Prioritize cost reduction
    feasibility_weight=1.0,
    exploration_weight=0.1,
    convergence_weight=0.2
)

# Use in environment
env = ALNSRLEnvironment(rust_interface)
env.reward_function = custom_reward
```

### Solution Analysis

```python
from src.py_alns.utils.validator import SolutionValidator
from src.py_alns.visualization import plot_solution

# Validate solution
validator = SolutionValidator()
result = validator.validate(solution)
print(f"Feasible: {result.is_feasible}")
print(f"Violations: {result.violations}")

# Visualize solution
plot_solution(solution, save_path="solution_plot.png")
```

This API documentation provides comprehensive coverage of all major components and their interfaces. Each component is designed to be modular and extensible for research and development purposes.