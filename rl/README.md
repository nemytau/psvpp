# ALNS Reinforcement Learning Environment

This directory contains a complete implementation of a **Gymnasium-compatible reinforcement learning environment** for the ALNS (Adaptive Large Neighborhood Search) algorithm. The environment allows RL agents to learn optimal operator selection strategies for vehicle routing and scheduling problems.

## Overview

The `ALNSEnvironment` class implements the **Gymnasium `Env` interface** and integrates with the **Rust ALNS engine** through PyO3 bindings. RL agents can learn to select destroy and repair operators dynamically, potentially outperforming traditional fixed or weighted strategies.

### Key Features

- **Gymnasium API Compatibility**: Full compliance with `gymnasium.Env` interface
- **Rust Integration**: High-performance ALNS execution via PyO3 bindings  
- **Flexible Action Space**: Discrete actions encoding operator combinations
- **Rich Observations**: 30-dimensional state vector with solution metrics
- **Shaped Rewards**: Multi-component reward function encouraging improvement
- **Episode Management**: Proper reset/step cycle with statistics tracking
- **Error Handling**: Robust error recovery and state validation
- **Stable-Baselines3 Ready**: Compatible with popular RL training libraries

## File Structure

```
rl/
|-- rl_alns_environment.py   # Main environment implementation
|-- train_alns_rl.py         # PPO training script with evaluation
|-- test_environment.py      # Basic environment testing script
`-- README.md                # This documentation
```

## Installation & Setup

### Prerequisites

1. **Build Rust ALNS Extension**:
   ```bash
   cd rust_alns
   maturin develop --release
   ```

2. **Install Python Dependencies**:
   ```bash
   pip install -r requirements.txt  # From repository root
   ```

### Verify Installation

```bash
cd rl
python test_environment.py
```

This will test:
- Basic imports (Gymnasium, NumPy)
- Rust interface accessibility  
- Environment creation and functionality
- Reset/step mechanics
- Basic validation checks

## Quick Start

### Basic Environment Usage

```python
from rl_alns_environment import ALNSEnvironment

# Create environment
env = ALNSEnvironment(
    problem_instance="SMALL_1",
    max_iterations=100,
    seed=42
)

# Reset and run episode
obs, info = env.reset()
for step in range(100):
    action = env.action_space.sample()  # Random action
    obs, reward, done, truncated, info = env.step(action)
    if done or truncated:
        break

# Get episode statistics
stats = env.get_episode_statistics()
print(f"Total improvement: {stats['total_improvement']:.1f}%")
env.close()
```

### Training with Stable-Baselines3

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Create and wrap environment
env = ALNSEnvironment(problem_instance="SMALL_1", max_iterations=100)
vec_env = DummyVecEnv([lambda: env])

# Train PPO agent
model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=50000)

# Save and evaluate
model.save("ppo_alns_model")
mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=10)
```

### Complete Training Pipeline

```bash
cd rl
python train_alns_rl.py
```

This script:
1. **Tests** the environment manually
2. **Trains** a PPO agent for 50,000 timesteps
3. **Evaluates** the trained model
4. **Compares** performance with random baseline
5. **Saves** results and logs

### Unified CLI Commands

The `rl` package now exposes structured CLI entrypoints that wrap the workflow above:

```bash
# Train, evaluate, and run baselines in a single command
python -m rl.train --config configs/ppo_default.yaml --exp-name ppo_v1 --algorithm-mode kisialiou

# Evaluate a saved model on the test split
python -m rl.test --model runs/ppo_v1/model.zip --include-baseline

# Generate comparison plots vs random baseline
python -m rl.evaluate --model runs/ppo_v1/model.zip --output-dir reports/ppo_v1_eval

# Solve a single processed instance and export metrics
python -m rl.solve --model runs/ppo_v1/model.zip --instance data/processed/alns/small/test/small_test_1_01
```

All commands accept `--config` to override defaults (dataset size, seeds, directories, module keys).

### Algorithm Modes

The Rust engine now supports multiple high-level ALNS execution modes. Every CLI exposes an `--algorithm-mode` flag:

| Mode | Description |
|------|-------------|
| `baseline` (default) | Classic destroy/repair loop with optional single improvement step. |
| `kisialiou` | Sequential destroy, repair, and ordered improvement sequence as proposed by Kisialiou et al. |
| `reinforcement_learning` / `rl` | Delegates improvement selection to the RL policy for fully learned search control. |

Examples:

```bash
# Train RL policy while driving the engine in Kisialiou mode
python -m rl.train --algorithm-mode kisialiou

# Run evaluation / comparison with the learned RL-centric iteration flow
python -m rl.evaluate --model runs/ppo_v1/model.zip --algorithm-mode reinforcement_learning

# Smoke-test the random baseline in each mode
python scripts/alns_baseline_smoke.py --algorithm-mode kisialiou
```

If omitted, the default mode is `baseline`. Passing `rl` is accepted as shorthand for `reinforcement_learning`.

### Experiment Manifests

- Training runs now receive canonical experiment IDs (e.g. `20241017_1432__small__ppo__A-default__S-default__R-default__seed42`).
- Artifacts are written under `runs/<exp_id>/` with subdirectories for tensorboard logs (`tb/`), evaluation outputs, baseline comparisons, and ad-hoc artefacts.
- The root of each run contains `config.yaml` (resolved configuration snapshot) and `manifest.json` capturing git metadata, package versions, dataset hashes, module versions, and evaluation summaries.
- CLI commands (`rl.test`, `rl.evaluate`, `rl.solve`) automatically load the manifest when invoked with `runs/<exp_id>/model.zip` to reuse dataset splits, iteration limits, and seed settings.

## Environment Specification

### Action Space

**Type**: `Discrete(num_destroy x num_repair)`

Actions encode combinations of destroy and repair operators:
- `action_idx = destroy_idx * num_repair + repair_idx`
- Typical size: 9 actions (3 destroy x 3 repair operators)

### Observation Space

**Type**: `Box(shape=(30,), dtype=float32)`

30-dimensional vector containing:

| Component | Size | Description |
|-----------|------|-------------|
| **Solution Quality** | 6 | Costs (absolute & normalized), improvement |
| **Algorithm State** | 4 | Temperature, stagnation, iteration, progress |
| **Solution Structure** | 6 | Voyages, utilization, feasibility flags |
| **Operator Performance** | 6 | Success rates for destroy/repair operators |
| **Recent History** | 8 | Recent rewards, statistics, trends |

### Reward Function

Multi-component reward encouraging solution improvement:

```python
reward = (
    cost_improvement * 10.0 +        # Relative cost reduction
    best_improvement * 15.0 +        # Best solution improvement  
    acceptance_bonus +               # +1.0 if accepted, -0.5 if not
    new_best_bonus * 5.0 +          # +5.0 for new best solution
    progress_bonus * 0.1            # Small progress reward
)
```

Clipped to range `[-20.0, 50.0]` for stability.

### Episode Termination

Episodes terminate when:
- Maximum iterations reached (`max_iterations`)
- Unrecoverable error occurs (returns penalty)

No early truncation for sub-optimal performance.

## Environment Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `problem_instance` | `"SMALL_1"` | Problem dataset identifier |
| `seed` | `42` | Random seed for reproducibility |
| `max_iterations` | `100` | Maximum ALNS iterations per episode |
| `temperature` | `500.0` | Initial simulated annealing temperature |
| `theta` | `0.9` | Temperature cooling factor |
| `weight_update_interval` | `10` | Operator weight update frequency |
| `operator_logging_future_window` | `5` | Iterations to look ahead when computing `best_cost_delta_future` in operator usage logs |

## Evaluation & Analysis

### Training Metrics

The training script tracks:
- **Episode Rewards**: Total reward per episode
- **Solution Quality**: Cost improvements and best solutions found
- **Operator Usage**: Which destroy/repair combinations are selected
   - CSV/JSON entries now include per-step `cost_delta`, `best_cost_delta`, and a delayed `best_cost_delta_future` computed over the configured lookahead window, alongside stagnation counters and elapsed time
- **Convergence**: Learning progress and stability

### Baseline Comparison

Performance is compared against:
- **Random Policy**: Uniform random operator selection
- **Weighted Policy**: Traditional adaptive weight-based selection
- **Fixed Strategies**: Hand-crafted operator sequences

### Expected Results

Well-trained agents typically achieve:
- **5-15% better** solution quality vs random
- **Faster convergence** to good solutions
- **Adaptive behavior** based on problem characteristics
- **Consistent performance** across different instances

## Troubleshooting

### Common Issues

1. **Import Error: `rust_alns_py` not found**
   ```bash
   cd rust_alns
   maturin develop --release
   ```

2. **Action Space Error: Invalid action index**
   - Check action is in range `[0, num_destroy * num_repair)`
   - Verify operator counts match Rust implementation

3. **Observation Shape Mismatch**
   - Environment always returns 30-dimensional observations
   - Check observation space configuration

4. **Training Instability**
   - Reduce learning rate: `learning_rate=1e-4`
   - Increase batch size: `batch_size=128`
   - Tune reward scaling in `_compute_reward()`

### Debug Mode

Enable detailed logging:
```python
env = ALNSEnvironment(...)
# Add logging in environment methods
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

### Potential Improvements

1. **Multi-Objective Rewards**: Balance solution quality, runtime, feasibility
2. **Hierarchical Actions**: Learn operator categories then specific operators
3. **State Representation**: Add graph neural network observations
4. **Transfer Learning**: Train on multiple problem instances
5. **Curriculum Learning**: Progressive difficulty increase
6. **Population-Based Training**: Multiple agent variants

### Advanced Features

- **Custom Operators**: Learn to combine or modify existing operators
- **Meta-Learning**: Quick adaptation to new problem types
- **Interpretability**: Understand learned operator selection strategies
- **Multi-Agent**: Cooperative search with multiple ALNS agents

## References

- **Gymnasium**: [https://gymnasium.farama.org/](https://gymnasium.farama.org/)
- **Stable-Baselines3**: [https://stable-baselines3.readthedocs.io/](https://stable-baselines3.readthedocs.io/)
- **PyO3**: [https://pyo3.rs/](https://pyo3.rs/)
- **ALNS Algorithm**: Pisinger & Ropke (2007) "A general heuristic for vehicle routing problems"

## License

This implementation is part of the PSVPP (Platform Supply Vessel Planning Problem) project. See the main project repository for license details.

---

**Happy Learning!** The environment is designed to be robust, efficient, and easy to extend. Feel free to experiment with different reward functions, observation spaces, and training strategies!