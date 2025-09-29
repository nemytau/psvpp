# Project Structure Guide

This document provides a comprehensive overview of the PSVPP project structure after the reorganization.

## 📁 Root Directory Overview

```
psvpp/
├── 📂 src/                     # Source code modules
├── 📂 tests/                   # Test suite
├── 📂 scripts/                 # Executable scripts
├── 📂 examples/                # Usage examples
├── 📂 docs/                    # Documentation
├── 📂 data/                    # Dataset files
├── 📂 config/                  # Configuration files
├── 📂 output/                  # Generated outputs
├── 📂 logs/                    # Log files
├── 📂 notebooks/               # Jupyter notebooks
├── 📄 README.md                # Main project documentation
├── 📄 pyproject.toml           # Python project configuration
└── 📄 requirements.txt         # Python dependencies
```

## 🔍 Detailed Structure

### 📂 src/ - Source Code
```
src/
├── rl_integration/             # RL-ALNS integration layer
│   ├── __init__.py
│   ├── environment.py          # Core RL environment (Gymnasium)
│   ├── actions.py             # Action space definitions
│   ├── observations.py        # Observation space handling
│   └── rewards.py             # Reward function implementations
├── py_alns/                   # Python ALNS implementation
│   ├── __init__.py
│   ├── alns.py               # Main ALNS algorithm
│   ├── operators/            # Destroy/repair operators
│   ├── utils/                # Utility functions
│   └── visualization/        # Plotting and visualization
└── rust_alns/                # Rust ALNS with PyO3 bindings
    ├── Cargo.toml            # Rust project configuration
    ├── src/                  # Rust source code
    ├── tests/                # Rust unit tests
    └── target/               # Rust build artifacts
```

### 🧪 tests/ - Test Suite
```
tests/
├── __init__.py
├── run_full_test_suite.py     # Main test runner
├── test_pyo3_interface.py     # PyO3 bindings tests
├── test_rl_environment.py     # RL environment validation
├── test_validation.py         # Comprehensive validation
├── test_random_policy.py      # Random policy baseline
├── test_ppo_training.py       # PPO training tests
├── integration/               # Integration tests
│   ├── test_small_instances.py
│   └── test_full_pipeline.py
└── fixtures/                  # Test data and fixtures
    ├── sample_solutions.json
    └── test_instances.pkl
```

### 🚀 scripts/ - Executable Scripts
```
scripts/
├── run_rust_alns.py          # Run Rust ALNS solver
├── run_python_alns.py        # Run Python ALNS solver
├── generate_dataset.py       # Generate problem instances
├── train_rl_agent.py         # Train RL agents
├── evaluate_solutions.py     # Solution evaluation
└── benchmarking/             # Performance benchmarks
    ├── benchmark_rust_vs_python.py
    └── performance_profiling.py
```

### 📚 examples/ - Usage Examples
```
examples/
├── basic_usage/              # Simple usage examples
│   ├── hello_alns.py
│   └── basic_rl_training.py
├── advanced/                 # Advanced techniques
│   ├── custom_operators.py
│   ├── hyperparameter_tuning.py
│   └── multi_instance_training.py
└── tutorials/                # Step-by-step tutorials
    ├── getting_started.md
    ├── rl_integration_guide.md
    └── custom_operators_tutorial.md
```

### 📖 docs/ - Documentation
```
docs/
├── PROJECT_STRUCTURE.md      # This file
├── DEVELOPMENT.md            # Development workflow guide
├── API.md                    # API documentation
├── ALGORITHMS.md             # Algorithm descriptions
├── CONFIGURATION.md          # Configuration guide
└── assets/                   # Documentation assets
    ├── diagrams/
    └── screenshots/
```

## 🗂️ Legacy Files (Moved from Root)

The following files were moved during reorganization:

### Moved to scripts/
- `alns_main.py` → `scripts/run_python_alns.py`
- `rust_main.py` → `scripts/run_rust_alns.py`
- `generate_dataset.py` → `scripts/generate_dataset.py`
- `main.py` → `scripts/main.py`

### Moved to tests/
- `test1.py` → `tests/test_validation.py`
- `coop_case.py` → `tests/integration/test_cooperation.py`

### Moved to notebooks/
- `visualization.ipynb` → `notebooks/solution_visualization.ipynb`
- `rust_solution_visualization.py` → `notebooks/rust_visualization.py`

## 🎯 Key Components

### Core RL Environment
- **Location**: `src/rl_integration/environment.py`
- **Purpose**: Gymnasium-compatible RL environment
- **Key Features**:
  - 9 discrete actions (3×3 destroy/repair combinations)
  - 18-dimensional observation space
  - Integration with Rust ALNS backend

### Test Suite
- **Location**: `tests/`
- **Coverage**: PyO3 bindings, RL environment, integration tests
- **Performance**: ~1,400 steps/second training validation

### Configuration System
- **Location**: `config/`
- **Files**: `settings.ini`, `config_utils.py`
- **Purpose**: Centralized parameter management

## 🚀 Quick Navigation

### To run tests:
```bash
python tests/run_full_test_suite.py
```

### To train an RL agent:
```bash
python scripts/train_rl_agent.py
```

### To run ALNS solver:
```bash
python scripts/run_rust_alns.py --instance SMALL_1
```

### To generate new instances:
```bash
python scripts/generate_dataset.py --size small --count 10
```

## 📊 Data Flow

```
Problem Instance (data/) 
    ↓
ALNS Solver (src/py_alns/ or src/rust_alns/)
    ↓
RL Environment (src/rl_integration/)
    ↓
RL Agent Training (scripts/train_rl_agent.py)
    ↓
Solution Output (output/)
```

## 🔧 Development Workflow

1. **Code Changes**: Edit files in `src/`
2. **Testing**: Run tests from `tests/`
3. **Scripts**: Execute experiments from `scripts/`
4. **Documentation**: Update files in `docs/`
5. **Examples**: Add tutorials to `examples/`

This structure follows Python packaging best practices and provides clear separation of concerns between source code, tests, documentation, and executable scripts.