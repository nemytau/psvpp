# Supply Vessel Planning with ALNS & Reinforcement Learning

## Overview

This project implements an Adaptive Large Neighborhood Search (ALNS) algorithm for the supply vessel planning problem. It builds upon Kisialiou’s ALNS implementation and extends it with a Reinforcement Learning (RL) layer to enhance adaptability to different problem layouts.

## Features

- **ALNS Implementation**: Core heuristic search for optimizing vessel routing.
- **Reinforcement Learning (RL) Enhancement**: Improves ALNS adaptability using RL.
- **Rust Integration**: Optimized performance for ALNS operators.
- **Configurable Inputs**: Flexible data input format for vessels, installations, and bases.
- **Logging and Visualization**: Tracks execution and performance metrics.

## Project Structure

```
psvpp
├─ alns/               # Python package for ALNS implementation
│  ├─ Beans/           # Core object representations (node, vessel, voyage, etc.)
│  ├─ alns/            # ALNS algorithm and operators
│  ├─ rl/              # Reinforcement Learning models
│  ├─ utils/           # Utility functions (I/O, distance calculations, TSP solver)
│  ├─ data_generator.py # Generates problem instances
├─ config/             # Configuration files (settings, meta-parameters)
├─ data/               # Input dataset (CSV, PKL) for vessels, installations, bases
├─ logs/               # Logs for debugging and tracking performance
├─ rust_alns/          # Rust-based ALNS implementation for performance improvement
│  ├─ src/             # Rust source code (operators, structures, utils)
├─ sample/             # Sample test cases of varying sizes
├─ tests/              # Unit tests for ALNS operators
├─ visualization.ipynb # Jupyter Notebook for result visualization
├─ main.py             # Main script for running ALNS
├─ generate_dataset.py # Script for data generation
└─ rust_main.py        # Python-Rust interface for ALNS execution
```

## Installation

### Requirements

- Python 3.x
- Rust (for performance-critical ALNS operators)
- Required Python packages (see `requirements.txt` if available)

### Setup

```bash
# Clone repository
git clone <repository-url>
cd psvpp

# Install Python dependencies
pip install -r requirements.txt

# Build Rust module (if required)
cd rust_alns
cargo build --release
```

## Usage

### Running ALNS Algorithm

```bash
python main.py
```

- Modify `config/settings.ini` for parameter tuning.
- Adjust `data/` for custom problem instances.

### Running ALNS with Reinforcement Learning

```bash
python rl/q-learn/q_learning.py
python rl/sarsa/sarsa.py
```

### Testing the Implementation

```bash
pytest tests/
```

## Known Issues

- **Python performance bottlenecks**: Currently being optimized with Rust.
- **Unresolved ALNS bugs**: Debugging in progress.
- **Data format instability**: Still refining input/output specifications.

## Future Work

- Finalizing data format and standardizing inputs.
- Fully integrating Rust for optimized ALNS performance.
- Refining RL training for better adaptation to problem variations.

## References

- Kisialiou’s ALNS Implementation (https://www.researchgate.net/publication/323188792_The_periodic_supply_vessel_planning_problem_with_flexible_departure_times_and_coupled_vessels)

## Contact

For questions or collaboration, reach out to nemytov.t@gmail.com .


```
psvpp
├─ README.md
├─ alns
│  ├─ Beans
│  │  ├─ __init__.py
│  │  ├─ node.py
│  │  ├─ schedule.py
│  │  ├─ vessel.py
│  │  ├─ visit.py
│  │  └─ voyage.py
│  ├─ __init__.py
│  ├─ alns
│  │  ├─ __init__.py
│  │  ├─ alns.py
│  │  ├─ destroy_operator.py
│  │  ├─ improve_operator.py
│  │  ├─ mutation_service.py
│  │  └─ repair_operator.py
│  ├─ data_generator.py
│  ├─ resource
│  │  ├─ __init__.py
│  │  ├─ generation_config.yaml
│  │  └─ io_config.yaml
│  ├─ rl
│  │  ├─ __init__.py
│  │  ├─ q-learn
│  │  │  ├─ __init__.py
│  │  │  └─ q learning.py
│  │  └─ sarsa
│  │     ├─ __init__.py
│  │     └─ sarsa.py
│  └─ utils
│     ├─ __init__.py
│     ├─ coord.py
│     ├─ distance_manager.py
│     ├─ io.py
│     ├─ tsp_solver.py
│     └─ utils.py
├─ alns_main.py
├─ config
│  ├─ __init__.py
│  ├─ config_utils.py
│  └─ settings.ini
├─ coop_case.py
├─ generate_dataset.py
├─ logs
├─ main.py
├─ rust_alns
│  ├─ Cargo.lock
│  ├─ Cargo.toml
│  ├─ src
│  │  ├─ lib.rs
│  │  ├─ main.rs
│  │  ├─ operators
│  │  ├─ structs
│  │  │  ├─ constants.rs
│  │  │  ├─ csv_reader.rs
│  │  │  ├─ data_loader.rs
│  │  │  ├─ distance_manager.rs
│  │  │  ├─ mod.rs
│  │  │  ├─ node.rs
│  │  │  ├─ schedule.rs
│  │  │  ├─ time_window.rs
│  │  │  ├─ transaction.rs
│  │  │  ├─ vessel.rs
│  │  │  ├─ visit.rs
│  │  │  └─ voyage.rs
│  │  └─ utils
│  │     ├─ mod.rs
│  │     └─ tsp_solver.rs
│  └─ tests
├─ rust_main.py
├─ sample
│  ├─ base
│  │  ├─ SMALL_1
│  │  │  ├─ b_test1.csv
│  │  │  └─ b_test1.pkl
│  │  ├─ SMALL_2
│  │  │  └─ b_test1.pkl
│  │  └─ SMALL_3
│  │     └─ b_test1.pkl
│  ├─ installations
│  │  ├─ SMALL_1
│  │  │  ├─ i_test1.csv
│  │  │  └─ i_test1.pkl
│  │  ├─ SMALL_2
│  │  │  └─ i_test1.pkl
│  │  └─ SMALL_3
│  │     └─ i_test1.pkl
│  ├─ solutions
│  │  ├─ SMALL_2
│  │  │  └─ sol_test1_2.pkl
│  │  └─ SMALL_3
│  │     └─ sol_test1_1.pkl
│  └─ vessels
│     ├─ SMALL_1
│     │  ├─ v_test1.csv
│     │  └─ v_test1.pkl
│     ├─ SMALL_2
│     │  └─ v_test1.pkl
│     └─ SMALL_3
│        └─ v_test1.pkl
├─ tests
│  └─ operators
│     └─ greedy.py
└─ visualization.ipynb

```