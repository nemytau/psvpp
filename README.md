# Supply Vessel Planning with ALNS & Reinforcement Learning

## Overview

This project implements an Adaptive Large Neighborhood Search (ALNS) algorithm for the supply vessel planning problem. It builds upon Kisialiou’s ALNS implementation and extends it with a Reinforcement Learning (RL) layer to enhance adaptability to different problem layouts.

## Features

- **ALNS Implementation**: Core heuristic search for optimizing vessel routing.
- **Reinforcement Learning (RL) Enhancement**: Improves ALNS adaptability using Q-learning and SARSA.
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

