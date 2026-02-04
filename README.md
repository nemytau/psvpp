# PSVPP: Platform Supply Vessel Pickup and Delivery Problem

A hybrid Rust-Python implementation combining ALNS (Adaptive Large Neighborhood Search) 
with Reinforcement Learning for maritime logistics optimization.

## 🏗️ Project Structure

- `src/py_alns/` - Python ALNS implementation
- `src/rust_alns/` - Rust ALNS implementation with PyO3 bindings  
- `src/rl_integration/` - RL-ALNS integration layer
- `tests/` - Comprehensive test suite
- `scripts/` - Executable scripts for running experiments
- `examples/` - Usage examples and tutorials
- `docs/` - Documentation and design documents
- `notebooks/` - Jupyter notebooks for analysis

📖 **[Detailed Project Structure Guide](docs/PROJECT_STRUCTURE.md)**

## 🚀 Quick Start

1. **Build the Rust extension:**
   ```bash
   cd src/rust_alns
   maturin develop --release
   cd ../..
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Generate train/test datasets (optional):**
   ```bash
   python scripts/generate_datasets.py --samples 5
   ```
   This produces stratified SMALL/MEDIUM/LARGE samples under `data/generated/` for both
   training and held-out testing splits.

3. **Run tests:**
   ```bash
   python -m pytest tests/
   ```

4. **Run full test suite:**
   ```bash
   python tests/run_full_test_suite.py
   ```

## 📚 Documentation

- **[Project Structure Guide](docs/PROJECT_STRUCTURE.md)** - Detailed overview of all folders and files
- **[Development Guide](docs/DEVELOPMENT.md)** - Setup, testing, and development workflow  
- **[API Documentation](docs/API.md)** - Complete API reference for all components
- **[Configuration Guide](docs/CONFIGURATION.md)** - Parameter and configuration management *(coming soon)*

## 🧪 Testing

- `tests/test_pyo3_interface.py` - PyO3 bindings tests
- `tests/test_rl_environment.py` - RL environment tests  
- `tests/test_validation.py` - Comprehensive validation
- `tests/integration/` - Integration tests with real instances

### Rust ALNS diagnostics

- Run `cargo test --package rust_alns --test fleet_and_cost_reduction -- --nocapture` to regenerate the before/after schedule snapshots written to [rust_alns/output/tests/fleet_reduction_before.json](rust_alns/output/tests/fleet_reduction_before.json) and [rust_alns/output/tests/fleet_reduction_after.json](rust_alns/output/tests/fleet_reduction_after.json).
- Use `python tests/plot_fleet_reduction.py` (or pass `--output report.html`) to render the comparison chart defined in [tests/plot_fleet_reduction.py](tests/plot_fleet_reduction.py).
- Run `python tests/run_deep_relocation.py --limit 10 --seed 0` to exercise destroy+repair plus deep relocation across the first ten processed datasets, writing per-case snapshots and plots under [rust_alns/output/python/deep_relocation](rust_alns/output/python/deep_relocation).
- Run `python tests/run_deep_swap.py --limit 10 --seed 0` to evaluate the deep swap operator with identical destroy/repair preparation, storing snapshots and Plotly reports under [rust_alns/output/python/deep_swap](rust_alns/output/python/deep_swap).

📋 **[Full Testing Guide](docs/DEVELOPMENT.md#-testing-procedures)**

## 🎯 Components

- **ALNS Engine**: High-performance Rust implementation
- **RL Integration**: Python RL environment using Gymnasium
- **Visualization**: Interactive Dash dashboards
- **Testing**: Comprehensive validation suite

## 📊 Performance

- **Training Speed**: ~1,400 steps/second
- **Instance Support**: SMALL_1, SMALL_2, SMALL_3
- **RL Algorithms**: PPO, A2C, DQN (via Stable-Baselines3)
