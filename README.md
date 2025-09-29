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
