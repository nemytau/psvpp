# Development Guide

This guide explains how to set up, develop, and contribute to the PSVPP project.

## 🛠️ Development Setup

### Prerequisites
- **Python 3.8+**: Main development language
- **Rust 1.70+**: For high-performance ALNS implementation
- **Maturin**: For building PyO3 bindings
- **Git**: Version control

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd psvpp
   ```

2. **Set up Python environment:**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install Python dependencies
   pip install -r requirements.txt
   ```

3. **Build Rust extension:**
   ```bash
   cd src/rust_alns
   maturin develop --release
   cd ../..
   ```

4. **Verify installation:**
   ```bash
   python tests/run_full_test_suite.py
   ```

## 🧪 Testing Procedures

### Running Tests

#### Full Test Suite
```bash
python tests/run_full_test_suite.py
```

#### Individual Test Components
```bash
# PyO3 interface tests
python tests/test_pyo3_interface.py

# RL environment tests
python tests/test_rl_environment.py

# Validation tests
python tests/test_validation.py

# Random policy baseline
python tests/test_random_policy.py

# PPO training tests
python tests/test_ppo_training.py
```

#### Using pytest
```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_rl_environment.py

# Run with verbose output
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src
```

### Test Structure

```
tests/
├── run_full_test_suite.py     # Orchestrates all tests
├── test_pyo3_interface.py     # Tests Rust-Python bindings
├── test_rl_environment.py     # Tests RL environment functionality
├── test_validation.py         # Comprehensive validation tests
├── test_random_policy.py      # Random policy baseline tests
├── test_ppo_training.py       # PPO training validation
└── integration/               # Integration tests
```

## 🔧 Build Process

### Rust Components

1. **Development build:**
   ```bash
   cd src/rust_alns
   maturin develop
   cd ../..
   ```

2. **Release build:**
   ```bash
   cd src/rust_alns
   maturin develop --release
   cd ../..
   ```

3. **Building wheels:**
   ```bash
   cd src/rust_alns
   maturin build --release
   cd ../..
   ```

### Python Components

The Python components don't require explicit building, but you can verify imports:

```bash
python -c "from src.rl_integration.environment import ALNSRLEnvironment; print('RL integration OK')"
python -c "import rust_alns_py; print('Rust bindings OK')"
```

## 📂 Development Workflow

### 1. Code Organization

- **Source Code**: Place new modules in appropriate `src/` subdirectories
- **Tests**: Add tests for new functionality in `tests/`
- **Scripts**: Create executable scripts in `scripts/`
- **Documentation**: Update docs in `docs/`

### 2. Branching Strategy

```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make changes and commit
git add .
git commit -m "Add: your feature description"

# Push and create PR
git push origin feature/your-feature-name
```

### 3. Making Changes

#### Adding New RL Features
1. Edit `src/rl_integration/environment.py`
2. Add tests in `tests/test_rl_environment.py`
3. Update documentation in `docs/API.md`
4. Run test suite to verify

#### Adding New ALNS Operators
1. For Python: Edit `src/py_alns/operators/`
2. For Rust: Edit `src/rust_alns/src/`
3. Add tests in appropriate test files
4. Update examples in `examples/`

#### Adding New Scripts
1. Create script in `scripts/`
2. Make it executable: `chmod +x scripts/your_script.py`
3. Add documentation in script header
4. Test functionality

### 4. Code Quality

#### Python Code Style
```bash
# Format code
black src/ tests/ scripts/

# Check style
flake8 src/ tests/ scripts/

# Type checking
mypy src/
```

#### Rust Code Style
```bash
cd src/rust_alns
cargo fmt
cargo clippy
cd ../..
```

## 🚀 Running Experiments

### Basic ALNS Runs

```bash
# Run Python ALNS
python scripts/run_python_alns.py --instance SMALL_1 --iterations 1000

# Run Rust ALNS
python scripts/run_rust_alns.py --instance SMALL_1 --iterations 1000
```

### RL Training

```bash
# Train PPO agent
python scripts/train_rl_agent.py --algorithm PPO --instance SMALL_1 --steps 10000

# Train with custom hyperparameters
python scripts/train_rl_agent.py --algorithm PPO --instance SMALL_1 --steps 10000 --learning_rate 0.001
```

### Dataset Generation

```bash
# Generate small instances
python scripts/generate_dataset.py --size small --count 10

# Generate with custom parameters
python scripts/generate_dataset.py --size medium --count 5 --vessels 3 --installations 8
```

## 📊 Performance Monitoring

### Benchmarking

```bash
# Compare Rust vs Python performance
python scripts/benchmarking/benchmark_rust_vs_python.py

# Profile RL training performance
python scripts/benchmarking/performance_profiling.py
```

### Expected Performance Metrics

- **RL Training Speed**: ~1,400 steps/second
- **ALNS Rust vs Python**: Rust typically 5-10x faster
- **Memory Usage**: Monitor for large instances

## 🐛 Debugging

### Common Issues

1. **PyO3 Import Errors**:
   ```bash
   # Rebuild Rust extension
   cd src/rust_alns
   maturin develop --release
   cd ../..
   ```

2. **RL Environment Issues**:
   ```bash
   # Check observation space dimensions
   python -c "from src.rl_integration.environment import ALNSRLEnvironment; env = ALNSRLEnvironment('SMALL_1'); print(env.observation_space)"
   ```

3. **Test Failures**:
   ```bash
   # Run individual failing test with verbose output
   python -m pytest tests/test_rl_environment.py::test_action_space -v
   ```

### Debug Mode

```bash
# Enable debug logging
export ALNS_DEBUG=1
python tests/run_full_test_suite.py
```

## 📝 Documentation Updates

### When to Update Documentation

- Adding new features or modules
- Changing APIs or interfaces
- Fixing bugs that affect usage
- Adding new examples or tutorials

### Documentation Files to Update

- `README.md`: Main project overview
- `docs/PROJECT_STRUCTURE.md`: Structure changes
- `docs/API.md`: API changes
- `docs/DEVELOPMENT.md`: This file for workflow changes

## 🔍 Code Review Checklist

### Before Submitting PRs

- [ ] All tests pass locally
- [ ] Code follows style guidelines
- [ ] New functionality has tests
- [ ] Documentation is updated
- [ ] Performance impact is considered
- [ ] Backward compatibility is maintained

### Review Focus Areas

- **Correctness**: Does the code do what it's supposed to?
- **Performance**: Are there any performance regressions?
- **Testing**: Is the test coverage adequate?
- **Documentation**: Is the code and changes well documented?
- **Integration**: Does it work well with existing components?

## 🚀 Deployment Considerations

### Package Building

```bash
# Build Python wheel
python setup.py bdist_wheel

# Build Rust extension wheel
cd src/rust_alns
maturin build --release --out ../../dist
cd ../..
```

### Environment Variables

- `ALNS_DEBUG`: Enable debug logging
- `ALNS_DATA_PATH`: Override default data directory
- `ALNS_OUTPUT_PATH`: Override default output directory

This development guide should help you navigate the codebase efficiently and contribute effectively to the project.