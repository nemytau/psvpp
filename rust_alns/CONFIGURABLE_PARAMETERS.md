# Configurable ALNS Parameters

The PyO3 interface now supports configurable parameters for fine-tuning the ALNS algorithm behavior.

## Updated Interface

### `initialize_alns(problem_instance, seed, temperature=None, theta=None, weight_update_interval=None, aggressive_search_factor=None, algorithm_mode=None, kisialiou_improvement_sequence=None, max_iterations=None, fuel_price=None)`

**Parameters:**
- `problem_instance` (str): Problem instance name (e.g., "SMALL_1")
- `seed` (int): Random seed for reproducibility
- `temperature` (float, optional): Initial simulated annealing temperature (default: 500.0)
- `theta` (float, optional): Cooling factor for temperature decay (default: 0.9) 
- `weight_update_interval` (int, optional): Iterations between operator weight updates (default: 10)
- `aggressive_search_factor` (float, optional): Fraction of iterations after which strict acceptance starts (default: 0.85)
- `max_iterations` (int, optional): Iteration horizon of the engine = the length of one restart (default: 1000, the historical value). It sets the cooling rate (`mu = 1 - theta / max_iterations` per iteration, so a full restart ends at `T0 * exp(-theta)`) and the start of the aggressive phase (`aggressive_search_factor * max_iterations`). Step-driven callers must pass their real restart length; `rl/rl_alns_environment.py` does (`PSVPP_ENGINE_HORIZON` overrides it). Experiments 8-10 ran 300-iteration restarts against the default 1000.
- `fuel_price` (float, optional): Fuel price in NOK per tonne (default: 13,500, the price of Experiment 10; `rust_alns_py.fuel_price_nok_per_tonne()`). Together with the weekly charter in the dataset it makes the cost of a solution: vessels used x charter + fuel price x fuel burnt. The default start temperature follows it (500 at 650 NOK/t, proportional). The environment of the main line passes `rl/cost_model.py`'s price (the original fuel-to-charter ratio at the current charter, 7,583 NOK/t).

**Backward Compatibility:**
The old interface `initialize_alns(problem_instance, seed)` still works with default values.

## Parameter Effects

### Temperature
- **Higher values** (e.g., 1000.0): More exploration, accepts worse solutions more frequently
- **Lower values** (e.g., 50.0): More exploitation, stricter about accepting worse solutions
- **Formula**: Acceptance probability = exp(-(cost_increase)/temperature)

### Theta (Cooling Factor)
- **Higher values** (e.g., 0.99): Slower cooling, maintains exploration longer
- **Lower values** (e.g., 0.8): Faster cooling, transitions to exploitation sooner
- **Formula** (as implemented in `alns/acceptance.rs`, Kisialiou et al. 2018, section 3.8): every iteration `temperature *= 1 - theta / max_iterations`

### Weight Update Interval
- **Lower values** (e.g., 3): Frequent weight updates, operators adapt quickly
- **Higher values** (e.g., 20): Infrequent updates, more stable operator selection
- **Effect**: Controls how often successful operators get higher selection weights

### Aggressive Search Factor
- **Rule**: Strict phase starts at `iteration >= aggressive_search_factor * max_iterations`
- **Strict behavior**: Only global-best improvements are accepted; otherwise the current solution is reset to the best known solution
- **Higher values** (e.g., 0.9): Strict mode starts later
- **Lower values** (e.g., 0.5): Strict mode starts earlier

## Examples

```python
from rust_alns_py import RustALNSInterface

interface = RustALNSInterface()

# Default parameters
interface.initialize_alns("SMALL_1", seed=42)

# High exploration setup
interface.initialize_alns("SMALL_1", seed=42, temperature=1000.0, theta=0.99)

# High exploitation setup  
interface.initialize_alns("SMALL_1", seed=42, temperature=50.0, theta=0.8)

# Frequent weight updates
interface.initialize_alns("SMALL_1", seed=42, weight_update_interval=5)

# Enable earlier strict acceptance phase
interface.initialize_alns("SMALL_1", seed=42, aggressive_search_factor=0.6)

# Combined custom settings
interface.initialize_alns("SMALL_1", seed=42, 
                         temperature=200.0, 
                         theta=0.95, 
                         weight_update_interval=7,
                         aggressive_search_factor=0.85)
```

## Files Modified

- `src/python_interface.rs`: Added optional parameters to PyO3 interface
- `ALNSState` struct: Added `weight_update_interval` field
- `initialize_from_instance()`: Updated to accept new parameters
- `run_iteration()`: Uses configurable weight update interval instead of hardcoded 10

## Testing

Run the example script to see different parameter combinations:
```bash
python examples/configurable_parameters.py
```

Or run the updated smoke test:
```bash  
python tests/python_interface_smoke.py
```