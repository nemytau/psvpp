#!/usr/bin/env python3
"""
Example demonstrating configurable ALNS parameters in the PyO3 interface.

This script shows how to use the new optional parameters:
- temperature: Controls simulated annealing acceptance probability
- theta: Cooling factor (temperature decay rate)  
- weight_update_interval: How often operator weights are updated

Usage:
    python examples/configurable_parameters.py
"""

from rust_alns_py import RustALNSInterface  # type: ignore[attr-defined]


def run_experiment(name: str, **kwargs) -> None:
    """Run an ALNS experiment with given parameters."""
    print(f"\n=== {name} ===")
    
    interface = RustALNSInterface()
    
    # Show the parameters being used
    temp = kwargs.get('temperature', 500.0)
    theta = kwargs.get('theta', 0.9)
    interval = kwargs.get('weight_update_interval', 10)
    print(f"Parameters: temperature={temp}, theta={theta}, weight_update_interval={interval}")
    
    # Initialize with custom parameters
    init_metrics = interface.initialize_alns("SMALL_1", seed=42, **kwargs)
    initial_cost = init_metrics['total_cost']
    print(f"Initial cost: {initial_cost:.4f}")
    
    # Run 15 iterations to see weight updates
    best_cost = initial_cost
    for iteration in range(15):
        destroy_idx = iteration % 3
        repair_idx = (iteration + 1) % 3
        metrics = interface.execute_iteration(destroy_idx, repair_idx, iteration)
        
        current_cost = metrics['current_cost']
        best_cost = min(best_cost, metrics['best_cost'])
        temp_now = metrics['temperature']
        accepted = metrics['accepted']
        
        # Mark weight update iterations
        weight_marker = " [WEIGHTS]" if (iteration + 1) % interval == 0 else ""
        
        print(f"  Iter {iteration+1:2d}: cost={current_cost:6.2f}, "
              f"best={best_cost:6.2f}, temp={temp_now:6.1f}, "
              f"accepted={str(accepted):5s}{weight_marker}")
    
    improvement = ((initial_cost - best_cost) / initial_cost) * 100
    print(f"Final improvement: {improvement:.2f}%")


def main() -> None:
    """Demonstrate various parameter configurations."""
    
    print("Configurable ALNS Parameters Demo")
    print("=" * 50)
    
    # Experiment 1: Default parameters (backward compatibility)
    run_experiment("Default Parameters")
    
    # Experiment 2: High temperature (more exploration)
    run_experiment("High Temperature", temperature=1000.0)
    
    # Experiment 3: Low temperature (more exploitation)  
    run_experiment("Low Temperature", temperature=50.0)
    
    # Experiment 4: Slow cooling
    run_experiment("Slow Cooling", theta=0.99)
    
    # Experiment 5: Fast cooling
    run_experiment("Fast Cooling", theta=0.8)
    
    # Experiment 6: Frequent weight updates
    run_experiment("Frequent Weight Updates", weight_update_interval=3)
    
    # Experiment 7: Infrequent weight updates
    run_experiment("Infrequent Weight Updates", weight_update_interval=20)
    
    # Experiment 8: Combined custom settings
    run_experiment("Custom Combination", 
                   temperature=200.0, 
                   theta=0.95, 
                   weight_update_interval=5)


if __name__ == "__main__":
    main()