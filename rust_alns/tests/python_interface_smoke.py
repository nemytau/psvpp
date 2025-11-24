"""Quick smoke test for the rust_alns_py module.

The script initializes an ALNS run and performs ten iterations
using a simple operator cycling policy. It prints basic metrics
per iteration so we can verify the solver makes progress and the
Python bindings behave as expected.
"""

from __future__ import annotations

from pathlib import Path

from rust_alns_py import RustALNSInterface  # type: ignore[attr-defined]


def main() -> None:
    # Ensure we execute from the repository root so relative data paths resolve.
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = repo_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    interface = RustALNSInterface()
    
    # Test 1: Default parameters (backward compatibility)
    print("=== Test 1: Default Parameters ===")
    init_metrics = interface.initialize_alns("SMALL_1", seed=42)
    print("Initial metrics (defaults):")
    print(dict(init_metrics))
    
    # Test 2: Custom parameters
    print("\n=== Test 2: Custom Parameters ===")
    interface2 = RustALNSInterface()
    init_metrics2 = interface2.initialize_alns("SMALL_1", seed=42, temperature=100.0, theta=0.95, weight_update_interval=5)
    print("Initial metrics (custom: temp=100, theta=0.95, weight_update=5):")
    print(dict(init_metrics2))

    operator_info = interface2.get_operator_info()
    destroy_ops = len(operator_info["destroy_operators"])
    repair_ops = len(operator_info["repair_operators"])
    improvement_ops = len(operator_info.get("improvement_operators", []))

    print(
        f"Configured operators -> destroy: {destroy_ops}, repair: {repair_ops}, improvement: {improvement_ops}"
    )

    print("\nRunning 20 ALNS iterations with custom parameters...")
    last_metrics = None
    for iteration in range(20):
        destroy_idx = iteration % destroy_ops
        repair_idx = (iteration + 1) % repair_ops
        metrics = interface2.execute_iteration(
            iteration,
            destroy_operator_idx=destroy_idx,
            repair_operator_idx=repair_idx,
        )
        metrics = dict(metrics)
        last_metrics = metrics
        
        # Show weight updates at custom interval (every 5 iterations)
        weight_update_marker = " [WEIGHTS UPDATED]" if (iteration + 1) % 5 == 0 else ""
        print(
            f"Iter {iteration + 1:02d}: cost={metrics['current_cost']:.4f}, "
            f"best={metrics['best_cost']:.4f}, accepted={metrics['accepted']}, "
            f"temp={metrics['temperature']:.2f}, destroy={destroy_idx}, repair={repair_idx}{weight_update_marker}"
        )

    # Export the final solution snapshot for manual inspection if needed.
    export_path = output_dir / "python_interface_solution.json"
    interface2.export_solution(str(export_path))
    print(f"\nSolution exported to {export_path.relative_to(repo_root)}")

    if last_metrics is not None:
        print("Final metrics:")
        print(last_metrics)


if __name__ == "__main__":
    main()
