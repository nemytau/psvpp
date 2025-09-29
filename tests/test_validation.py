#!/usr/bin/env python3
"""
Comprehensive validation suite for RL-ALNS integration using SMALL_1 in    sys.path.append('.')
    from src.rl_integration.environment import ALNSRLEnvironment
    from rust_alns_py import RustALNSInterface  # type: ignore
    
    rust_interface = RustALNSInterface()e
"""

import time
import sys
import numpy as np
from pathlib import Path

def run_validation_suite():
    """Complete validation of RL-ALNS integration"""
    
    print("🚀 Starting RL-ALNS Integration Validation")
    print("=" * 60)
    
    tests = [
        ("PyO3 Module Import", test_pyo3_import),
        ("ALNS Initialization", test_alns_init), 
        ("Single Iteration", test_single_iteration),
        ("RL Environment Creation", test_env_creation),
        ("Environment Reset", test_env_reset),
        ("Environment Step", test_env_step),
        ("Complete Episode", test_complete_episode),
        ("State Extraction", test_state_extraction),
        ("Action Mapping", test_action_mapping),
        ("Performance Benchmark", test_performance),
    ]
    
    results = {}
    total_start = time.time()
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        start_time = time.time()
        
        try:
            test_func()
            elapsed = time.time() - start_time
            results[test_name] = f"✅ PASS ({elapsed:.2f}s)"
            print(f"   ✅ PASSED in {elapsed:.2f}s")
        except Exception as e:
            elapsed = time.time() - start_time
            results[test_name] = f"❌ FAIL: {str(e)[:50]}..."
            print(f"   ❌ FAILED in {elapsed:.2f}s: {e}")
    
    total_elapsed = time.time() - total_start
    
    # Results summary
    print("\n" + "=" * 60)
    print("📊 VALIDATION RESULTS SUMMARY")
    print("=" * 60)
    
    passed_count = sum(1 for result in results.values() if "✅" in result)
    total_count = len(results)
    
    for test_name, result in results.items():
        print(f"{test_name:.<40} {result}")
    
    print(f"\n🎯 OVERALL RESULT: {passed_count}/{total_count} tests passed")
    print(f"⏱️  Total time: {total_elapsed:.2f}s")
    
    if passed_count == total_count:
        print("🎉 ALL TESTS PASSED - RL-ALNS integration is ready!")
        return True
    else:
        print(f"❌ {total_count - passed_count} tests failed - check errors above")
        return False

def test_pyo3_import():
    """Test PyO3 module import"""
    try:
        from rust_alns_py import RustALNSInterface  # type: ignore
        interface = RustALNSInterface()
        assert interface is not None
    except ImportError:
        raise Exception("rust_alns_py module not found - run 'maturin develop' first")

def test_alns_init():
    """Test ALNS initialization"""
    from rust_alns_py import RustALNSInterface  # type: ignore
    interface = RustALNSInterface()
    result = interface.initialize_alns("SMALL_1", 42)
    assert isinstance(result, dict)
    assert 'total_cost' in result

def test_single_iteration():
    """Test single ALNS iteration"""
    from rust_alns_py import RustALNSInterface  # type: ignore
    interface = RustALNSInterface()
    interface.initialize_alns("SMALL_1", 42)
    result = interface.execute_iteration(0, 0, 0)
    assert isinstance(result, dict)
    assert 'current_cost' in result

def test_env_creation():
    """Test RL environment creation"""
    import sys
    sys.path.append('.')
    from src.rl_integration.environment import ALNSRLEnvironment
    from rust_alns_py import RustALNSInterface # type: ignore
    
    rust_interface = RustALNSInterface()
    env = ALNSRLEnvironment(rust_alns_interface=rust_interface, problem_instance="SMALL_1", max_iterations=10)
    assert env is not None
    assert hasattr(env, 'action_space')
    assert hasattr(env, 'observation_space')

def test_env_reset():
    """Test environment reset"""
    import sys
    sys.path.append('.')
    from src.rl_integration.environment import ALNSRLEnvironment
    from rust_alns_py import RustALNSInterface # type: ignore
    
    rust_interface = RustALNSInterface()
    env = ALNSRLEnvironment(rust_alns_interface=rust_interface, problem_instance="SMALL_1", max_iterations=10)
    obs, info = env.reset(seed=42)
    assert obs is not None
    assert isinstance(info, dict)

def test_env_step():
    """Test environment step"""
    import sys
    sys.path.append('.')
    from src.rl_integration.environment import ALNSRLEnvironment
    from rust_alns_py import RustALNSInterface # type: ignore
    
    rust_interface = RustALNSInterface()
    env = ALNSRLEnvironment(rust_alns_interface=rust_interface, problem_instance="SMALL_1", max_iterations=10)
    obs, info = env.reset(seed=42)
    obs, reward, terminated, truncated, info = env.step(0)
    assert obs is not None
    assert isinstance(reward, (int, float))

def test_complete_episode():
    """Test complete episode"""
    import sys
    sys.path.append('.')
    from src.rl_integration.environment import ALNSRLEnvironment
    from rust_alns_py import RustALNSInterface # type: ignore
    
    rust_interface = RustALNSInterface()
    env = ALNSRLEnvironment(rust_alns_interface=rust_interface, problem_instance="SMALL_1", max_iterations=10)
    obs, info = env.reset(seed=42)
    
    for _ in range(10):
        action = int(env.action_space.sample()) if env.action_space else 0
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    
    assert True  # If we get here without errors, test passes

def test_state_extraction():
    """Test state feature extraction"""
    import sys
    sys.path.append('.')
    from src.rl_integration.environment import SolutionMetrics
    
    # Create dummy metrics
    metrics = SolutionMetrics(
        total_cost=1000.0,
        cost_normalized=1.0,
        is_complete=True,
        is_feasible=True,
        feasibility_violations=0,
        num_voyages=3,
        num_empty_voyages=1,
        num_vessels_used=2,
        avg_voyage_utilization=2.5,
        cost_improvement_ratio=0.05,
        stagnation_count=2,
        iteration=5,
        iteration_normalized=0.25,
        temperature=100.0,
        temperature_normalized=0.1,
        destroy_operator_success_rates=[0.6, 0.7, 0.5],
        repair_operator_success_rates=[0.8, 0.6, 0.7],
        recent_operator_rewards=[10.0, -5.0, 15.0, 0.0, 8.0]
    )
    
    feature_vector = metrics.to_feature_vector()
    assert len(feature_vector) > 0
    assert all(isinstance(x, (int, float, np.integer, np.floating)) for x in feature_vector)
    assert not any(np.isnan(x) for x in feature_vector)

def test_action_mapping():
    """Test action to operator mapping"""
    import sys
    sys.path.append('.')
    from src.rl_integration.environment import OperatorSelectionAction
    
    destroy_ops = ["shaw_removal", "random_removal", "worst_removal"]
    repair_ops = ["deep_greedy", "k_regret_2", "k_regret_3"]
    action_mapper = OperatorSelectionAction(destroy_ops, repair_ops)
    
    # Test all possible actions
    for action in range(9):  # 3x3 = 9 combinations
        destroy_idx, repair_idx = action_mapper.action_to_operators(action)
        assert 0 <= destroy_idx < 3
        assert 0 <= repair_idx < 3
        
        # Test description
        desc = action_mapper.get_action_description(action)
        assert isinstance(desc, str)
        assert len(desc) > 0

def test_performance():
    """Test performance benchmarks"""
    import sys
    sys.path.append('.')
    from src.rl_integration.environment import ALNSRLEnvironment
    from rust_alns_py import RustALNSInterface # type: ignore
    
    rust_interface = RustALNSInterface()
    env = ALNSRLEnvironment(rust_alns_interface=rust_interface, problem_instance="SMALL_1", max_iterations=20)
    
    # Benchmark reset time
    start_time = time.time()
    for _ in range(3):  # Reduced iterations for faster testing
        env.reset(seed=42)
    reset_time = (time.time() - start_time) / 3
    
    # Benchmark step time  
    obs, info = env.reset(seed=42)
    start_time = time.time()
    for i in range(10):
        action = int(i % (env.action_space.n if env.action_space else 9))
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    step_time = (time.time() - start_time) / 10
    
    print(f"     Reset time: {reset_time*1000:.1f}ms")
    print(f"     Step time: {step_time*1000:.1f}ms")
    print(f"     Training speed estimate: {1/step_time:.0f} steps/sec")
    
    # Basic performance requirements (relaxed for initial testing)
    assert reset_time < 5.0  # Reset should be under 5 seconds
    assert step_time < 2.0   # Step should be under 2 seconds

if __name__ == "__main__":
    success = run_validation_suite()
    sys.exit(0 if success else 1)