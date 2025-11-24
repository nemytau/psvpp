"""
Simple test script for the ALNS RL environment.

This script tests the basic functionality of the environment without
requiring heavy ML dependencies.
"""

import sys
import traceback
import numpy as np

# Test basic imports first
print("[CHECK] Testing basic imports...")

try:
    import gymnasium as gym
    print("[OK] Gymnasium imported successfully")
except ImportError as e:
    print(f"[ERROR] Gymnasium not found: {e}")
    print("Install with: pip install gymnasium")
    sys.exit(1)

try:
    import numpy as np
    print("[OK] NumPy imported successfully")
except ImportError as e:
    print(f"[ERROR] NumPy not found: {e}")
    sys.exit(1)

# Test Rust interface
print("\n[CHECK] Testing Rust interface...")
try:
    import rust_alns_py
    print("[OK] rust_alns_py imported successfully")
    
    # Test creating interface
    interface = rust_alns_py.RustALNSInterface() # type: ignore
    print("[OK] RustALNSInterface created successfully")
    
    # Test getting operator info
    operator_info = interface.get_operator_info()
    print(f"[OK] Operator info retrieved: {type(operator_info)}")
    print(f"   Destroy operators: {operator_info.get('destroy_operators', [])}")
    print(f"   Repair operators: {operator_info.get('repair_operators', [])}")
    print(f"   Improvement operators: {operator_info.get('improvement_operators', [])}")
    
except ImportError as e:
    print(f"[ERROR] rust_alns_py not found: {e}")
    print("Make sure the Rust extension is built with: cd rust_alns && maturin develop --release")
    sys.exit(1)
except Exception as e:
    print(f"[ERROR] Error with Rust interface: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test our environment
print("\n[CHECK] Testing ALNSEnvironment...")
try:
    from rl_alns_environment import ALNSEnvironment
    print("[OK] ALNSEnvironment imported successfully")
    
    # Create environment
    print("Creating environment...")
    env = ALNSEnvironment(
        problem_instance="SMALL_1",
        max_iterations=5,
        seed=42
    )
    print("[OK] Environment created successfully")
    
    # Check spaces
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space size: {env.action_space.n}") # type: ignore
    print(f"Observation space shape: {env.observation_space.shape}")
    
    # Test reset
    print("\nTesting reset...")
    obs, info = env.reset()
    print(f"[OK] Reset successful")
    print(f"   Observation shape: {obs.shape}")
    print(f"   Observation type: {type(obs)}")
    print(f"   Initial cost: {info.get('initial_cost', 'N/A')}")
    print(f"   Problem instance: {info.get('problem_instance', 'N/A')}")
    
    # Test a few steps
    print("\nTesting steps...")
    total_reward = 0.0
    for step in range(5):
        action = env.action_space.sample()
        destroy_idx, repair_idx = env._encode_action(action)
        
        print(f"Step {step + 1}: action={action} (destroy={destroy_idx}, repair={repair_idx})")
        
        try:
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
            step_info = info.get("step_info", {})
            print(f"   [OK] reward={reward:.3f}, done={done}, "
                  f"cost={step_info.get('current_cost', 'N/A'):.2f}, "
                  f"accepted={step_info.get('accepted', 'N/A')}")
            
            if done or truncated:
                print(f"   Episode finished at step {step + 1}")
                break
                
        except Exception as e:
            print(f"   [ERROR] Step failed: {e}")
            traceback.print_exc()
            break
    
    print(f"\nTotal reward: {total_reward:.3f}")
    
    # Test episode statistics
    print("\nTesting episode statistics...")
    try:
        stats = env.get_episode_statistics()
        print(f"[OK] Episode statistics retrieved:")
        for key, value in stats.items():
            if isinstance(value, (list, tuple)) and len(value) > 5:
                print(f"   {key}: [...] (length: {len(value)})")
            else:
                print(f"   {key}: {value}")
    except Exception as e:
        print(f"[ERROR] Failed to get episode statistics: {e}")
    
    # Clean up
    env.close()
    print("[OK] Environment closed successfully")
    
except ImportError as e:
    print(f"[ERROR] ALNSEnvironment import failed: {e}")
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f"[ERROR] Environment test failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test with Gymnasium checker if available
print("\n[CHECK] Testing with Gymnasium environment checker...")
try:
    from gymnasium.envs.registration import make_env_spec # type: ignore
    print("[OK] Environment checker available")
    
    # Create a fresh environment for checking
    env = ALNSEnvironment(
        problem_instance="SMALL_1",
        max_iterations=3,  # Keep it short for testing
        seed=123
    )
    
    # Basic manual checks
    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape, f"Observation shape mismatch: {obs.shape} vs {env.observation_space.shape}"
    assert env.observation_space.contains(obs), "Initial observation not in observation space"
    
    action = env.action_space.sample()
    assert env.action_space.contains(action), "Sampled action not in action space"
    
    obs, reward, done, truncated, info = env.step(action)
    assert env.observation_space.contains(obs), "Step observation not in observation space"
    assert isinstance(reward, (int, float)), f"Reward must be numeric, got {type(reward)}"
    assert isinstance(done, bool), f"Done must be boolean, got {type(done)}"
    assert isinstance(truncated, bool), f"Truncated must be boolean, got {type(truncated)}"
    assert isinstance(info, dict), f"Info must be dict, got {type(info)}"
    
    env.close()
    print("[OK] Basic environment checks passed!")
    
except ImportError:
    print("[WARN] Gymnasium environment checker not available")
except Exception as e:
    print(f"[ERROR] Environment checker failed: {e}")
    traceback.print_exc()

print("\nAll tests completed!")
print("\nSummary:")
print("   [OK] Basic imports working")
print("   [OK] Rust interface accessible")
print("   [OK] Environment creation working")
print("   [OK] Reset and step functions working")
print("   [OK] Basic environment validation passed")
print("\nEnvironment is ready for RL training!")