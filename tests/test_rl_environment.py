#!/usr/bin/env python3
"""
Phase 3: Test RL environment with SMALL_1 instance
"""

def test_rl_environment():
    """Test the RL environment with SMALL_1 instance"""
    print("🧪 Testing RL Environment with SMALL_1 instance...")
    
    try:
        import sys
        sys.path.append('.')
        from src.rl_integration.environment import ALNSRLEnvironment, SolutionMetrics
        print("✅ Successfully imported RL environment classes")
    except ImportError as e:
        print(f"❌ Failed to import RL environment: {e}")
        return False
    
    try:
        from src.rl_integration.environment import ALNSRLEnvironment, SolutionMetrics
        print("✅ Successfully imported RL environment classes")
    except ImportError as e:
        print(f"❌ Failed to import RL environment: {e}")
        return False
    
    try:
        # Import the PyO3 interface to pass to environment
        from rust_alns_py import RustALNSInterface # type: ignore
        rust_interface = RustALNSInterface()
        
        # Test 1: Create environment
        print("\n1. Creating RL environment...")
        env = ALNSRLEnvironment(
            rust_alns_interface=rust_interface,
            problem_instance="SMALL_1", 
            max_iterations=50
        )
        print("✅ Environment created successfully")
        print(f"   Action space: {env.action_space}")
        print(f"   Observation space shape: {env.observation_space.shape if env.observation_space else 'Not initialized'}")
        
        # Test 2: Reset environment
        print("\n2. Testing environment reset...")
        observation, info = env.reset(seed=42)
        print("✅ Environment reset successful")
        print(f"   Observation shape: {observation.shape}")
        print(f"   Initial cost: {info.get('initial_cost', 'N/A')}")
        
        # Test 3: Test all actions
        print("\n3. Testing all possible actions...")
        for action in range(min(9, env.action_space.n if env.action_space else 9)):  # Test first 9 actions
            obs, reward, terminated, truncated, info = env.step(action)
            
            destroy_idx, repair_idx = env.action_selector.action_to_operators(action)
            destroy_name = env.destroy_operators[destroy_idx] if destroy_idx < len(env.destroy_operators) else f"Op{destroy_idx}"
            repair_name = env.repair_operators[repair_idx] if repair_idx < len(env.repair_operators) else f"Op{repair_idx}"
            
            print(f"   Action {action} ({destroy_name} + {repair_name}): "
                  f"Reward={reward:.3f}, Cost={info.get('current_cost', 0):.2f}")
            
            if terminated or truncated:
                print("   Episode terminated, resetting...")
                obs, info = env.reset()
                break
        
        # Test 4: Complete episode
        print("\n4. Testing complete episode with random policy...")
        obs, info = env.reset(seed=42)
        episode_rewards = []
        episode_costs = []
        
        for step in range(20):  # Short episode
            action = int(env.action_space.sample()) if env.action_space else 0
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_rewards.append(reward)
            episode_costs.append(info.get('current_cost', float('inf')))
            
            if step % 5 == 0:
                print(f"   Step {step}: Action={action}, Reward={reward:.3f}, Cost={info.get('current_cost', 0):.2f}")
            
            if terminated or truncated:
                print(f"   Episode ended at step {step}")
                break
        
        print(f"✅ Episode completed:")
        print(f"   Total steps: {len(episode_rewards)}")
        print(f"   Total reward: {sum(episode_rewards):.3f}")
        print(f"   Best cost: {min(episode_costs):.2f}")
        print(f"   Final cost: {episode_costs[-1]:.2f}")
        
        print("\n🎉 All RL environment tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ RL environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_rl_environment()