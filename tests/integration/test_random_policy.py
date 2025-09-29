#!/usr/bin/env python3
"""
Phase 4: Test with random policy on SMALL_1 instance
"""

import numpy as np
import sys
sys.path.append('.')
from src.rl_integration.environment import ALNSRLEnvironment
from rust_alns_py import RustALNSInterface # type: ignore

def test_random_policy(episodes=5):
    """Test random policy performance on SMALL_1 instance"""
    print("🎲 Testing Random Policy on SMALL_1 instance...")
    
    try:
        rust_interface = RustALNSInterface()
        env = ALNSRLEnvironment(
            rust_alns_interface=rust_interface, 
            problem_instance="SMALL_1", 
            max_iterations=50
        )
        
        all_episode_rewards = []
        all_best_costs = []
        all_final_costs = []
        all_steps = []
        
        for episode in range(episodes):
            print(f"\n📊 Episode {episode + 1}/{episodes}")
            
            obs, info = env.reset(seed=42 + episode)
            initial_cost = info.get('initial_cost', float('inf'))
            print(f"   Initial cost: {initial_cost:.2f}")
            
            episode_reward = 0
            costs = [initial_cost]
            rewards = []
            actions_taken = []
            
            for step in range(50):
                action = int(env.action_space.sample()) if env.action_space else 0
                obs, reward, terminated, truncated, info = env.step(action)
                
                episode_reward += reward
                current_cost = info.get('current_cost', float('inf'))
                costs.append(current_cost)
                rewards.append(reward)
                actions_taken.append(action)
                
                # Log every 10 steps
                if step % 10 == 0:
                    destroy_idx, repair_idx = env.action_selector.action_to_operators(int(action))
                    print(f"   Step {step}: Action={action}({destroy_idx},{repair_idx}), "
                          f"Reward={reward:.3f}, Cost={current_cost:.2f}")
                
                if terminated or truncated:
                    print(f"   Episode terminated at step {step}")
                    break
            
            # Episode statistics
            best_cost = min(costs)
            final_cost = costs[-1]
            improvement = (initial_cost - best_cost) / initial_cost * 100 if initial_cost > 0 else 0
            
            all_episode_rewards.append(episode_reward)
            all_best_costs.append(best_cost)
            all_final_costs.append(final_cost)
            all_steps.append(len(costs) - 1)  # -1 because costs includes initial
            
            print(f"   📈 Episode {episode + 1} Results:")
            print(f"      Total Reward: {episode_reward:.3f}")
            print(f"      Best Cost: {best_cost:.2f} (improvement: {improvement:.1f}%)")
            print(f"      Final Cost: {final_cost:.2f}")
            print(f"      Steps: {len(costs) - 1}")
            print(f"      Avg Reward per Step: {episode_reward / len(rewards):.3f}")
        
        # Overall statistics
        print(f"\n📊 Overall Random Policy Results ({episodes} episodes):")
        print(f"   Average Episode Reward: {np.mean(all_episode_rewards):.3f} ± {np.std(all_episode_rewards):.3f}")
        print(f"   Average Best Cost: {np.mean(all_best_costs):.2f} ± {np.std(all_best_costs):.2f}")
        print(f"   Average Final Cost: {np.mean(all_final_costs):.2f} ± {np.std(all_final_costs):.2f}")
        print(f"   Average Steps: {np.mean(all_steps):.1f} ± {np.std(all_steps):.1f}")
        
        # Best episode
        best_episode_idx = np.argmin(all_best_costs)
        print(f"   Best Episode: #{best_episode_idx + 1} with cost {all_best_costs[best_episode_idx]:.2f}")
        
        print("\n✅ Random policy test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Random policy test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def analyze_action_usage():
    """Analyze which actions are used and their effectiveness"""
    print("\n🔍 Analyzing Action Usage and Effectiveness...")
    
    try:
        rust_interface = RustALNSInterface()
        env = ALNSRLEnvironment(
            rust_alns_interface=rust_interface,
            problem_instance="SMALL_1", 
            max_iterations=30
        )
        
        # Track action statistics
        action_counts = {}
        action_rewards = {}
        action_improvements = {}
        
        obs, info = env.reset(seed=42)
        
        for step in range(100):  # Longer episode for statistics
            action = int(env.action_space.sample()) if env.action_space else 0
            prev_cost = info.get('current_cost', float('inf'))
            
            obs, reward, terminated, truncated, info = env.step(action)
            current_cost = info.get('current_cost', float('inf'))
            
            # Track statistics
            if action not in action_counts:
                action_counts[action] = 0
                action_rewards[action] = []
                action_improvements[action] = []
            
            action_counts[action] += 1
            action_rewards[action].append(reward)
            
            if prev_cost < float('inf') and current_cost < float('inf'):
                improvement = prev_cost - current_cost
                action_improvements[action].append(improvement)
            
            if terminated or truncated:
                obs, info = env.reset()
                if step > 80:  # Don't reset too close to end
                    break
        
        # Analyze results
        print("\n📊 Action Analysis Results:")
        print(f"{'Action':<8} {'Count':<6} {'Avg Reward':<12} {'Avg Improvement':<15} {'Description'}")
        print("-" * 70)
        
        for action in sorted(action_counts.keys()):
            count = action_counts[action]
            avg_reward = np.mean(action_rewards[action]) if action_rewards[action] else 0.0
            avg_improvement = np.mean(action_improvements[action]) if action_improvements[action] else 0.0
            description = env.action_selector.get_action_description(action)
            
            print(f"{action:<8} {count:<6} {avg_reward:<12.3f} {avg_improvement:<15.2f} {description}")
        
        # Find best and worst actions
        best_action = max(action_rewards.keys(), key=lambda a: np.mean(action_rewards[a]) if action_rewards[a] else -float('inf'))
        worst_action = min(action_rewards.keys(), key=lambda a: np.mean(action_rewards[a]) if action_rewards[a] else float('inf'))
        
        print(f"\n🎯 Best performing action: {best_action} ({env.action_selector.get_action_description(best_action)})")
        print(f"   Average reward: {np.mean(action_rewards[best_action]):.3f}")
        
        print(f"🚫 Worst performing action: {worst_action} ({env.action_selector.get_action_description(worst_action)})")
        print(f"   Average reward: {np.mean(action_rewards[worst_action]):.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Action analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting Random Policy Testing")
    
    # Test random policy
    success = test_random_policy(episodes=3)
    
    if success:
        # Analyze action usage
        analyze_action_usage()
    
    print(f"\n{'✅ SUCCESS' if success else '❌ FAILED'}: Random policy testing completed")