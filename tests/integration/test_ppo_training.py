#!/usr/bin/env python3
"""
Phase 4B: Test PPO training on SMALL_1 instance
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
from pathlib import Path

def test_ppo_training():
    """Test PPO training on SMALL_1 instance"""
    print("🤖 Testing PPO Training on SMALL_1 instance...")
    
    try:
        # Import required libraries
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.logger import configure
        import sys
        sys.path.append('.')
        from src.rl_integration.environment import ALNSRLEnvironment
        from rust_alns_py import RustALNSInterface # type: ignore
        
        print("✅ Successfully imported PPO and dependencies")
        
        # Create environment
        rust_interface = RustALNSInterface()
        env = ALNSRLEnvironment(
            rust_alns_interface=rust_interface,
            problem_instance="SMALL_1", 
            max_iterations=100
        )
        print("✅ Environment created")
        
        # Test random baseline first
        print("\n📊 Testing random baseline...")
        baseline_results = test_baseline_policy(env, episodes=3)
        
        # Custom callback to track training
        class TrainingCallback(BaseCallback):
            def __init__(self, verbose=0):
                super(TrainingCallback, self).__init__(verbose)
                self.episode_rewards = []
                self.episode_costs = []
                
            def _on_step(self) -> bool:
                # Check if episode ended
                if self.locals.get('dones', [False])[0]:
                    # Get info from environment
                    infos = self.locals.get('infos', [{}])
                    if infos and len(infos) > 0:
                        info = infos[0]
                        if 'episode' in info:
                            episode_reward = info['episode']['r']
                            best_cost = info.get('best_cost', 'N/A')
                            self.episode_rewards.append(episode_reward)
                            if best_cost != 'N/A':
                                self.episode_costs.append(best_cost)
                            
                            if len(self.episode_rewards) % 10 == 0:
                                print(f"   Episode {len(self.episode_rewards)}: Reward={episode_reward:.3f}, Best Cost={best_cost}")
                
                return True
        
        # Create callback
        callback = TrainingCallback()
        
        # Create PPO model with conservative hyperparameters
        print("\n🧠 Creating PPO model...")
        model = PPO(
            "MlpPolicy", 
            env, 
            verbose=1,
            learning_rate=0.0003,  # Conservative learning rate
            n_steps=64,           # Smaller batch size
            batch_size=32,        # Small batch size
            n_epochs=3,           # Fewer epochs
            gamma=0.95,           # Discount factor
            gae_lambda=0.95,      # GAE lambda
            clip_range=0.2,       # PPO clip range
            ent_coef=0.01,        # Entropy coefficient for exploration
            device="cpu"
        )
        print("✅ PPO model created")
        
        # Train model (short training for testing)
        print("\n🚀 Starting PPO training...")
        print("   Training for 2000 timesteps...")
        model.learn(total_timesteps=2000, callback=callback)
        print("✅ Training completed")
        
        # Test trained model
        print("\n🧪 Testing trained PPO agent...")
        trained_results = test_trained_policy(model, env, episodes=5)
        
        # Compare results
        print("\n📊 Performance Comparison:")
        print(f"Random Policy:")
        print(f"   Average Best Cost: {baseline_results['avg_best_cost']:.2f}")
        print(f"   Average Final Cost: {baseline_results['avg_final_cost']:.2f}")
        print(f"   Average Episode Reward: {baseline_results['avg_episode_reward']:.3f}")
        
        print(f"\nTrained PPO Agent:")
        print(f"   Average Best Cost: {trained_results['avg_best_cost']:.2f}")
        print(f"   Average Final Cost: {trained_results['avg_final_cost']:.2f}")
        print(f"   Average Episode Reward: {trained_results['avg_episode_reward']:.3f}")
        
        # Calculate improvement
        cost_improvement = (baseline_results['avg_best_cost'] - trained_results['avg_best_cost']) / baseline_results['avg_best_cost'] * 100
        reward_improvement = (trained_results['avg_episode_reward'] - baseline_results['avg_episode_reward']) / abs(baseline_results['avg_episode_reward']) * 100 if baseline_results['avg_episode_reward'] != 0 else 0
        
        print(f"\nImprovement:")
        print(f"   Cost: {cost_improvement:+.1f}% ({'better' if cost_improvement > 0 else 'worse'})")
        print(f"   Reward: {reward_improvement:+.1f}% ({'better' if reward_improvement > 0 else 'worse'})")
        
        # Save model
        model.save("ppo_small1_model")
        print("✅ Model saved as 'ppo_small1_model'")
        
        # Generate training plot if we have episode data
        if len(callback.episode_rewards) > 0:
            create_training_plot(callback.episode_rewards, callback.episode_costs)
        
        print("\n🎉 PPO training test completed successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Missing required libraries: {e}")
        print("📋 Install with: pip install stable-baselines3[extra] matplotlib")
        return False
    except Exception as e:
        print(f"❌ PPO training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_baseline_policy(env, episodes=3):
    """Test random baseline policy"""
    print("🎲 Testing random baseline...")
    
    all_best_costs = []
    all_final_costs = []
    all_episode_rewards = []
    
    for episode in range(episodes):
        obs, info = env.reset(seed=42 + episode)
        episode_reward = 0
        costs = [info.get('initial_cost', float('inf'))]
        
        for step in range(50):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            costs.append(info.get('current_cost', float('inf')))
            
            if terminated or truncated:
                break
        
        best_cost = min(costs)
        final_cost = costs[-1]
        
        all_best_costs.append(best_cost)
        all_final_costs.append(final_cost)
        all_episode_rewards.append(episode_reward)
        
        print(f"   Baseline Episode {episode + 1}: Best={best_cost:.2f}, Final={final_cost:.2f}, Reward={episode_reward:.3f}")
    
    return {
        'avg_best_cost': np.mean(all_best_costs),
        'avg_final_cost': np.mean(all_final_costs),
        'avg_episode_reward': np.mean(all_episode_rewards)
    }

def test_trained_policy(model, env, episodes=5):
    """Test trained PPO policy"""
    print("🤖 Testing trained PPO policy...")
    
    all_best_costs = []
    all_final_costs = []
    all_episode_rewards = []
    
    for episode in range(episodes):
        obs, info = env.reset(seed=42 + episode)
        episode_reward = 0
        costs = [info.get('initial_cost', float('inf'))]
        
        for step in range(50):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            costs.append(info.get('current_cost', float('inf')))
            
            if terminated or truncated:
                break
        
        best_cost = min(costs)
        final_cost = costs[-1]
        
        all_best_costs.append(best_cost)
        all_final_costs.append(final_cost)
        all_episode_rewards.append(episode_reward)
        
        print(f"   PPO Episode {episode + 1}: Best={best_cost:.2f}, Final={final_cost:.2f}, Reward={episode_reward:.3f}")
    
    return {
        'avg_best_cost': np.mean(all_best_costs),
        'avg_final_cost': np.mean(all_final_costs),
        'avg_episode_reward': np.mean(all_episode_rewards)
    }

def create_training_plot(episode_rewards, episode_costs):
    """Create a simple training progress plot"""
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot episode rewards
        ax1.plot(episode_rewards)
        ax1.set_title('Episode Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.grid(True)
        
        # Plot episode costs (if available)
        if len(episode_costs) > 0:
            ax2.plot(episode_costs)
            ax2.set_title('Best Costs per Episode')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Best Cost')
            ax2.grid(True)
        else:
            ax2.text(0.5, 0.5, 'No cost data available', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Best Costs per Episode')
        
        plt.tight_layout()
        plt.savefig('ppo_training_progress.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("📊 Training plot saved as 'ppo_training_progress.png'")
        
    except Exception as e:
        print(f"⚠️  Could not create training plot: {e}")

if __name__ == "__main__":
    print("🚀 Starting PPO Training Test")
    
    success = test_ppo_training()
    
    print(f"\n{'✅ SUCCESS' if success else '❌ FAILED'}: PPO training test completed")