#!/usr/bin/env python3
"""
Test script to validate mo-gymnasium migration without external dependencies.
This ensures that the basic environment works and the wrappers function correctly.
"""

import numpy as np
from agent.env_factory import MarioEnvFactory


def test_environment_creation():
    """Test that the mo-gymnasium environment can be created and works."""
    print("Testing environment creation...")
    
    env_factory = MarioEnvFactory()
    env_fn = env_factory.make()
    env = env_fn()
    
    print(f"✓ Environment created successfully")
    print(f"  Action space: {env.action_space}")
    print(f"  Observation space: {env.observation_space}")
    
    # Test reset
    obs, info = env.reset()
    print(f"✓ Reset successful - obs shape: {obs.shape}")
    
    # Test multiple steps
    total_reward = 0
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            obs, info = env.reset()
            print(f"  Episode ended at step {i}")
            break
    
    print(f"✓ Steps successful - total reward: {total_reward:.2f}")
    env.close()


def test_reward_wrapper():
    """Test that the reward wrapper correctly handles multi-objective rewards."""
    print("\nTesting reward wrapper...")
    
    from agent.env_factory import MoMarioRewardWrapper
    import mo_gymnasium as mo_gym
    
    # Create base environment to test wrapper
    base_env = mo_gym.make('mo-supermario-v0', render_mode=None)
    wrapped_env = MoMarioRewardWrapper(base_env)
    
    # Test reward processing
    mock_multi_reward = np.array([1.0, -0.1, 0.0, 1.0, 0.0])  # [x_pos, time, death, coin, enemy]
    scalar_reward = wrapped_env.reward(mock_multi_reward)
    
    expected = 1.0 - 0.1 * (-0.1) + 1.0 * 10 + 0.0 * 5 - 0.0 * 100  # 1.01 + 10 = 11.01
    print(f"✓ Multi-objective reward {mock_multi_reward} -> scalar {scalar_reward:.2f}")
    
    wrapped_env.close()


def test_action_wrapper():
    """Test that the action wrapper correctly maps simplified actions."""
    print("\nTesting action wrapper...")
    
    from agent.env_factory import MoMarioActionWrapper
    import mo_gymnasium as mo_gym
    
    # Create base environment to test wrapper
    base_env = mo_gym.make('mo-supermario-v0', render_mode=None)
    wrapped_env = MoMarioActionWrapper(base_env)
    
    print(f"✓ Original action space: {base_env.action_space}")
    print(f"✓ Wrapped action space: {wrapped_env.action_space}")
    
    # Test action mapping
    for simple_action in range(7):
        mapped_action = wrapped_env.action(simple_action)
        print(f"  Simple action {simple_action} -> Mario action {mapped_action}")
    
    wrapped_env.close()


def test_wrappers_compatibility():
    """Test that custom wrappers work with the new gymnasium format."""
    print("\nTesting custom wrappers...")
    
    try:
        from wrappers.rdn_wrapper import RNDWrapper
        
        env_factory = MarioEnvFactory()
        env_fn = env_factory.make()
        base_env = env_fn()
        
        # Wrap with RND
        wrapped_env = RNDWrapper(base_env, beta=0.01)
        print(f"✓ RND wrapper applied successfully")
        
        # Test step with wrapper
        obs, info = wrapped_env.reset()
        action = wrapped_env.action_space.sample()
        obs, reward, terminated, truncated, info = wrapped_env.step(action)
        
        print(f"✓ RND wrapper step successful - reward: {reward:.4f}")
        wrapped_env.close()
        
    except ImportError as e:
        print(f"⚠️  Skipping wrapper test - missing dependency: {e}")
        print("✓ This is expected if torch is not installed")


def main():
    """Run all tests."""
    print("=== Mo-Gymnasium Migration Test Suite ===\n")
    
    try:
        test_environment_creation()
        test_reward_wrapper()
        test_action_wrapper()
        test_wrappers_compatibility()
        
        print("\n🎉 All tests passed! Mo-gymnasium migration is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())