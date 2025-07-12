# RL-PPO-Mario

A Reinforcement Learning implementation using PPO (Proximal Policy Optimization) to train an agent to play Super Mario Bros. 

## Recent Migration to mo-gymnasium

This project has been migrated from `gym-super-mario-bros` to `mo-gymnasium` for better compatibility with:
- 🍎 Apple Silicon (M1/M2) - no more compilation issues
- 🔄 Gymnasium API - fully compatible with modern RL libraries  
- 📦 Simplified dependencies - no more `nes-py` complications
- 🏃‍♂️ Stable Baselines3 - seamless integration

### Key Changes
- **Environment**: Now uses `mo-supermario-v0` instead of `SuperMarioBros-v0`
- **Action Space**: Simplified to 7 actions (compatible with original SIMPLE_MOVEMENT)
- **Rewards**: Multi-objective rewards automatically converted to scalar
- **API**: Full Gymnasium API compliance

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Test the migration
python test_mo_gymnasium_migration.py

# Train the agent
python main.py --mode train --timesteps 1000000

# Evaluate the agent
python main.py --mode eval --render
```
