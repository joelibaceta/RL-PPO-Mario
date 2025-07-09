import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np

class RNDWrapper(gym.Wrapper):
    def __init__(self, env, beta=0.01):
        super(RNDWrapper, self).__init__(env)
        self.beta = beta
        obs_shape = env.observation_space.shape

        self.target = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(obs_shape), 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        ).eval()  # Frozen

        self.predictor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(obs_shape), 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=1e-3)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Handle multi-objective reward from mo-gymnasium (take first objective)
        if isinstance(reward, (list, tuple)) and len(reward) > 1:
            reward = reward[0]  # Primary objective (x-position progress)
        
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

        # Calculate intrinsic reward
        with torch.no_grad():
            target_features = self.target(obs_tensor)
        predicted_features = self.predictor(obs_tensor)
        intrinsic_reward = torch.norm(target_features - predicted_features, p=2).item()

        # Add curiosity
        total_reward = reward + self.beta * intrinsic_reward

        # Update predictor
        loss = torch.nn.functional.mse_loss(predicted_features, target_features)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Return gymnasium format (terminated, truncated)
        return obs, total_reward, terminated, truncated, info