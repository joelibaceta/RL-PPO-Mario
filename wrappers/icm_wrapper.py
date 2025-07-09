import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np

class ICMWrapper(gym.Wrapper):
    def __init__(self, env, beta=0.01, feature_dim=256):
        super(ICMWrapper, self).__init__(env)
        self.beta = beta  # balance curiosity
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Feature extractor (e.g. simple CNN or identity)
        self.feature = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(env.observation_space.shape), feature_dim),
            nn.ReLU()
        ).to(self.device)
        
        # Inverse Model: predicts action given (state, next_state)
        self.inverse_model = nn.Sequential(
            nn.Linear(2 * feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, env.action_space.n)
        ).to(self.device)

        # Forward Model: predicts next_state_features
        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + env.action_space.n, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim)
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            list(self.inverse_model.parameters()) + list(self.forward_model.parameters()), lr=1e-3
        )

    def step(self, action):
        state = self.env.state if hasattr(self.env, 'state') else None
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Handle multi-objective reward from mo-gymnasium (take first objective)
        if isinstance(reward, (list, tuple)) and len(reward) > 1:
            reward = reward[0]  # Primary objective (x-position progress)

        # Compute intrinsic reward
        s_t = torch.tensor(self.feature(obs).detach(), device=self.device).float()
        s_tp1 = torch.tensor(self.feature(obs).detach(), device=self.device).float()

        # One-hot encode action
        a_t = torch.zeros((1, self.env.action_space.n), device=self.device)
        a_t[0, action] = 1.0

        # Forward prediction
        pred_s_tp1 = self.forward_model(torch.cat([s_t, a_t], dim=1))
        intrinsic_reward = torch.norm(pred_s_tp1 - s_tp1, dim=1).mean().item()

        # Add curiosity reward
        total_reward = reward + self.beta * intrinsic_reward

        # Update models
        self.optimizer.zero_grad()
        forward_loss = nn.functional.mse_loss(pred_s_tp1, s_tp1)
        inverse_logits = self.inverse_model(torch.cat([s_t, s_tp1], dim=1))
        inverse_loss = nn.functional.cross_entropy(inverse_logits, torch.tensor([action], device=self.device))
        loss = forward_loss + inverse_loss
        loss.backward()
        self.optimizer.step()

        # Return gymnasium format (terminated, truncated)
        return obs, total_reward, terminated, truncated, info