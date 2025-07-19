import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from agent.mario_cnn import MarioCNN
from agent.env_builder import make_vec_mario_env
from torch.utils.tensorboard import SummaryWriter
from agent.utils.reward_normalizer import RewardNormalizer
import time
from collections import Counter
import copy
from agent.utils.log_metrics import log_metrics
from tqdm import trange

class PPOTrainer:
    def __init__(
        self,
        total_timesteps=1_000_000,
        lr=2.5e-4,
        gamma=0.99,
        gae_lambda=0.98,
        clip_coef=0.2,
        update_epochs=4,
        batch_size=256,
        num_envs=8,
    ):

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.writer = SummaryWriter(log_dir=f"runs/ppo_mario/{timestamp}")

        # Hiperparámetros
        self.total_timesteps = total_timesteps
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.num_envs = num_envs

        self.reward_normalizer = RewardNormalizer()

        self.initial_kl = 0.3
        self.final_kl = 0.05

        self.action_counter = Counter()

        # Entorno vectorizado
        self.env = make_vec_mario_env(num_envs=self.num_envs, seed=42)
        obs_shape = self.env.single_observation_space.shape  # (stack, H, W, C)
        n_actions = self.env.single_action_space.n

        c, h, w = obs_shape  # Modelo y optimizador

        self.device = (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = MarioCNN((c, h, w), n_actions).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def compute_gae(self, rewards, dones, values, next_value):
        advantages = []
        gae = 0
        for step in reversed(range(len(rewards))):
            delta = (
                rewards[step]
                + self.gamma * next_value * (1 - dones[step])
                - values[step]
            )
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[step]) * gae
            advantages.insert(0, gae)
            next_value = values[step]
        return advantages

    def to_tensor(self, array, dtype, flatten=True):
        t = torch.tensor(np.array(array), dtype=dtype).to(self.device)
        return t.view(-1) if flatten else t

    def train(self, num_steps=128):
        obs, _ = self.env.reset()
        print(
            "shape obs:", obs.shape
        )  # Debugging: Verifica la forma de las observaciones
        obs = torch.tensor(obs, dtype=torch.float32)  # (N, C, H, W)
        obs /= 255.0  # normalizar

        total_updates = self.total_timesteps // (num_steps * self.num_envs)
        device = next(self.model.parameters()).device

        with trange(total_updates, desc="Training", leave=True) as pbar:
            for update in pbar:
                
                obs_buf, actions_buf, logprobs_buf, rewards_buf, dones_buf, values_buf = ([], [], [], [], [], [])
                prev_x_pos, max_x_pos = np.zeros(self.num_envs, dtype=np.float32), np.zeros(self.num_envs, dtype=np.float32)

                # lr = self.lr * (1 - (update / total_updates)) ** 0.5
                # for param_group in self.optimizer.param_groups:
                #     param_group['lr'] = lr

                # Rollout
                for step in range(num_steps):
                    logits, value = self.model(obs.to(device))
                    dist = Categorical(logits=logits.cpu())
                    action = dist.sample().to(device)
                    epsilon = max(0.05, 0.5 * (1 - update / total_updates))

                    if np.random.rand() < epsilon:
                        action = torch.tensor(
                            [
                                self.env.single_action_space.sample()
                                for _ in range(self.num_envs)
                            ]
                        )
                    else:
                        action = dist.sample()

                    logprob = dist.log_prob(action.cpu()).to(device)

                    next_obs, reward, terminated, truncated, info = self.env.step(
                        action.cpu().numpy()
                    )
                    done = np.logical_or(terminated, truncated)

                    # Guardar en buffers
                    obs_buf.append(obs.cpu().numpy())
                    actions_buf.append(action.cpu().numpy())
                    logprobs_buf.append(logprob.detach().cpu().numpy())

                    self.action_counter.update(action.cpu().numpy().tolist())

                    #reward = np.clip(reward, -1.0, 1.0)

                    normalized_reward = self.reward_normalizer.normalize(np.array(reward))

                    curr_x_pos = np.array(info["x_pos"])
                    # Checkpoints rewards
                    for env in range(self.num_envs):
                        if curr_x_pos[env] > max_x_pos[env]:
                            normalized_reward[env] += 0.01 * (curr_x_pos[env] - max_x_pos[env])
                            max_x_pos[env] = curr_x_pos[env]
                        #if curr_x_pos[env] > 2000:
                        #    normalized_reward[env] += 0.1  # Bonus por llegar a 2000

                    delta_x = curr_x_pos - prev_x_pos
                    progress_reward = delta_x * 0.01
                    normalized_reward += np.where(delta_x > 0, progress_reward, 0)

                    rewards_buf.append(normalized_reward)
                    dones_buf.append(done)
                    values_buf.append(value.squeeze(-1).detach().cpu().numpy())

                    obs = torch.tensor(next_obs, dtype=torch.float32)
                    obs /= 255.0  # normalizar
                    prev_x_pos = curr_x_pos

                    if "flag_get" in info:
                        flag_bonus = np.array(info["flag_get"], dtype=np.float32)
                        normalized_reward += flag_bonus * 5.0

                # Último valor para GAE
                with torch.no_grad():
                    _, next_value = self.model(obs.to(device))
                next_value = next_value.squeeze(-1).cpu().numpy()

                # Procesar buffers
                rewards_buf = np.array(rewards_buf)
                dones_buf = np.array(dones_buf)
                values_buf = np.array(values_buf)
                advantages = self.compute_gae(
                    rewards_buf, dones_buf, values_buf, next_value
                )
                returns = advantages + values_buf

                # Convertir a tensores
                obs_tensor = self.to_tensor(obs_buf, dtype=torch.float32, flatten=False).view(-1, *obs_buf[0].shape[1:])
                actions_tensor = self.to_tensor(actions_buf, torch.long)
                logprobs_tensor = self.to_tensor(logprobs_buf, torch.float32)
                advantages_tensor = self.to_tensor(advantages, torch.float32)
                returns_tensor = self.to_tensor(returns, torch.float32)

                # Normalizar ventajas
                advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (
                    advantages_tensor.std() + 1e-8
                )

                old_model_state = copy.deepcopy(self.model.state_dict())
                old_optimizer_state = copy.deepcopy(self.optimizer.state_dict())

                # PPO Updates
                for epoch in range(self.update_epochs):
                    logits, values = self.model(obs_tensor)
                    if torch.isnan(logits).any():
                        print("🚨 NaN detected in logits")
                        continue
                    dist = Categorical(logits=logits)
                    new_logprobs = dist.log_prob(actions_tensor)
                    ratio = (new_logprobs - logprobs_tensor).exp()
                    ratio = torch.clamp(ratio, 1e-4, 1e4)

                    # Policy loss
                    surr1 = ratio * advantages_tensor
                    surr2 = (
                        torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef)
                        * advantages_tensor
                    )
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # Value loss
                    value_loss = nn.functional.mse_loss(values.squeeze(-1), returns_tensor)
                    entropy_coef = max(0.05, 0.15 * (1 - (update / total_updates) ** 0.5))

                    # Total loss
                    loss = ( policy_loss + 0.5 * value_loss - entropy_coef * dist.entropy().mean() )

                    progress = update / total_updates
                    target_kl = self.initial_kl * (1 - progress) + self.final_kl * progress
                    # 🎯 Approx KL before optimizer step
                    approx_kl = (logprobs_tensor - new_logprobs).mean().item()

                    if (
                        approx_kl > target_kl * 2
                    ):  # 🚨 rollback threshold (stricter than early stop)
                        print(f"⏪ Rollback: KL={approx_kl:.4f} > 2*target={target_kl:.4f}")
                        # 🔄 Restaurar modelo y optimizador
                        self.model.load_state_dict(old_model_state)
                        self.optimizer.load_state_dict(old_optimizer_state)
                        self.optimizer.zero_grad()
                        break  # Salir del bucle de epochs (rollback)

                    elif approx_kl > target_kl:
                        print(
                            f"🚨 Early stopping at epoch {epoch} due to KL={approx_kl:.4f} (target={target_kl:.4f})"
                        )
                        break  # 🛑 skip optimizer step`

                    # Backprop
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                    self.optimizer.step()

                total_actions = sum(self.action_counter.values())
                for action_id in range(self.env.single_action_space.n):
                    count = self.action_counter[action_id]
                    proportion = count / total_actions if total_actions > 0 else 0.0
                    self.writer.add_scalar(
                        f"actions/action_{action_id}_proportion", proportion, update
                    )

                log_metrics(
                    writer=self.writer,
                    update=update,
                    rewards_buf=rewards_buf,
                    max_x_pos=max_x_pos,
                    info=info,
                    dist_entropy=dist.entropy().mean().item(),
                    action_counter=self.action_counter,
                    action_space_n=self.env.single_action_space.n,
                    loss=loss.item()
                )
                self.action_counter.clear()
    
                pbar.set_postfix(loss=loss.item(), entropy=dist.entropy().mean().item())

        # Guardar modelo
        torch.save(self.model.state_dict(), "ppo_mario.pth")
        print("🎉 Modelo guardado en ppo_mario.pth")

