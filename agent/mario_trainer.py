import os
import time
import numpy as np
import torch
import torch.nn as nn
import random
from tqdm import tqdm
from torch.distributions.categorical import Categorical
from agent.env_builder import make_mario_env
from agent.mario_lstm import MarioConvLSTM

class RewardNormalizer:
    def __init__(self, epsilon=1e-8, clip_range=(-1.0, 1.0), alpha=0.999):
        self.mean = 0.0
        self.var = 1.0
        self.alpha = alpha  # tasa de actualización
        self.epsilon = epsilon
        self.clip_range = clip_range

    def normalize(self, reward):
        # Actualiza la media y varianza
        self.mean = self.alpha * self.mean + (1 - self.alpha) * reward
        self.var = self.alpha * self.var + (1 - self.alpha) * (reward - self.mean) ** 2

        std = np.sqrt(self.var) + self.epsilon

        # Normaliza y recorta
        normalized = (reward - self.mean) / std
        clipped = np.clip(normalized, *self.clip_range)

        return clipped


class MarioTrainer:
    def __init__(self,
                 total_timesteps=2_000_000,
                 learning_rate=1e-4,
                 num_steps=1024,
                 update_epochs=6,
                 gamma=0.99,
                 gae_lambda=0.99,
                 clip_coef=0.05,
                 ent_coef=0.1,  
                 vf_coef=0.5,
                 max_grad_norm=0.5,
                 exploration_ratio=0.1,
                 seed=0,
                 model_dir="models_cleanrl",
                 model_name="ppo_mario_cleanrl.pth"):

        # Hiperparámetros
        self.total_timesteps = total_timesteps
        self.learning_rate = learning_rate
        self.num_steps = num_steps
        self.update_epochs = update_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.exploration_ratio = exploration_ratio
        self.seed = random.seed(1000)

        self.reward_normalizer = RewardNormalizer()

        # Dispositivo
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"⚡️ Usando dispositivo: {self.device}")

        # Directorio para guardar modelo
        self.model_dir = model_dir
        self.model_name = model_name
        os.makedirs(self.model_dir, exist_ok=True)

        # Entorno
        self.env = make_mario_env(seed=self.seed)
        obs_shape = self.env.observation_space.shape  # (stack, H, W, C)
        stack, H, W, C = obs_shape
        channels = stack * C
        obs_shape_pt = (channels, H, W)
        n_actions = self.env.action_space.n

        # Modelo LSTM
        self.model = MarioConvLSTM(obs_shape_pt, 512, n_actions).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-5)

        # Buffers
        self._init_buffers(obs_shape)

        # Cálculo de exploración
        total_updates = self.total_timesteps // self.num_steps
        self.exploration_updates = int(self.exploration_ratio * total_updates)

    def _init_buffers(self, obs_shape):
        """Inicializa buffers para almacenar rollout"""
        self.obs_buf = np.zeros((self.num_steps, *obs_shape), dtype=np.uint8)
        self.actions_buf = np.zeros(self.num_steps, dtype=np.int64)
        self.logprobs_buf = np.zeros(self.num_steps, dtype=np.float32)
        self.rewards_buf = np.zeros(self.num_steps, dtype=np.float32)
        self.dones_buf = np.zeros(self.num_steps, dtype=np.float32)
        self.values_buf = np.zeros(self.num_steps, dtype=np.float32)

    def linear_schedule(initial_value):
        def func(progress_remaining):
            return progress_remaining * initial_value
        return func

    def preprocess_obs(self, obs):
        """Convierte (stack, H, W, C) → (1, stack*C, H, W)"""
        obs = torch.tensor(np.array(obs), dtype=torch.float32)  # asegura ndarray
        obs = obs.permute(0, 3, 1, 2)  # (stack, C, H, W)
        obs = obs.reshape(1, -1, obs.shape[-2], obs.shape[-1])  # (batch=1, stack*C, H, W)
        return obs.to(self.device) / 255.0

    def compute_gae(self, rewards, dones, values, next_value):
        """Calcula Generalized Advantage Estimation (GAE)"""
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * next_value * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[step]) * gae
            next_value = values[step]
            returns.insert(0, gae + values[step])
        return returns

    def random_noop_start(self, env, max_noops=30):
        """Realiza un número aleatorio de acciones para aleatorizar el estado inicial"""
        noop_count = np.random.randint(1, max_noops + 1)
        for _ in range(noop_count):
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()
        return obs
                
    def visualize_activations(self, obs, num_filters=8):
        import matplotlib.pyplot as plt

        # Preprocesa igual que para la CNN
        obs_tensor = torch.tensor(np.array(obs), dtype=torch.float32).permute(0, 3, 1, 2).to(self.device) / 255.0
        obs_tensor = obs_tensor.reshape(1, -1, obs_tensor.shape[-2], obs_tensor.shape[-1])  # (batch=1, stack*C, H, W)

        # Extrae capas convolucionales (sin Flatten ni Linear)
        conv_layers = list(self.model.feature_extractor.children())
        conv_layers = [layer for layer in conv_layers if isinstance(layer, (nn.Conv2d, nn.ReLU, nn.MaxPool2d))]
        conv_extractor = nn.Sequential(*conv_layers).to(self.device)

        # Calcula activaciones
        with torch.no_grad():
            activations = conv_extractor(obs_tensor).mps()

        print(f"📊 Activations shape: {activations.shape}")  # (batch, channels, H, W)

        # Quita dimensión batch
        activations = activations.squeeze(0)  # (channels, H, W)
        num_filters = min(num_filters, activations.shape[0])  # No más que canales disponibles

        # Dibuja imagen original + filtros
        fig, axes = plt.subplots(1, num_filters + 1, figsize=(2 * (num_filters + 1), 2))

        # Convierte obs a numpy para visualizar
        obs_np = np.array(obs)
        input_frame = obs_np[0] if obs_np.shape[0] == 1 else obs_np[0, :, :, :]  # Selecciona primer frame
        input_frame = np.squeeze(input_frame)  # Quita dimensiones extra si (H,W,1)

        if input_frame.ndim == 2:  # Escala de grises
            axes[0].imshow(input_frame, cmap="gray")
        else:  # RGB
            axes[0].imshow(input_frame)
        axes[0].set_title("Input")
        axes[0].axis("off")

        # Muestra activaciones
        for i in range(num_filters):
            ax = axes[i + 1]
            act = activations[i]
            ax.imshow(act, cmap="viridis")
            ax.set_title(f"Filtro {i}")
            ax.axis("off")

        plt.tight_layout()
        plt.show()

    def train(self):
        obs, _ = self.env.reset()
        obs = np.expand_dims(obs, axis=0)
        print(f"🎨 Obs shape después de FrameStack: {obs.shape}")
        obs = self.random_noop_start(self.env)

        hidden_state = None  # Inicializa estado oculto LSTM
        max_x_pos = 0
        self.episode_length = 0

        total_updates = self.total_timesteps // self.num_steps

        print(f"🧵 PyTorch usando {torch.get_num_threads()} hilos")

        for update in tqdm(range(1, total_updates + 1), desc="📈 Entrenando PPO", unit="update"):
            self.update_start_time = time.time()
            prev_x_pos = 0


            progress = update / total_updates 

            # 🔥 Annealing del learning rate
            lr = self.learning_rate * (1 - progress)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            # 🔥 Annealing de entropy coeficiente
            self.ent_coef = 0.1 * (1 - progress) + 0.01 * progress

            # Rollout
            for step in range(self.num_steps):
                #if step % 100 == 0:
                #   print("🔍 Visualizando activaciones CNN...")
                #   self.visualize_activations(obs)
                self.episode_length += 1
                tensor_obs = self.preprocess_obs(obs).unsqueeze(1)

                logits, value, hidden_state = self.model(tensor_obs, hidden_state)

                # 🔥 Detach el estado oculto para evitar backprop de grafos anteriores
                hidden_state = (hidden_state[0].detach(), hidden_state[1].detach())

                dist = Categorical(logits=logits)

                # Exploración
                epsilon = max(0.1, 0.5 - (update / (self.exploration_updates * 1.5)))
                if torch.rand(1).item() < epsilon:
                    action = self.env.action_space.sample()
                    logprob = 0.0
                    value = torch.tensor(0.0).to(self.device)
                else:
                    action = dist.sample().item()
                    logprob = dist.log_prob(torch.tensor(action).to(self.device)).item()

                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                reward = self.reward_normalizer.normalize(reward)

                # Recompensas
                curr_x_pos = info.get("x_pos", 0)
                
                reward += np.clip((curr_x_pos - prev_x_pos) / 10.0, -1.0, 1.0) # recompensa por avance
                #reward -= 0.05  # penaliza quedarse quieto
                


                if action == 3 or action == 7:  # saltar o saltar y avanzar
                    reward += 0.05  # premio base
                    if curr_x_pos > prev_x_pos:
                        reward += (curr_x_pos - prev_x_pos) * 0.005
                    else:
                        reward -= 0.05

                if prev_x_pos == curr_x_pos:
                    reward -= 0.01

                prev_x_pos = curr_x_pos
 

                # Buffers
                self.obs_buf[step] = obs
                self.actions_buf[step] = action
                self.logprobs_buf[step] = logprob
                self.rewards_buf[step] = reward
                self.dones_buf[step] = done
                self.values_buf[step] = value.item()

                obs = next_obs

                if done:

                    self.episode_length = 0

                    if curr_x_pos > max_x_pos:
                        max_x_pos = curr_x_pos
                        reward += 0.1  # bonificación por alcanzar nueva posición máxima

                    if curr_x_pos > 800: reward += 0.1
                    if curr_x_pos > 1000: reward += 0.2
                    if curr_x_pos > 1200: reward += 0.3
                    if curr_x_pos > 1500: reward += 0.4
                    if curr_x_pos > 2000: reward += 0.5
                    if curr_x_pos > 2500: reward += 0.6
                    if curr_x_pos > 3000: reward += 0.7

                    hidden_state = (
                        torch.zeros(1, 1, 512).to(self.device),
                        torch.zeros(1, 1, 512).to(self.device)
                    )
                    obs, _ = self.env.reset()

            # GAE y PPO
            self._update_policy(obs, update)

        # Guardar modelo
        torch.save(self.model.state_dict(), os.path.join(self.model_dir, self.model_name))
        print(f"📦 Modelo guardado: {self.model_name}")

        self.env.close()
        print("✅ Entrenamiento CleanRL completado.")

    def _update_policy(self, obs, update):
        """Actualiza la política PPO"""
        tensor_obs = torch.tensor(np.array(self.obs_buf), dtype=torch.float32)
        tensor_obs = tensor_obs.permute(0, 1, 4, 2, 3)
        tensor_obs = tensor_obs.reshape(self.num_steps, -1, tensor_obs.shape[-2], tensor_obs.shape[-1]) / 255.0
        tensor_obs = tensor_obs.to(self.device)

        hidden_state = None  # 🔥 Reinicia hidden state para el update
        _, next_value, _ = self.model(self.preprocess_obs(obs).unsqueeze(1), hidden_state)
        returns = self.compute_gae(self.rewards_buf, self.dones_buf, self.values_buf, next_value.item())

        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        values = torch.tensor(self.values_buf, dtype=torch.float32).to(self.device)
        logprobs = torch.tensor(self.logprobs_buf, dtype=torch.float32).to(self.device)
        actions = torch.tensor(self.actions_buf, dtype=torch.int64).to(self.device)
        logprobs_old = torch.tensor(self.logprobs_buf, dtype=torch.float32).to(self.device)

        initial_kl = 1.5   # Mucho más permisivo al principio
        final_kl   = 0.02  # Estricto cuando ya aprendió

        total_updates = self.total_timesteps // self.num_steps
        progress = update / total_updates
        target_kl = max(0.05, initial_kl * (1 - progress) + final_kl * progress)
        

        warmup_updates = 20

        for epoch in range(self.update_epochs):
            logits, value, hidden_state = self.model(tensor_obs.unsqueeze(1), hidden_state)

            # 🔥 Detach hidden_state para evitar acumulación de grafos
            hidden_state = (hidden_state[0].detach(), hidden_state[1].detach())

            dist = Categorical(logits=logits)
            entropy = dist.entropy().mean()
            newlogprob = dist.log_prob(actions)
            ratio = (newlogprob - logprobs).exp()

            pg_loss1 = -ratio * (returns - values).detach()
            pg_loss2 = -torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef) * (returns - values).detach()
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()
            v_loss = ((returns - value.squeeze()) ** 2).mean()
            loss = pg_loss + self.vf_coef * v_loss - self.ent_coef * entropy

            # ✅ Cálculo de KL divergence
            if update > warmup_updates:
                approx_kl = (logprobs_old - newlogprob).mean()
                if update > warmup_updates and approx_kl > target_kl:
                    print(f"🚨 Early stopping at epoch {epoch} due to KL={approx_kl:.4f} (target={target_kl:.4f})")
                    break  # Detenemos el update si KL se pasa
            else:
                approx_kl = (logprobs_old - newlogprob).mean()

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # Log
            self.env.end_update(self.num_steps, losses={
                "entropy": entropy.item(),
                "policy": pg_loss.item(),
                "value": v_loss.item(),
                "total": loss.item(),
            })