import os
import cv2
import time
import numpy as np
import math
import torch
import torch.nn as nn
import random
from tqdm import tqdm
from torch.distributions.categorical import Categorical
from agent.env_builder import make_mario_env, make_vec_mario_env
from agent.mario_lstm import MarioConvLSTM
from agent.mario_cnn import MarioCNN
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from collections import Counter

from torch.utils.tensorboard import SummaryWriter
from utils.reward_normalizer import RewardNormalizer

class MarioTrainer:
    def __init__(self,
                 total_timesteps=1_000_000,
                 learning_rate=2.5e-4,
                 num_steps=256,
                 update_epochs=4,
                 gamma=0.99,
                 gae_lambda=0.95,
                 clip_coef=0.2,
                 ent_coef=0.05,  
                 vf_coef=0.5,
                 noop_interval=5,
                 max_grad_norm=0.5,
                 exploration_ratio=0.2,
                 model_dir="models_cleanrl",
                 model_name="ppo_mario_cleanrl.pth",
                 num_envs=10):

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
        self.num_envs = num_envs
        self.noop_interval = noop_interval
        self.action_counter = Counter()
        self.reward_normalizer = RewardNormalizer()
        self.current_episode_rewards = np.zeros(self.num_envs, dtype=np.float32)

        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") # Dispositivo

        # Directorio para guardar modelo
        self.model_dir = model_dir
        self.model_name = model_name
        os.makedirs(self.model_dir, exist_ok=True)

        run_name = time.strftime("run_%Y%m%d_%H%M%S")
        self.writer = SummaryWriter(log_dir=f"logs/env_{self.seed}/run_{run_name}")
        print(f"📓 TensorBoard logs en: logs/env_{self.seed}/run_{run_name}")

        self.update_count = 0
        self.update_start_time = time.time()
        self.update_episode_rewards = []
        self.update_episode_lengths = []
        self.update_max_x_positions = []
        self.update_deaths = 0

        self.current_episode_lengths = np.zeros(self.num_envs)
        self.current_max_x_positions = np.zeros(self.num_envs, dtype=np.float32)
        # Entorno
        self.env = env = make_vec_mario_env(num_envs=8, seed=42)
        single_obs_space = self.env.single_observation_space
        obs_shape = single_obs_space.shape  # (stack, H, W, C)
        stack, H, W, C = obs_shape
        channels = stack * C
        obs_shape_pt = (channels, H, W)
        n_actions = self.env.single_action_space.n

        # Modelo LSTM
        self.model = MarioCNN(obs_shape_pt, 512, n_actions).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-5)

        # Buffers
        self._init_buffers(obs_shape)

        # Cálculo de exploración
        total_updates = self.total_timesteps // self.num_steps
        self.exploration_updates = int(self.exploration_ratio * total_updates)

    def _init_buffers(self, obs_shape):
        """Inicializa buffers para almacenar rollout"""
        self.obs_buf = np.zeros((self.num_steps, self.num_envs, *obs_shape), dtype=np.uint8)
        self.actions_buf = np.zeros((self.num_steps, self.num_envs), dtype=np.int64)
        self.logprobs_buf = np.zeros((self.num_steps, self.num_envs), dtype=np.float32)
        self.rewards_buf = np.zeros((self.num_steps, self.num_envs), dtype=np.float32)
        self.dones_buf = np.zeros((self.num_steps, self.num_envs), dtype=np.float32)
        self.values_buf = np.zeros((self.num_steps, self.num_envs), dtype=np.float32)

    def preprocess_obs(self, obs):
        """
        Convierte:
        - (num_envs, stack, H, W, C) → (num_envs, stack*C, H, W)
        - (stack, H, W, C) → (1, stack*C, H, W)
        """
        obs = torch.tensor(obs, dtype=torch.float32)

        if obs.ndim == 5: # Multi-env: (num_envs, stack, H, W, C)
            obs = obs.permute(0, 1, 4, 2, 3)  # (num_envs, stack, C, H, W)
            obs = obs.reshape(obs.shape[0], -1, obs.shape[-2], obs.shape[-1])  # (num_envs, stack*C, H, W)
        elif obs.ndim == 4: # Single env: (stack, H, W, C)
            obs = obs.permute(0, 3, 1, 2)  # (stack, C, H, W)
            obs = obs.reshape(1, -1, obs.shape[-2], obs.shape[-1])  # (1, stack*C, H, W)
        else:
            raise ValueError(f"Unexpected obs shape: {obs.shape}")

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
    
    def random_noop_start(self, env, max_steps=30):
        """
        Realiza pasos aleatorios al inicio para diversificar el estado inicial.
        Reinicia solo los entornos que terminen durante el warm-up.
        """
        obs, _ = env.reset()

        for step in range(max_steps):
            actions = [random.randint(0, env.single_action_space.n - 1)
                    for _ in range(env.num_envs)]
            obs, _, terminated, truncated, _ = env.step(actions)

            # Reinicia solo los envs que terminaron
            for i, (term, trunc) in enumerate(zip(terminated, truncated)):
                if term or trunc:
                    obs_i, _ = env.reset()
                    obs[i] = obs_i[i]
        return obs
                        
    def visualize_activations(self, obs, num_filters=8):
        import matplotlib.pyplot as plt

        if isinstance(obs, torch.Tensor):
            obs_tensor = obs
        else:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)

        frame_tensor = obs_tensor[0].unsqueeze(0)  # (1, C_total, H, W)
        print("Frame shape:", frame_tensor.shape)

        expected_c = self.model.feature_extractor[0].in_channels
        if frame_tensor.shape[1] != expected_c:
            raise ValueError(f"⚠️ CNN espera {expected_c} canales pero el frame tiene {frame_tensor.shape[1]} canales")

        frame_for_plot = frame_tensor[0]
        input_image = frame_for_plot[0].cpu().numpy()

        conv_layers = [layer for layer in self.model.feature_extractor]

        current_activation = frame_tensor
        for idx, layer in enumerate(conv_layers):
            with torch.no_grad():
                current_activation = layer(current_activation)

            activations = current_activation.squeeze(0)
            num_filters = min(num_filters, activations.shape[0])

            print(f"📊 Activations Layer {idx} shape: {activations.shape}")

            fig, axes = plt.subplots(1, num_filters + 1, figsize=(2 * (num_filters + 1), 2))

            if idx == 0:
                # En la primera capa, muestra la imagen original
                axes[0].imshow(input_image, cmap="gray")
                axes[0].set_title("Input")
            else:
                # En capas posteriores, muestra un placeholder
                axes[0].text(0.5, 0.5, f"Layer {idx-1}\noutput", 
                            ha='center', va='center', fontsize=8)
                axes[0].set_title(f"Layer {idx-1}")
            axes[0].axis("off")

            for i in range(num_filters):
                act = activations[i].cpu().numpy()
                act_min, act_max = act.min(), act.max()
                if act_max > act_min:
                    act = (act - act_min) / (act_max - act_min)
                else:
                    act = np.zeros_like(act)

                ax = axes[i + 1]
                ax.imshow(act, cmap="viridis")
                ax.set_title(f"F{i}")
                ax.axis("off")

            plt.suptitle(f"Activations after Layer {idx}")
            plt.tight_layout()
            plt.show()
            
    def k_greedy_sample(self, probs, k=3, temperature=1.0):
        probs = np.asarray(probs).flatten()  # 🔥 convierte a array 1D seguro
        k = min(k, len(probs))  # Ajusta k si hay menos acciones

        top_k_idx = probs.argsort()[::-1][:k]
        top_k_probs = probs[top_k_idx]

        # 🔥 Aplica temperatura para suavizar
        logits = np.log(top_k_probs + 1e-8) / temperature
        soft_probs = np.exp(logits)
        soft_probs /= soft_probs.sum()

        return np.random.choice(top_k_idx, p=soft_probs)
    
    def record_video(self, update, video_length=1000, fps=30):
        """
        Graba un video del agente jugando durante `video_length` steps usando OpenCV.
        """
        video_dir = os.path.join("videos", f"update_{update}")
        os.makedirs(video_dir, exist_ok=True)
        video_path = os.path.join(video_dir, f"ppo_mario_update_{update}.mp4")

        # Crear entorno
        env = make_mario_env(render_mode="rgb_array")
        obs, _ = env.reset()
        hidden_state = (
            torch.zeros(1, 1, 512).to(self.device),
            torch.zeros(1, 1, 512).to(self.device),
        )

        # Obtener dimensiones del frame
        sample_frame = env.render()
        height, width, channels = sample_frame.shape

        # Configurar el escritor de video
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec para .mp4
        out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

        for step in range(video_length):
            frame = env.render()

            if step % self.noop_interval == 0:
                noop_action = np.zeros(num_envs, dtype=np.int32)
                obs, _, terminated, truncated, infos = self.env.step(noop_action.tolist())

            # Convertir de RGB a BGR para OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)

            # Inferencia del agente
            tensor_obs = self.preprocess_obs(obs)
            logits, _, hidden_state = self.model(tensor_obs, hidden_state)

            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            print(f"Step {step}: Probabilidades = {probs}")

            action = torch.argmax(logits, dim=-1).item()  # Acción determinista
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()
                hidden_state = (
                    torch.zeros(1, 1, 512).to(self.device),
                    torch.zeros(1, 1, 512).to(self.device),
                )

        # Cerrar
        out.release()
        env.close()

        print(f"✅ Video guardado en: {video_path}")
    
    def train(self):
        obs, _ = self.env.reset()
        num_envs = self.env.num_envs  
        print(f"🎨 Obs shape después de FrameStack: {obs.shape}")
        obs = self.random_noop_start(self.env)

        hidden_state = (
            torch.zeros(1, self.num_envs, 512).to(self.device),
            torch.zeros(1, self.num_envs, 512).to(self.device),
        )
        max_x_pos = 0
        self.episode_length = 0
        prev_x_pos = np.zeros(num_envs, dtype=np.float32)
        max_x_pos = np.zeros(num_envs, dtype=np.float32)

        total_updates = self.total_timesteps // self.num_steps

        for update in tqdm(range(1, total_updates + 1), desc="📈 Entrenando PPO", unit="update"):
            self.update_start_time = time.time()
            
            progress = update / total_updates 
            self.ent_coef = max(0.02, 0.03 * (1 - progress * 0.3))

            for step in range(self.num_steps):
                
                self.episode_length += 1
                tensor_obs = self.preprocess_obs(obs)

                # if step % 100 == 0:
                #   print("🔍 Visualizando activaciones CNN...")
                #   self.visualize_activations(tensor_obs)

                self.current_episode_lengths += 1


                logits, value, hidden_state = self.model(tensor_obs, hidden_state)

                # 🔥 Detach el estado oculto para evitar backprop de grafos anteriores
                if hidden_state is not None:
                    hidden_state = (hidden_state[0].detach(), hidden_state[1].detach())
                dist = Categorical(logits=logits)

                # Exploración
                epsilon = max(0.05, 0.1 * (1 - progress**2))
                #topk_bias = 0.05 + 0.95 * (1 / (1 + math.exp(5 * (progress - 0.7))))
                self.writer.add_scalar("loss/epsilon", epsilon, self.update_count)
                
                if torch.rand(1).item() < epsilon:
                    # Exploración más agresiva: completamente aleatoria
                    actions = [self.env.single_action_space.sample() for _ in range(num_envs)]
                    logprobs = np.zeros(num_envs, dtype=np.float32)  # Exploración no tiene logprobs útiles
                    values = torch.zeros(num_envs, device=self.device)  # Ni valores útiles
                else:
                    # Explotación: usar la política aprendida
                    actions_tensor = dist.sample()
                    actions = actions_tensor.cpu().numpy()
                    logprobs = dist.log_prob(actions_tensor).detach().cpu().numpy()
                    values = value.squeeze(-1).detach().cpu().numpy()
                
                actions = np.array(actions, dtype=np.int32) 
                next_obs, rewards, terminated, truncated, infos = self.env.step(actions.flatten().tolist())
                dones = np.logical_or(terminated, truncated)
                rewards = self.reward_normalizer.normalize(rewards)

                try:
                    infos_list = infos.get("env", infos) if isinstance(infos, dict) else list(infos)
                except Exception:
                    raise TypeError(f"Formato inesperado en infos: {type(infos)}")

                if isinstance(infos, dict):
                    curr_x_pos = np.array(infos.get("x_pos", infos.get("x_scroll", [0]*self.num_envs)))
                else:
                    curr_x_pos = np.array([info.get("x_pos", info.get("x_scroll", 0)) for info in infos])

                curr_x_pos = curr_x_pos[:self.num_envs]

                self.current_max_x_positions = np.maximum(self.current_max_x_positions, curr_x_pos)

                # rewards += np.clip((curr_x_pos - prev_x_pos) / 10.0, -1.0, 1.0)

                self.action_counter.update(actions.tolist())

                # for i in range(num_envs):
                #     if int(actions[i]) in (2, 3, 4):  # saltar o saltar+avanzar
                #         rewards[i] += 0.005
                #         if curr_x_pos[i] > prev_x_pos[i]:
                #             rewards[i] += (curr_x_pos[i] - prev_x_pos[i]) * 0.001
                #         else:
                #             rewards[i] -= 0.01

                #     rewards[i] += np.clip((curr_x_pos[i] - prev_x_pos[i]) / 10.0, -1.0, 1.0)

                # Penaliza quedarse quieto
                # if curr_x_pos[i] <= prev_x_pos[i]:
                #     rewards[i] -= 0.005
                
                # if curr_x_pos[i] > max_x_pos[i]:
                #     max_x_pos[i] = curr_x_pos[i]
                #     rewards[i] += 0.005

                self.current_episode_rewards += rewards

                prev_x_pos = curr_x_pos.copy()
 
                self.obs_buf[step] = obs
                self.actions_buf[step] = actions
                self.logprobs_buf[step] = logprobs
                self.rewards_buf[step] = rewards
                self.dones_buf[step] = dones

                if isinstance(values, torch.Tensor):
                    self.values_buf[step] = values.detach().cpu().numpy()
                else:
                    self.values_buf[step] = values

                obs = next_obs

                for i in range(num_envs):
                    if dones[i]:
                        obs_frame = self.obs_buf[step, i]  # (stack, H, W, C)

                        # Convierte de Torch Tensor si es necesario
                        if isinstance(obs_frame, torch.Tensor):
                            obs_frame = obs_frame.cpu().numpy()

                        # Si está normalizado (0..1), escala a 0..255
                        if obs_frame.max() <= 1.0:
                            obs_frame = (obs_frame * 255).astype(np.uint8)

                        # Si está en formato (stack, H, W, C), usa solo el último frame del stack
                        if obs_frame.ndim == 4:
                            obs_frame = obs_frame[-1]  # toma el último frame (H, W, C)

                        # Convierte de RGB a BGR para OpenCV
                        obs_frame_bgr = cv2.cvtColor(obs_frame, cv2.COLOR_RGB2BGR)

                        # Guarda imagen
                        snapshot_dir = "snapshots"
                        os.makedirs(snapshot_dir, exist_ok=True)
                        snapshot_path = os.path.join(snapshot_dir, f"update_{self.update_count}_env_{i}.png")
                        cv2.imwrite(snapshot_path, obs_frame_bgr)
                        print(f"📸 Snapshot guardado: {snapshot_path}")
                        rewards[i] -= 1.0

                        #if curr_x_pos[i] > 800:  rewards[i] += 0.1
                        #if curr_x_pos[i] > 1000: rewards[i] += 0.2
                        #if curr_x_pos[i] > 1200: rewards[i] += 0.3
                        #if curr_x_pos[i] > 1500: rewards[i] += 0.4
                        #if curr_x_pos[i] > 2000: rewards[i] += 0.5
                        #if curr_x_pos[i] > 2500: rewards[i] += 0.6
                        #if curr_x_pos[i] > 3000: rewards[i] += 0.7

                        self.update_episode_lengths.append(self.current_episode_lengths[i])
                        self.update_episode_rewards.append(self.current_episode_rewards[i])
                        self.current_episode_rewards[i] = 0.0  # Reset SOLO para ese entorno
                        self.current_episode_lengths[i] = 0
                        self.update_deaths += 1
                        self.episode_length = 0
                        prev_x_pos[i] = 0.0

                        # Reinicia solo el entorno i
                        obs_all, _ = self.env.reset()
                        next_obs[i] = obs_all[i]

                        # # 🔥 Warm-up más inteligente
                        # random_warmup_steps = 15  # ajusta el número de pasos aleatorios
                        # for _ in range(random_warmup_steps):
                        #     # 📜 Escoge acción con mayor probabilidad de avanzar
                        #     if random.random() < 0.6:
                        #         # 60% de las veces: avanzar (run right o run+jump)
                        #         random_action = random.choice([1, 2, 3, 4])  # avanza o avanza saltando
                        #     elif random.random() < 0.3:
                        #         # 30% de las veces: saltar en el sitio
                        #         random_action = 5  # salto
                        #     else:
                        #         # 10% de las veces: NOOP
                        #         random_action = 0

                        #     obs_all, _, terminated_warmup, truncated_warmup, _ = self.env.step([random_action]*self.num_envs)
                        #     if terminated_warmup[i] or truncated_warmup[i]:
                        #         # Reinicia solo el entorno que murió
                        #         obs_all, _ = self.env.reset()

                        # # ⚡️ Salimos del warm-up con un estado más “rico”
                        # next_obs[i] = obs_all[i]

            # GAE y PPO
            avg_x_pos = np.mean(self.current_max_x_positions)
            self.update_max_x_positions.append(avg_x_pos)
            self.current_max_x_positions[:] = 0

            self._update_policy(obs, hidden_state, update)

        # Guardar modelo
        torch.save(self.model.state_dict(), os.path.join(self.model_dir, self.model_name))
        print(f"📦 Modelo guardado: {self.model_name}")

        self.writer.close() 
        self.env.close()
        print("✅ Entrenamiento CleanRL completado.")
 

    def _update_policy(self, obs, hidden_state, update):
        """Actualiza la política PPO para multi-env"""
        b, n = self.num_steps, self.num_envs  # steps y entornos

        obs_array = np.array(self.obs_buf, dtype=np.float32)  # (steps, envs, stack, H, W, 1) o (steps, envs, stack, H, W)

        # Quita canal extra si está presente
        if obs_array.ndim == 6 and obs_array.shape[-1] == 1:
            obs_array = obs_array.squeeze(-1)  # (steps, envs, stack, H, W)

        # 🔥 Convierte a tensor y normaliza
        tensor_obs = torch.tensor(obs_array, dtype=torch.float32) / 255.0  # (steps, envs, stack, H, W, 1) o (steps, envs, stack, H, W)

        if tensor_obs.ndim == 6 and tensor_obs.shape[-1] == 1:
            tensor_obs = tensor_obs.squeeze(-1)  # quita el canal extra (steps, envs, stack, H, W)

        tensor_obs = tensor_obs.to(self.device)

        # Calcula next_value para todos los envs
        tensor_next_obs = self.preprocess_obs(obs)  # shape: (num_envs, C, H, W)
        _, next_values, _ = self.model(tensor_next_obs.unsqueeze(0), hidden_state)
        next_values = next_values.squeeze(0).detach().cpu().numpy()
        
        all_returns = [] # Calcula GAE

        total = sum(self.action_counter.values())
        for action_id, count in self.action_counter.items():
            proportion = count / total
            self.writer.add_scalar(f"actions/action_{action_id}_proportion", proportion, self.update_count)

        for env_idx in range(self.num_envs):
            env_rewards = self.rewards_buf[:, env_idx]  # shape: (steps,)
            env_dones = self.dones_buf[:, env_idx]      # shape: (steps,)
            env_values = self.values_buf[:, env_idx]    # shape: (steps,)
            next_value = next_values[env_idx]           # escalar

            env_returns = self.compute_gae(env_rewards, env_dones, env_values, next_value)
            all_returns.append(env_returns)

        # Stack para formar shape (steps, num_envs)
        returns = np.stack(all_returns, axis=1)

        advantages = returns - self.values_buf
        self.writer.add_scalar("charts/advantage_mean", advantages.mean(), self.update_count)
        self.writer.add_scalar("charts/advantage_std", advantages.std(), self.update_count)
        
        # Prepara tensores para PPO
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        values = torch.tensor(self.values_buf, dtype=torch.float32).to(self.device)  # (128, 8)values = torch.tensor(self.values_buf.reshape(-1), dtype=torch.float32).to(self.device)
        logprobs = torch.tensor(self.logprobs_buf, dtype=torch.float32).to(self.device)  # (128, 8)
        actions = torch.tensor(self.actions_buf, dtype=torch.int64).to(self.device)  # (num_steps, num_envs)
        logprobs_old = logprobs.clone()

        initial_kl = 1.5
        final_kl = 0.02
        total_updates = self.total_timesteps // self.num_steps
        progress = update / total_updates
        target_kl = max(0.05, initial_kl * (1 - progress) + final_kl * progress)
        warmup_updates = 20

        for epoch in range(self.update_epochs):

            expanded_hidden_state = (
                torch.zeros(1, self.num_envs, 512).to(self.device),
                torch.zeros(1, self.num_envs, 512).to(self.device),
            )

            logits, value, hidden_state = self.model(tensor_obs, expanded_hidden_state)

            # Detach hidden_state
            hidden_state = (hidden_state[0].detach(), hidden_state[1].detach())

            dist = Categorical(logits=logits)
            entropy = dist.entropy().mean()
            newlogprob = dist.log_prob(actions.view(-1))
            if logprobs.shape != newlogprob.shape:
                logprobs = logprobs.view(-1).expand_as(newlogprob)
            ratio = (newlogprob - logprobs).exp()

            if returns.shape != ratio.shape:
                returns = returns.view(-1).expand_as(ratio)
                values = values.view(-1).expand_as(ratio)

            pg_loss1 = -ratio * (returns - values).detach()
            pg_loss2 = -torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef) * (returns - values).detach()
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()
            v_loss = ((returns - value.squeeze()) ** 2).mean()
            loss = pg_loss + self.vf_coef * v_loss - self.ent_coef * entropy

            # Save losses for logging
            losses = {
                "entropy": entropy.item(),
                "policy": pg_loss.item(),
                "value": v_loss.item(),
                "total": loss.item(),
            }

            self.optimizer.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            total_grad_norm = torch.norm(torch.stack([
                torch.norm(p.grad.detach(), 2)
                for p in self.model.parameters()
                if p.grad is not None
            ]), 2)

            self.writer.add_scalar("gradients/total_grad_norm", total_grad_norm, self.update_count)
            self.optimizer.step()

            # Early stopping
            if update > warmup_updates:
                if logprobs_old.shape != newlogprob.shape:
                    logprobs_old = logprobs_old.view(-1).expand_as(newlogprob)
                approx_kl = (logprobs_old - newlogprob).mean()
                if approx_kl > target_kl:
                    print(f"🚨 Early stopping at epoch {epoch} due to KL={approx_kl:.4f} (target={target_kl:.4f})")
                    break

        elapsed = time.time() - self.update_start_time
        fps = self.num_steps * self.num_envs / elapsed

        # Métricas promedio
        if self.update_episode_rewards:
            avg_reward = sum(self.update_episode_rewards) / len(self.update_episode_rewards)
            avg_length = sum(self.update_episode_lengths) / len(self.update_episode_lengths)
            max_x_pos = max(self.update_max_x_positions)

            self.writer.add_scalar("charts/episode_reward", avg_reward, self.update_count)
            self.writer.add_scalar("charts/episode_length", avg_length, self.update_count)
            self.writer.add_scalar("charts/x_pos", max_x_pos, self.update_count)

        # Deaths normalizadas
        death_rate = self.update_deaths / (len(self.update_episode_rewards) or 1)
        self.writer.add_scalar("charts/deaths", death_rate, self.update_count)

        # Rendimiento
        self.writer.add_scalar("charts/fps", fps, self.update_count)
        self.writer.add_scalar("charts/update_time_sec", elapsed, self.update_count)

        # Pérdidas (si existen)
        if losses:
            self.writer.add_scalar("loss/entropy", losses["entropy"], self.update_count)
            self.writer.add_scalar("loss/policy", losses["policy"], self.update_count)
            self.writer.add_scalar("loss/value", losses["value"], self.update_count)
            self.writer.add_scalar("loss/total", losses["total"], self.update_count)

        # Reset para el próximo update
        self.update_episode_rewards.clear()
        self.update_episode_lengths.clear()
        self.update_max_x_positions.clear()
        self.update_deaths = 0
        self.update_count += 1
        self.update_start_time = time.time()