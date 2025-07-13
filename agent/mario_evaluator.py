import torch
import numpy as np
import cv2
import os
import time
from agent.mario_lstm import MarioConvLSTM
from agent.env_builder import make_mario_env

class MarioRLEvaluator:
    def __init__(self, video_dir="videos", fps=30):
        self.env = make_mario_env(seed=42)
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.video_dir = video_dir
        self.fps = fps

        os.makedirs(self.video_dir, exist_ok=True)

    def preprocess_obs(self, obs):
        """Convierte (stack, H, W, C) -> (1, stack*C, H, W) normalizado"""
        stack, H, W, C = obs.shape
        obs = torch.tensor(obs, dtype=torch.float32)
        obs = obs.permute(0, 3, 1, 2).reshape(1, -1, H, W)
        return obs.to(self.device) / 255.0

    def k_greedy_sample(self, probs, k=3, temperature=1.0):
        """
        Selecciona aleatoriamente una acción entre las top-k más probables,
        aplicando suavizado con temperatura para evitar políticas ultra deterministas.
        """
        top_k_idx = probs.argsort()[::-1][:k]
        top_k_probs = probs[top_k_idx]

        # 🔥 Aplica temperatura para suavizar
        logits = np.log(top_k_probs + 1e-8) / temperature
        soft_probs = np.exp(logits)
        soft_probs /= soft_probs.sum()

        return np.random.choice(top_k_idx, p=soft_probs)

    def load_model(self, model_path, obs_shape, n_actions):
        """Carga siempre MarioConvLSTM"""
        lstm_hidden_size = 512
        print("🔄 Cargando modelo como MarioConvLSTM...")
        model = MarioConvLSTM(obs_shape, lstm_hidden_size, n_actions).to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        print("✅ Modelo LSTM cargado correctamente")
        return model, lstm_hidden_size

    def evaluate(self, model_path, episodes=1):
        obs_shape = self.env.observation_space.shape
        stack, H, W, C = obs_shape
        channels = stack * C

        # Cargar modelo
        model, lstm_hidden_size = self.load_model(model_path, (channels, H, W), self.env.action_space.n)

        # Nombre único para el video
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        video_path = os.path.join(self.video_dir, f"mario_eval_{timestamp}.mp4")

        # Inicializa VideoWriter
        obs, _ = self.env.reset()
        frame = self.env.render()
        frame_height, frame_width = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(video_path, fourcc, self.fps, (frame_width, frame_height))

        print(f"🎮 Evaluando modelo durante {episodes} episodios...")

        for ep in range(episodes):
            obs, _ = self.env.reset()
            done = False
            total_reward = 0

            # 🔥 Inicializa el hidden_state explícitamente
            hidden_state = (
                torch.zeros(1, 1, lstm_hidden_size).to(self.device),
                torch.zeros(1, 1, lstm_hidden_size).to(self.device)
            )

            while not done:
                tensor_obs = self.preprocess_obs(obs).unsqueeze(1)
                with torch.no_grad():
                    logits, _, hidden_state = model(tensor_obs, hidden_state)
                    hidden_state = (hidden_state[0].detach(), hidden_state[1].detach())

                    probs = torch.softmax(logits, dim=1).cpu().numpy()
                    action = self.k_greedy_sample(probs[0], k=3)  # 🔥 Top-k sampling

                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                total_reward += reward

                # Graba frame en el video
                frame = self.env.render()
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)

            print(f"🏁 Episodio {ep+1}: Total reward = {total_reward}")

        out.release()
        self.env.close()
        print(f"🎥 Video guardado en: {video_path}")