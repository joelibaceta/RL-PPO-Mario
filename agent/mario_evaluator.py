import torch
import numpy as np
import cv2
import os
import time
from agent.mario_cnn import MarioCNN
from agent.env_builder import make_mario_env

class MarioRLEvaluator:
    def __init__(self, video_dir="videos", fps=30):
        self.env = make_mario_env(seed=42)
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
        self.video_dir = video_dir
        self.fps = fps

        os.makedirs(self.video_dir, exist_ok=True)

    def preprocess_obs(self, obs):
        """Convierte (stack, H, W) → (1, stack, H, W) normalizado"""
        stack, H, W = obs.shape
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)  # Añade batch dim
        return obs.to(self.device) / 255.0

 

    def k_greedy_sample(self, probs, k=3, temperature=1.0, min_prob=0.05):
        """
        Suaviza el muestreo entre top-k acciones con probabilidad mínima forzada.
        """
        if isinstance(probs, torch.Tensor):
            probs = probs.detach().cpu().numpy()
        probs = np.squeeze(probs)

        k = min(k, len(probs))
        top_k_idx = probs.argsort()[::-1][:k]
        top_k_probs = probs[top_k_idx]

        # Forzar a que ninguna acción sea > 1 - min_prob
        top_k_probs = np.clip(top_k_probs, min_prob, 1.0 - min_prob)
        top_k_probs /= top_k_probs.sum()

        # Ajustar con temperatura
        logits = np.log(top_k_probs + 1e-8) / temperature
        soft_probs = np.exp(logits - np.max(logits))
        soft_probs /= soft_probs.sum()

        print(f"Top-{k} acciones: {top_k_idx}, Softmax probabilidades: {soft_probs}")

        return np.random.choice(top_k_idx, p=soft_probs)

    def load_model(self, model_path, obs_shape, n_actions):
        """Carga modelo MarioCNN"""
        print("🔄 Cargando modelo MarioCNN...")
        model = MarioCNN(obs_shape, n_actions).to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        print("✅ Modelo CNN cargado correctamente")
        return model

    def evaluate(self, model_path, episodes=1, exploration=False):
        obs_shape = self.env.observation_space.shape  # (stack, H, W)
        stack, H, W = obs_shape

        model = self.load_model(model_path, (stack, H, W), self.env.action_space.n)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        video_path = os.path.join(self.video_dir, f"mario_cnn_eval_{timestamp}.mp4")

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
            max_x_pos = 0
            step = 0

            while not done:
                step += 1
                tensor_obs = self.preprocess_obs(obs)
                with torch.no_grad():
                    logits, _ = model(tensor_obs)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()

                noop_interval = 6

                # NOOP cada 5 pasos para simular “soltar botones”
                if step % noop_interval == 0:
                    action = 0  # NOOP
                    print(f"🔄 Paso {step}: NOOP")
                else:
                    if exploration and np.random.rand() < 0.2:
                        action = self.env.action_space.sample()
                        print(f"🎲 Paso {step}: acción aleatoria {action}")
                    else:
                        action = self.k_greedy_sample(probs[0], k=3, temperature=2)
                        print(f"🎯 Paso {step}: acción elegida {action}")
                print(f"Acción elegida: {action} (Exploración: {exploration})")

                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                total_reward += reward

                x_pos = info.get("x_pos", info.get("x_scroll", 0))
                max_x_pos = max(max_x_pos, x_pos)

                frame = self.env.render()
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)

            print(f"🏁 Episodio {ep+1}: Total reward = {total_reward}, Max X = {max_x_pos}")

        out.release()
        self.env.close()
        print(f"🎥 Video guardado en: {video_path}")