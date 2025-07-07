import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.callbacks import CheckpointCallback
from agent.env_factory import MarioEnvFactory
import torch
import numpy as np
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from gym.wrappers import (
    ResizeObservation,
    GrayScaleObservation,
    FrameStack,
    RecordVideo,
    LazyFrames,
)
from agent.mario_cnn import MarioCNN
import cv2
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
import gym_super_mario_bros
from shimmy.openai_gym_compatibility import GymV21CompatibilityV0
import gym
from gymnasium.spaces import Discrete as GymnasiumDiscrete

from stable_baselines3.common.vec_env import VecMonitor

class MarioRLTrainer:
    def __init__(
        self,
        world="SuperMarioBros-v0",
        n_envs=8,
        log_dir="data/logs",
        model_dir="data/models",
        video_dir="data/videos",
        render=False,
    ):
        self.world = world
        self.n_envs = n_envs
        self.log_dir = log_dir
        self.model_dir = model_dir
        self.video_dir = video_dir
        self.render = render

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.video_dir, exist_ok=True)

    def train(self, total_timesteps=1_000_000):
        """Entrena un agente PPO en Mario con métricas automáticas."""
        print("[INFO] Creando entorno vectorizado (DummyVecEnv)…")
        factory = MarioEnvFactory(world=self.world, render=False)
        env = DummyVecEnv([factory.make() for _ in range(self.n_envs)])
        env = VecMonitor(env, filename=os.path.join(self.log_dir, "monitor.csv"))
        device = "mps" if torch.backends.mps.is_available() else "cpu"

        policy_kwargs = dict(
            features_extractor_class=MarioCNN,
            features_extractor_kwargs=dict(features_dim=512),
        )

        print("[INFO] Configurando modelo PPO(CnnPolicy)…")
        model = PPO(
            "CnnPolicy",
            env,
            verbose=1,  # 📢 imprime métricas en consola
            tensorboard_log=self.log_dir,  # ✅ activa TensorBoard
            device=device,
            policy_kwargs=policy_kwargs,
        )

        eval_env = DummyVecEnv([factory.make() for _ in range(self.n_envs)])
        eval_env = VecMonitor(eval_env, filename=os.path.join(self.log_dir, "monitor.csv"))
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=self.model_dir,
            log_path=self.log_dir,
            eval_freq=50_000,  # evalúa cada 50k steps
            deterministic=True,
            render=False,
            verbose=1,
        )

        checkpoint = CheckpointCallback(
            save_freq=100_000,
            save_path=self.model_dir,
            name_prefix="mario_ppo",
        )

        print(f"[INFO] Entrenando {total_timesteps} timesteps…")
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint, eval_callback],
        )

        final_path = os.path.join(self.model_dir, "mario_ppo_final")
        model.save(final_path)
        print(f"[INFO] Entrenamiento terminado. Modelo en: {final_path}")

        env.close()

    def evaluate(self, model_path=None, episodes=5, record_video=True):
        """Evalúa un modelo PPO entrenado y graba un vídeo sin deformaciones."""
        if model_path is None:
            model_path = os.path.join(self.model_dir, "mario_ppo_final.zip")
        print(f"[INFO] Cargando modelo: {model_path}")

        # 🔥 Carga modelo entrenado
        model = PPO.load(model_path)

        # 🎮 Entorno para el modelo (con preprocesado igual al entrenamiento)
        env_model = gym_super_mario_bros.make(self.world, apply_api_compatibility=True)
        env_model = JoypadSpace(env_model, SIMPLE_MOVEMENT)
        env_model = ResizeObservation(env_model, shape=(80, 75))
        env_model = GrayScaleObservation(env_model, keep_dim=False)
        env_model = FrameStack(env_model, num_stack=4)

        # 🎥 Entorno para vídeo (sin preprocesado, frames originales NES)
        if record_video:
            env_video = gym_super_mario_bros.make(
                self.world, apply_api_compatibility=True, render_mode="rgb_array"
            )
            env_video = JoypadSpace(env_video, SIMPLE_MOVEMENT)

            # ⚙️ Configura OpenCV video writer
            video_path = os.path.join(self.video_dir, "mario_eval.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = 30  # NES ~60fps, pero reducimos para compatibilidad
            out = cv2.VideoWriter(video_path, fourcc, fps, (256, 240))
            print(f"[INFO] Grabando vídeo en: {video_path}")

        for ep in range(1, episodes + 1):
            obs_model, _ = env_model.reset()
            if record_video:
                obs_video, _ = env_video.reset()
                frame = env_video.render()
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            done = False
            total_reward = 0

            while not done:
                # ➡️ Predecir acción
                obs_batch = np.expand_dims(obs_model, axis=0)  # añade batch dimension
                action, _ = model.predict(obs_batch, deterministic=True)
                action = int(action)

                # ➡️ Paso en el entorno del modelo
                obs_model, reward, terminated, truncated, _ = env_model.step(action)
                done = terminated or truncated
                total_reward += reward

                # 🎥 Grabar frame original
                if record_video:
                    obs_video, _, _, _, _ = env_video.step(action)
                    frame = env_video.render()
                    out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

                # 🖥 Mostrar en pantalla si render=True
                if self.render and not record_video:
                    env_video.render()

            print(f"[INFO] Episodio {ep}: recompensa = {total_reward}")

        # 🧹 Limpiar
        env_model.close()
        if record_video:
            env_video.close()
            out.release()
            print(f"[INFO] Vídeo guardado correctamente ✅")

        print("[INFO] Evaluación finalizada.")
