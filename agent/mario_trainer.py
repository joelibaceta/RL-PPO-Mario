# agent/mario_trainer.py

import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from agent.mario_cnn import MarioCNN


class MarioRLTrainer:
    """
    Trainer for PPO agent in Super Mario Bros.

    Allows custom environment factories and training configurations.
    """

    def __init__(self, env_factory, log_dir="data/logs", model_dir="data/models", n_envs=4):
        """
        Initialize the trainer.

        :param env_factory: Factory function to create environments.
        :param log_dir: Directory for TensorBoard logs.
        :param model_dir: Directory for saved models.
        :param n_envs: Number of parallel environments.
        """
        self.env_factory = env_factory
        self.log_dir = log_dir
        self.model_dir = model_dir
        self.n_envs = n_envs

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

    def train(self, total_timesteps=1_000_000):
        """
        Train PPO agent with checkpointing and metrics.

        :param total_timesteps: Total training timesteps.
        """
        print("[INFO] Creating vectorized environments...")
        env = DummyVecEnv([self.env_factory.make() for _ in range(self.n_envs)])
        env = VecMonitor(env)

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        policy_kwargs = dict(
            features_extractor_class=MarioCNN,
            features_extractor_kwargs=dict(features_dim=512),
        )

        print("[INFO] Initializing PPO model...")
        model = PPO(
            policy="CnnPolicy",
            env=env,
            verbose=1,
            tensorboard_log=self.log_dir,
            device=device,
            n_steps=4096,            # pasos más largos para aprender patrones temporales
            batch_size=1024,         # batches grandes = actualizaciones más estables
            learning_rate=1e-4,      # más bajo para que no sobrescriba conocimientos útiles
            gamma=0.99,              # valora recompensas futuras (no solo avanzar)
            gae_lambda=0.95,         # suaviza ventajas, mejor generalización
            clip_range=0.2,          # actualizaciones moderadas
            ent_coef=0.1,            # más exploración (saltará y probará combinaciones)
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(
                features_extractor_class=MarioCNN,
                features_extractor_kwargs=dict(features_dim=512),
                ortho_init=False,
                activation_fn=torch.nn.ReLU
            )
        )

        checkpoint = CheckpointCallback(
            save_freq=100_000,
            save_path=self.model_dir,
            name_prefix="mario_ppo",
        )

        eval_env = DummyVecEnv([self.env_factory.make()])
        eval_env = VecMonitor(eval_env)
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=self.model_dir,
            log_path=self.log_dir,
            eval_freq=50_000,
            deterministic=True,
            render=False,
            verbose=1,
        )

        print(f"[INFO] Starting training for {total_timesteps} timesteps...")
        model.learn(total_timesteps=total_timesteps, callback=[checkpoint, eval_callback])

        final_path = os.path.join(self.model_dir, "mario_ppo_final")
        model.save(final_path)
        print(f"[INFO] Training completed. Model saved at {final_path}")