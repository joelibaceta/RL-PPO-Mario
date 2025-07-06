import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.callbacks import CheckpointCallback
from agent.env_factory import MarioEnvFactory


class MarioRLTrainer:
    def __init__(self,
                 world="SuperMarioBros-v3",
                 n_envs=8,
                 log_dir="data/logs",
                 model_dir="data/models",
                 video_dir="data/videos",
                 render=False):
        self.world     = world
        self.n_envs    = n_envs
        self.log_dir   = log_dir
        self.model_dir = model_dir
        self.video_dir = video_dir
        self.render    = render

        os.makedirs(self.log_dir,   exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.video_dir, exist_ok=True)

    def train(self, total_timesteps=1_000_000):
        print("[INFO] Creando entornos vectorizados de Mario…")
        factory = MarioEnvFactory(world=self.world, record=False, render=False)
        env_fns = [factory.make(rank=i) for i in range(self.n_envs)]
        env     = SubprocVecEnv(env_fns)  # paralelo real

        print("[INFO] Configurando modelo…")
        model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=self.log_dir)

        checkpoint = CheckpointCallback(
            save_freq=100_000,
            save_path=self.model_dir,
            name_prefix="mario_ppo"
        )

        print(f"[INFO] Entrenando {total_timesteps} timesteps…")
        model.learn(total_timesteps=total_timesteps, callback=checkpoint)

        final_path = os.path.join(self.model_dir, "mario_ppo_final")
        model.save(final_path)
        print(f"[INFO] Entrenamiento completo. Modelo guardado en: {final_path}")

        env.close()

    def evaluate(self,
                 model_path=None,
                 episodes=5,
                 record_video=True):
        if model_path is None:
            model_path = os.path.join(self.model_dir, "mario_ppo_final.zip")
        print(f"[INFO] Cargando modelo desde: {model_path}")

        factory = MarioEnvFactory(
            world=self.world,
            record=record_video,
            record_path=self.video_dir,
            render=self.render
        )
        env = DummyVecEnv([factory.make(rank=0)])

        if record_video:
            env = VecVideoRecorder(
                env,
                video_folder=self.video_dir,
                record_video_trigger=lambda step: step == 0,
                video_length=2000,
                name_prefix="mario_eval"
            )
            print(f"[INFO] Grabando videos en: {self.video_dir}")

        model = PPO.load(model_path)

        for ep in range(episodes):
            obs, _ = env.reset()
            done = False
            total_reward = 0
            while not done:
                action, _ = model.predict(obs)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                if self.render:
                    env.render()
                total_reward += reward
            print(f"[INFO] Episodio {ep+1}: Recompensa total = {total_reward}")

        env.close()
        print("[INFO] Evaluación finalizada.")