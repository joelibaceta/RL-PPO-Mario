# agent/mario_evaluator.py

import os
import numpy as np
import cv2
from stable_baselines3 import PPO


class MarioRLEvaluator:
    """
    Evaluator for trained PPO agents in Super Mario Bros.
    """

    def __init__(self, env_factory, video_dir="data/videos", render=False):
        """
        Initialize the evaluator.

        :param env_factory: Factory to create environments.
        :param video_dir: Directory for saving evaluation videos.
        :param render: Whether to render the environment in a window.
        """
        self.env_factory = env_factory
        self.video_dir = video_dir
        self.render = render
        os.makedirs(self.video_dir, exist_ok=True)

    def evaluate(self, model_path, episodes=5, record_video=True):
        """
        Evaluate PPO model and optionally record video.

        :param model_path: Path to trained PPO model.
        :param episodes: Number of episodes to run.
        :param record_video: Whether to record a video.
        """
        print(f"[INFO] Loading model from {model_path}")
        model = PPO.load(model_path)

        # Create environment with render_mode preset for video
        if record_video:
            base_factory = self.env_factory.make_base_factory(render_mode="rgb_array")
            env_model = self.env_factory.preprocess_pipeline.build(base_factory)()
        else:
            env_model = self.env_factory.make()()

        if record_video:
            video_path = os.path.join(self.video_dir, "mario_eval.mp4")
            out = cv2.VideoWriter(
                video_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (256, 240)
            )
            print(f"[INFO] Recording evaluation video to {video_path}")

        for ep in range(1, episodes + 1):
            obs, _ = env_model.reset()
            done = False
            total_reward = 0

            while not done:
                # Predict action deterministically
                action, _ = model.predict(obs, deterministic=True)
                if isinstance(action, np.ndarray):  # Fix for vectorized obs
                    action = int(action)

                obs, reward, terminated, truncated, _ = env_model.step(action)
                done = terminated or truncated
                total_reward += reward

                # Get frame for video if enabled
                if record_video:
                    frame = env_model.render()  # No mode needed
                    out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

                # Render live if requested
                if self.render and not record_video:
                    env_model.render()

            print(f"[INFO] Episode {ep}: Total reward = {total_reward}")

        if record_video:
            out.release()
            print("[INFO] Video saved successfully ✅")

        env_model.close()