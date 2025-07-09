import argparse
import os
from agent.mario_trainer import MarioRLTrainer
from agent.env_factory import MarioEnvFactory

MODEL_DIR = "data/models"
LOG_DIR = "data/logs"
VIDEO_DIR = "data/videos"
for d in (MODEL_DIR, LOG_DIR, VIDEO_DIR):
    os.makedirs(d, exist_ok=True)


def main():
    p = argparse.ArgumentParser("Mario PPO Trainer/Evaluator")
    p.add_argument("--mode", choices=["train", "eval"], required=True)
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument(
        "--model", type=str, default=os.path.join(MODEL_DIR, "mario_ppo_final.zip")
    )
    p.add_argument(
        "--render", action="store_true", help="Si true, abre ventana en evaluación"
    )
    args = p.parse_args()

    # Si render, usa solo 1 env (para que se vea bien)
    n_envs = 1 if args.render or args.mode == "eval" else 4

    # Create environment factory
    env_factory = MarioEnvFactory(
        world="SuperMarioBros-v0",
        render=args.render,
    )

    trainer = MarioRLTrainer(
        env_factory=env_factory,
        n_envs=n_envs,
        log_dir=LOG_DIR,
        model_dir=MODEL_DIR,
    )

    if args.mode == "train":
        print("[INFO] Iniciando entrenamiento…")
        trainer.train(total_timesteps=args.timesteps)
    else:
        print("[INFO] Iniciando evaluación…")
        from agent.mario_evaluator import MarioRLEvaluator
        evaluator = MarioRLEvaluator(
            env_factory=env_factory,
            video_dir=VIDEO_DIR,
            render=args.render,
        )
        evaluator.evaluate(model_path=args.model)


if __name__ == "__main__":
    main()
