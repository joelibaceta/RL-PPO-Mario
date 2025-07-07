# cli.py
import argparse
from agent.env_factory import MarioEnvFactory
from agent.mario_trainer import MarioRLTrainer
from agent.mario_evaluator import MarioRLEvaluator


def main():
    parser = argparse.ArgumentParser("Mario PPO CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Train PPO agent")
    train_parser.add_argument("--timesteps", type=int, default=1_000_000)
    train_parser.add_argument("--n-envs", type=int, default=4)

    # Evaluate subcommand
    eval_parser = subparsers.add_parser("eval", help="Evaluate PPO agent")
    eval_parser.add_argument("--model", type=str, required=True)
    eval_parser.add_argument("--episodes", type=int, default=5)
    eval_parser.add_argument("--render", action="store_true")

    args = parser.parse_args()
    factory = MarioEnvFactory()

    if args.command == "train":
        trainer = MarioRLTrainer(env_factory=factory, n_envs=args.n_envs)
        trainer.train(total_timesteps=args.timesteps)
    elif args.command == "eval":
        evaluator = MarioRLEvaluator(env_factory=factory, render=args.render)
        evaluator.evaluate(model_path=args.model, episodes=args.episodes)


if __name__ == "__main__":
    main()