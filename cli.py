import argparse
from agent.mario_trainer import MarioTrainer
from agent.mario_evaluator import MarioRLEvaluator
from agent.env_builder import make_mario_env

def main():
    parser = argparse.ArgumentParser("Mario PPO CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcomando: train
    train_parser = subparsers.add_parser("train", help="Entrena un agente PPO en Mario")
    train_parser.add_argument("--timesteps", type=int, default=200_000, help="Total de timesteps (default: 200_000)")
    train_parser.add_argument("--learning-rate", type=float, default=2.5e-4, help="Learning rate del optimizador")
    train_parser.add_argument("--num-steps", type=int, default=128, help="Número de pasos por rollout")
    train_parser.add_argument("--update-epochs", type=int, default=4, help="Epochs por actualización PPO")
    train_parser.add_argument("--seed", type=int, default=0, help="Semilla RNG para reproducibilidad")
    train_parser.add_argument("--model-name", type=str, default="ppo_mario_cleanrl.pth", help="Nombre del archivo del modelo")
    train_parser.add_argument("--model-dir", type=str, default="models_cleanrl", help="Directorio donde guardar el modelo")
    train_parser.add_argument("--num-envs", type=int, default=8, help="Número de entornos paralelos (default: 8)")

    # Subcomando: eval
    eval_parser = subparsers.add_parser("eval", help="Evalúa un agente PPO entrenado")
    eval_parser.add_argument("--model-path", type=str, required=True, help="Ruta al archivo del modelo entrenado")
    eval_parser.add_argument("--episodes", type=int, default=1, help="Número de episodios para evaluar")
    eval_parser.add_argument("--render", action="store_true", help="Renderiza la pantalla durante la evaluación")

    args = parser.parse_args()

    if args.command == "train":
        print(f"🚀 Entrenando Mario PPO con {args.timesteps} timesteps en {args.num_envs} entornos...")
        trainer = MarioTrainer(
            total_timesteps=args.timesteps,
            learning_rate=args.learning_rate,
            num_steps=args.num_steps,
            update_epochs=args.update_epochs,
            seed=args.seed,
            model_dir=args.model_dir,
            model_name=args.model_name,
            num_envs=args.num_envs  # 👈 PASA EL NUEVO ARGUMENTO
        )
        trainer.train()

    elif args.command == "eval":
        print(f"🎮 Evaluando modelo: {args.model_path} por {args.episodes} episodios...")
        evaluator = MarioRLEvaluator()
        evaluator.evaluate(
            model_path=args.model_path,
            episodes=args.episodes
        )

if __name__ == "__main__":
    main()