import argparse
import subprocess
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from agent.mario_ppo_trainer import PPOTrainer
from agent.mario_evaluator import MarioRLEvaluator

# Valores por defecto para hiperparámetros
default_params = {
    "learning_rate": 2.5e-4,
    "clip_coef": 0.2,
    "batch_size": 512,
    "update_epochs": 4,
    "num_envs": 8,
    "timesteps": 200_000
}


def run_experiment_subprocess(cmd, run_name):
    """
    Ejecuta un experimento como subprocesso y devuelve el resultado
    """
    print(f"🚀 Lanzando experimento: {run_name}")
    subprocess.run(cmd)


def run_benchmark(param_name, param_values, timesteps, output_dir, full_grid, max_workers):
    """
    Ejecuta benchmarks en paralelo
    """
    os.makedirs(output_dir, exist_ok=True)

    futures = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:

        if full_grid:
            import itertools
            print("⚠️ Ejecutando grid completo de hiperparámetros (puede ser muy costoso)")
            param_sweep = {
                "learning_rate": [1e-5, 5e-5, 1e-4, 2.5e-4, 5e-4],
                "clip_coef": [0.1, 0.2, 0.3, 0.4],
                "batch_size": [128, 256, 512, 1024],
                "update_epochs": [2, 4, 6, 8]
            }
            all_combos = list(itertools.product(
                param_sweep["learning_rate"],
                param_sweep["clip_coef"],
                param_sweep["batch_size"],
                param_sweep["update_epochs"]
            ))
            for lr, clip, batch, epochs in all_combos:
                run_name = f"grid_lr{lr}_clip{clip}_batch{batch}_ep{epochs}"
                model_name = f"{run_name}.pth"
                cmd = [
                    "python", "cli.py", "train",
                    "--timesteps", str(timesteps),
                    "--learning-rate", str(lr),
                    "--clip-coef", str(clip),
                    "--batch-size", str(int(batch)),
                    "--update-epochs", str(int(epochs)),
                    "--model-name", model_name,
                    "--model-dir", output_dir,
                    "--run-name", run_name
                ]
                futures.append(executor.submit(run_experiment_subprocess, cmd, run_name))
        else:
            for value in param_values:
                clean_value = int(value) if param_name in ["batch_size", "update_epochs", "num_envs"] else float(value)
                run_name = f"{param_name}_{clean_value}"
                model_name = f"{run_name}.pth"
                cmd = [
                    "python", "cli.py", "train",
                    "--timesteps", str(timesteps),
                    "--learning-rate", str(default_params["learning_rate"]),
                    "--clip-coef", str(default_params["clip_coef"]),
                    "--batch-size", str(default_params["batch_size"]),
                    "--update-epochs", str(default_params["update_epochs"]),
                    "--num-envs", str(default_params["num_envs"]),
                    "--model-name", model_name,
                    "--model-dir", output_dir,
                    "--run-name", run_name
                ]

                # Sobrescribir solo el hiperparámetro que estamos comparando
                if param_name == "learning_rate":
                    cmd[cmd.index("--learning-rate") + 1] = str(clean_value)
                elif param_name == "clip_coef":
                    cmd[cmd.index("--clip-coef") + 1] = str(clean_value)
                elif param_name == "batch_size":
                    cmd[cmd.index("--batch-size") + 1] = str(clean_value)
                elif param_name == "update_epochs":
                    cmd[cmd.index("--update-epochs") + 1] = str(clean_value)

                futures.append(executor.submit(run_experiment_subprocess, cmd, run_name))

        # Esperar a que todos terminen
        for future in as_completed(futures):
            future.result()

    print("✅ Benchmark completado.")


def main():
    parser = argparse.ArgumentParser("Mario PPO CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcomando: train
    train_parser = subparsers.add_parser("train", help="Entrena un agente PPO en Mario")
    train_parser.add_argument("--timesteps", type=int, default=200_000, help="Total de timesteps (default: 200_000)")
    train_parser.add_argument("--learning-rate", type=float, default=2.5e-4, help="Learning rate del optimizador")
    train_parser.add_argument("--clip-coef", type=float, default=0.2, help="Coeficiente de clipping PPO (default: 0.2)")
    train_parser.add_argument("--batch-size", type=int, default=512, help="Tamaño del batch (default: 512)")
    train_parser.add_argument("--num-steps", type=int, default=128, help="Número de pasos por rollout")
    train_parser.add_argument("--update-epochs", type=int, default=4, help="Epochs por actualización PPO")
    train_parser.add_argument("--seed", type=int, default=0, help="Semilla RNG para reproducibilidad")
    train_parser.add_argument("--model-name", type=str, default="ppo_mario_cleanrl.pth", help="Nombre del archivo del modelo")
    train_parser.add_argument("--model-dir", type=str, default="models_cleanrl", help="Directorio donde guardar el modelo")
    train_parser.add_argument("--num-envs", type=int, default=8, help="Número de entornos paralelos (default: 8)")
    train_parser.add_argument("--run-name", type=str, default=None, help="Nombre del experimento para TensorBoard")

    # Subcomando: eval
    eval_parser = subparsers.add_parser("eval", help="Evalúa un agente PPO entrenado")
    eval_parser.add_argument("--model-path", type=str, required=True, help="Ruta al archivo del modelo entrenado")
    eval_parser.add_argument("--episodes", type=int, default=1, help="Número de episodios para evaluar")
    eval_parser.add_argument("--render", action="store_true", help="Renderiza la pantalla durante la evaluación")

    # Subcomando: benchmark
    benchmark_parser = subparsers.add_parser("benchmark", help="Ejecuta benchmarks con distintos hiperparámetros")
    benchmark_parser.add_argument("--param", type=str, choices=["learning_rate", "clip_coef", "batch_size", "update_epochs"], required=True, help="Hiperparámetro a variar")
    benchmark_parser.add_argument("--values", type=float, nargs="+", required=True, help="Valores del hiperparámetro a comparar")
    benchmark_parser.add_argument("--timesteps", type=int, default=100_000, help="Timesteps por experimento (default: 100_000)")
    benchmark_parser.add_argument("--output-dir", type=str, default="benchmarks", help="Directorio donde guardar modelos y logs de benchmark")
    benchmark_parser.add_argument("--full-grid", action="store_true", help="Si se especifica, explora todas las combinaciones posibles")
    benchmark_parser.add_argument("--max-workers", type=int, default=2, help="Número máximo de procesos en paralelo (default: 2)")

    args = parser.parse_args()

    if args.command == "train":
        print(f"🚀 Entrenando Mario PPO con {args.timesteps} timesteps en {args.num_envs} entornos...")
        trainer = PPOTrainer(
            total_timesteps=args.timesteps,
            lr=args.learning_rate,
            clip_coef=args.clip_coef,
            batch_size=args.batch_size,
            update_epochs=args.update_epochs,
            num_envs=args.num_envs,
            run_name=args.run_name or args.model_name.replace(".pth", "")
        )
        trainer.train()
        print(f"✅ Modelo guardado en {args.model_dir}/{args.model_name}")

    elif args.command == "eval":
        print(f"🎮 Evaluando modelo: {args.model_path} por {args.episodes} episodios...")
        evaluator = MarioRLEvaluator()
        evaluator.evaluate(model_path=args.model_path, episodes=args.episodes, render=args.render)

    elif args.command == "benchmark":
        print(f"📊 Ejecutando benchmark variando {args.param} con valores {args.values}")
        run_benchmark(
            param_name=args.param,
            param_values=args.values,
            timesteps=args.timesteps,
            output_dir=args.output_dir,
            full_grid=args.full_grid,
            max_workers=args.max_workers
        )

if __name__ == "__main__":
    main()