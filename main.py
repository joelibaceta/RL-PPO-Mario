import argparse
import os
from agent.mario_trainer import MarioRLTrainer

# 📦 Configuración de rutas
MODEL_DIR = "data/models"
LOG_DIR   = "data/logs"
VIDEO_DIR = "data/videos"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR,   exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)

def main():
    parser = argparse.ArgumentParser(description="Entrenador/Evaluador PPO para Super Mario Bros")
    parser.add_argument("--mode", choices=["train", "eval"], required=True, help="Modo: train o eval")
    parser.add_argument("--timesteps", type=int, default=1_000_000, help="Tiempos de entrenamiento (solo train)")
    parser.add_argument("--model", type=str, default=os.path.join(MODEL_DIR, "mario_ppo_final"),
                        help="Ruta al modelo para evaluación (solo eval)")
    parser.add_argument("--render", action="store_true",
                        help="Activar renderizado en modo debug (solo eval o n_envs=1)")
    args = parser.parse_args()

    # Si renderizas o evaluas, sólo 1 entorno; si no, 8 procesos en paralelo
    n_envs = 1 if args.render or args.mode == "eval" else 8

    trainer = MarioRLTrainer(
        world="SuperMarioBros-v3",
        n_envs=n_envs,
        log_dir=LOG_DIR,
        model_dir=MODEL_DIR,
        video_dir=VIDEO_DIR,
        render=args.render,
    )

    if args.mode == "train":
        print("[INFO] Iniciando entrenamiento…")
        trainer.train(total_timesteps=args.timesteps)
    else:
        print("[INFO] Iniciando evaluación…")
        trainer.evaluate(model_path=args.model, record_video=True)

if __name__ == "__main__":
    main()