# PPO Mario Agent 🕹️

Este proyecto implementa un agente **Proximal Policy Optimization (PPO)** para jugar *Super Mario Bros*, utilizando PyTorch y un entorno personalizado basado en `gym-super-mario-bros`. Incluye una CNN como policy network, normalización de recompensas y varios mecanismos para estabilidad del entrenamiento.

---

## 📜 Contenido

- [Arquitectura CNN](#arquitectura-cnn)
- [Entrenador PPO](#entrenador-ppo)
- [Hiperparámetros](#hiperparámetros)
- [Construcción del entorno](#construcción-del-entorno)
- [Uso](#uso)

---

## 🧠 Arquitectura CNN

La policy/value network usa una CNN ligera que procesa stacks de 4 frames para capturar información temporal.

```
Input: (4, 84, 84)

Conv2D(4, 32, kernel_size=8, stride=4) → ReLU
Conv2D(32, 64, kernel_size=4, stride=2) → ReLU
Conv2D(64, 64, kernel_size=3, stride=1) → ReLU
Flatten
FC(3136, 512) → ReLU
Output policy logits: FC(512, n_actions)
Output value: FC(512, 1)
```

- **Policy head**: genera logits para la distribución categórica sobre acciones.
- **Value head**: estima el valor del estado para ventaja.

---

## 🏋️‍♂️ Entrenador PPO

El entrenamiento sigue un ciclo:

1. **Rollout**:

   - Ejecuta la política actual durante `num_steps` en `num_envs` entornos paralelos.
   - Guarda: observaciones, acciones, logprobs, recompensas normalizadas.

2. **Ventaja (GAE)**:

   - Estima ventajas con *Generalized Advantage Estimation*.

3. **Actualización PPO**:

   - Calcula la pérdida de política con clipping: \(\min(ratio \cdot A, clip(ratio) \cdot A)\)
   - Añade pérdida de valor y entropía.
   - Early stopping o rollback si el KL-divergence supera el target.

4. **Logging**:

   - Guarda métricas en TensorBoard: reward promedio, max\_x\_pos, entropía, proporción de acciones.

---

## ⚙️ Hiperparámetros

| Parámetro         | Valor     | Descripción                                                |
| ----------------- | --------- | ---------------------------------------------------------- |
| `total_timesteps` | 1,000,000 | Total de pasos de entrenamiento.                           |
| `lr`              | 2.5e-4    | Learning rate inicial, decae en el tiempo.                 |
| `gamma`           | 0.99      | Factor de descuento de recompensas.                        |
| `gae_lambda`      | 0.98      | Mezcla bias-variance en ventajas.                          |
| `clip_coef`       | 0.2       | Límite de clipping en ratio PPO.                           |
| `entropy_coef`    | Dinámico  | Incentiva la exploración (disminuye con los updates).      |
| `epsilon`         | Dinámico  | Probabilidad de exploración aleatoria, nunca menor a 0.05. |
| `num_envs`        | 8         | Número de entornos paralelos para colectar experiencias.   |
| `num_steps`       | 128       | Pasos por rollout antes de actualizar.                     |

**Técnicas adicionales**:

- Normalización online de recompensas.
- Penalización a retrocesos (para evitar que el agente se quede bloqueado).
- Bonificaciones por milestones en X para incentivar el progreso.
- Decaimiento de learning rate.

---

## 🌱 Construcción del entorno

El entorno utiliza *wrappers* personalizados y de `gym` para preparar los datos de entrada y gestionar la interacción del agente con el juego:

- **FrameStack**: apila 4 frames para incluir información temporal.
- **GrayScaleObservation**: convierte a escala de grises para simplificar la entrada.
- **ResizeObservation**: reduce las imágenes a 84x84 píxeles para eficiencia.
- **FrameSkipWrapper**: omite frames para acelerar la simulación.
- **FrameCropWrapper**: recorta la HUD para evitar información redundante.
- **LifeResetWrapper** (opcional): reinicia el entorno al perder una vida.
- **FilterColorsWrapper** (opcional): filtra colores específicos.
- **Vector Envs (Sync/Async)**: ejecuta múltiples copias del entorno en paralelo para mejorar la eficiencia de recolección de datos.

Estos *wrappers* permiten un procesamiento más rápido y estable durante el entrenamiento.

---

## 🚀 Uso

1. Instalar dependencias:

```bash
pip install -r requirements.txt
```

2. Entrenar el agente:

```bash
python cli.py
```

3. Visualizar métricas en TensorBoard:

```bash
tensorboard --logdir=runs
```

4. Evaluar el modelo entrenado:

```bash
python evaluate.py --model ppo_mario.pth
```

---

## 📈 Métricas registradas

- `episode_reward`: Recompensa promedio por episodio.
- `max_x_pos`: Máximo progreso horizontal alcanzado.
- `avg_x_pos`: Progreso promedio por update.
- `policy_entropy`: Diversidad de acciones elegidas.
- Proporción de cada acción.

---

## 📌 Notas

- El agente usa PPO con **early stopping** y rollback para evitar colapsos.


## 💡 Sugerencias de mejoras

Progreso hasta el momento:
<center>
     <img src="figs/Progreso.gif" alt="Progreso del agente" width="300">
</center>

- Implementar una etapa de pre-entrenamiento con movimientos aleatorios para diversificar la experiencia inicial.
- Implementar una etapa sin movimiento para dejar que el entorno cambie y diversificar la experiencia.
- Probar con diferentes arquitecturas de CNN y comparar resultados (Quizas incluir una capa LSTM).
- Probar diferentes combinaciones de hiperparámetros y documentar resultados.
- Crear un script para entrenamiento múltiple con hiperparámetros variados.
- Grabar un video del mejor episodio durante el entrenamiento (con OpenCV).
- Agregar métricas adicionales: epsilon, learning rate, duración de episodios.
- Probar con el rango de acciones `RIGHT_ONLY` en lugar de `SIMPLE_MOVEMENT`.
- Incorporar recompensas progresivas por superar posiciones X (checkpoints: 1500, 2000, 3000).
- Penalizar retrocesos o quedarse quieto demasiado tiempo.
- Incluir recompensas por recolectar monedas o eliminar enemigos.
- Experimentar con un *replay buffer* para estabilizar el aprendizaje.