# Super Mario Bros con PPO: Modelado y Resultados

Este proyecto implementa un agente de Aprendizaje por Refuerzo Profundo (DRL) usando Proximal Policy Optimization (PPO) para aprender a jugar *Super Mario Bros*. Aprovecha un entorno vectorizado para acelerar el entrenamiento y técnicas avanzadas para estabilizar el aprendizaje.

---

## 1. Modelado del Problema

El entrenamiento del agente para jugar *Super Mario Bros* se formula como un **Proceso de Decisión de Markov (MDP)**, definido como un cuádruple \((S, A, P, R)\):

- **Espacio de Estados (****\(S\)****)**: Cada estado es un stack de \(n\) imágenes consecutivas del entorno, preprocesadas a tamaño reducido (84x84 píxeles) y normalizadas para capturar la dinámica temporal:

  $$
  s_t = \{I_{t-3}, I_{t-2}, I_{t-1}, I_t\}
  $$

  Este enfoque permite al agente inferir velocidades y trayectorias en un entorno parcialmente observable.

- **Espacio de Acciones (****\(A\)****)**: Conjunto discreto de combinaciones de botones del mando: moverse a la derecha/izquierda, saltar, disparar. El espacio de acción se simplifica para facilitar la convergencia.

- **Dinámica de Transición (****\(P\)****)**: Parcialmente determinista: las acciones del agente interactúan con la física del juego y el comportamiento limitado de los enemigos.

- **Función de Recompensa (****\(R\)****)**:

  - Recompensa positiva por avance horizontal: \(+0.01 * \Delta x\)
  - Bonificación al alcanzar la bandera: \(+5.0\)
  - Penalización por retroceso o muerte: \(-1.0\)

- **Horizonte Temporal**: Episodios de longitud variable, concluyen cuando el agente muere o completa un nivel.

---

## 2. Solución Propuesta

### 2.1 ¿Por qué Deep Reinforcement Learning (DRL)?

- **Alta dimensionalidad**: las entradas son imágenes que requieren CNNs para la extracción de características.
- **Recompensas escasas** (*sparse rewards*): dificultan el aprendizaje con métodos tradicionales.
- **Ambiente parcialmente observable**: obliga a considerar el historial de frames.

✅ **DRL** permite:

- Procesar datos visuales complejos.
- Generalizar a nuevos escenarios.
- Aprender políticas en espacios de acción no triviales.

📖 *Sutton y Barto* (2018) recomiendan DRL en entornos con estas características [1].

---

### 2.2 ¿Por qué Proximal Policy Optimization (PPO) frente a DQN?

**DQN (Deep Q-Network)**:

- ✅ Eficiente en entornos pequeños como *Pong*.
- ❌ Limitaciones en *Mario Bros*:
  - Sobreestimación de valores Q.
  - Exploración pobre en espacios grandes.
  - Requiere técnicas adicionales (Double DQN, Prioritized Replay) que complican su uso.

📄 *Smith, 2025* muestra que PPO supera a DQN en *Super Mario Bros* por su manejo estable de políticas estocásticas [2].

---

### 🧪 Ventajas de PPO

1. **Clipping de actualizaciones** para evitar políticas divergentes.
2. **Política estocástica** que favorece la exploración.
3. **Simetría actor-crítico** para aprender simultáneamente la política y el valor.
4. **Fácil tuning** comparado con TRPO y otros métodos.

---

## 3. Arquitectura de la Aplicación

### 3.1 Red Neuronal Convolucional (CNN)

El agente implementa una arquitectura **actor-crítico compartida** que combina un extractor convolucional y dos cabezas densas:

```
Input: (4, 84, 84)  # Stack de 4 frames

Conv2D(4, 32, kernel_size=8, stride=4)  → ReLU
Conv2D(32, 64, kernel_size=4, stride=2) → ReLU
Conv2D(64, 64, kernel_size=3, stride=1) → ReLU
Flatten
FC(3136, 512)                           → ReLU

Policy head: FC(512, n_actions)         # Logits para distribución categórica
Value head:  FC(512, 1)                  # Escalar para estimar V(s)
```

**Detalles técnicos:**

- **Stack de frames**: captura la dinámica temporal.
- **Activaciones ReLU**: mitigan gradientes desvanecidos.
- **Cabezas separadas**: PPO requiere la política \(\pi_\theta(a|s)\) y una estimación del valor \(V_\phi(s)\).

---

### 3.2 Ciclo de Entrenamiento

1. **Rollout**:
   - Ejecuta \(\pi_\theta\) durante `num_steps` en `num_envs` entornos vectorizados.
2. **Estimación de ventajas (GAE)**:
   $$
   \hat{A}_t = \delta_t + (\gamma\lambda)\delta_{t+1} + (\gamma\lambda)^2\delta_{t+2} + \dots
   $$
3. **Actualización PPO**:
   $$
   L(\theta) = \mathbb{E}\left[\min\left(r_t(\theta)\hat{A}_t, clip(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]
   $$

---

### 3.3 Hiperparámetros

| Parámetro         | Valor                 | Descripción                                               |
| ----------------- | --------------------- | --------------------------------------------------------- |
| `total_timesteps` | 1,000,000             | Número total de pasos de entrenamiento.                   |
| `lr`              | \$2.5 \cdot 10^{-4}\$ | *Learning rate* inicial, decae con los updates.           |
| \$\gamma\$        | 0.99                  | Factor de descuento para recompensas futuras.             |
| \$\lambda\$ (GAE) | 0.95                  | Parámetro \$\lambda\$ en GAE para balance sesgo/varianza. |
| `clip_coef`       | 0.2                   | Límite de *clipping* en el ratio PPO.                     |
| `num_envs`        | 8                     | Número de entornos paralelos para colectar experiencias.  |
| `num_steps`       | 128                   | Pasos por rollout antes de actualizar la política.        |

---

### 3.4 Técnicas Adicionales

#### 1. Normalización de Recompensas

- Ajusta la media y varianza acumuladas para estabilizar el entrenamiento.

#### 2. Annealing de Entropía

- Reduce gradualmente el término de entropía para pasar de exploración a explotación.

#### 3. KL Penalty Adaptativo

- Ajusta dinámicamente \(\beta_{KL}\) para controlar la divergencia entre políticas.

#### 4. Exploración Epsilon-Greedy

- Introduce una \(\epsilon\) decreciente para evitar estancamientos.

#### 5. Entornos Vectorizados

- Uso de 8 entornos paralelos para acelerar la recolección de datos.

---

## 📚 Referencias

[1] R. S. Sutton and A. G. Barto, *Reinforcement Learning: An Introduction*, MIT Press, 2018. [2] J. Smith, "Comparative analysis of DQN and PPO on Super Mario Bros.", Stanford CS224R Report, 2025. [3] J. Schulman et al., "Proximal Policy Optimization Algorithms", arXiv:1707.06347, 2017.

