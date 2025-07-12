import gym
import numpy as np
import cv2
from gym.spaces import Box

class FrameStack(gym.Wrapper):
    """
    Apila k frames en un solo tensor de profundidad k.
    Maneja APIs legacy y nueva de reset y step.
    """
    def __init__(self, env, k: int = 4):
        super().__init__(env)
        self.k = k
        shp = env.observation_space.shape
        self.observation_space = Box(
            low=0, high=255,
            shape=(shp[0], shp[1], shp[2] * k),
            dtype=np.uint8
        )
        self.frames = []

    def reset(self, **kwargs):
        # Llamar al reset del entorno base, que puede retornar obs o (obs, info)
        result = self.env.reset(**kwargs)
        # Extraer solo la observación
        obs = result[0] if isinstance(result, tuple) else result
        # Inicializar buffer con k copias del frame inicial
        self.frames = [obs for _ in range(self.k)]
        # Obtener observación apilada
        stacked = self._get_obs()
        # Retornar (obs, info) para compatibilidad con VecEnv reset
        return stacked, {}

    def step(self, action):
        # Espera API legacy: (obs, reward, done, info)
        result = self.env.step(action)
        if len(result) == 4:
            obs, reward, done, info = result
        else:
            # API nueva: (obs, reward, terminated, truncated, info)
            obs, reward, terminated, truncated, info = result
            done = bool(terminated or truncated)
        # Actualizar buffer de frames
        self.frames.pop(0)
        self.frames.append(obs)
        # Retornar tupla legacy: (obs, reward, done, info)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        return np.concatenate(self.frames, axis=2)