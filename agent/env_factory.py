import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from gym.wrappers import ResizeObservation, GrayScaleObservation, FrameStack
from gym.spaces import Box

import numpy as np
class MarioEnvFactory:
    def __init__(self, world="SuperMarioBros-v0", actions=None, render=False):
        self.world = world
        self.actions = actions or SIMPLE_MOVEMENT
        self.render = render

    def make(self):
        def _init():
            # Crear el entorno con compatibilidad Gymnasium
            env = gym_super_mario_bros.make(self.world, apply_api_compatibility=True)

            # Limitar el conjunto de acciones
            env = JoypadSpace(env, self.actions)

            env = ResizeObservation(env, shape=(80, 75))

            # Preprocesado de la imagen
            env = GrayScaleObservation(env, keep_dim=False)  # Pasa a escala de grises
            env = FrameStack(env, num_stack=4)            # Apila 4 frames
            orig_space = env.observation_space
            env.observation_space = Box(
                low=0.0,
                high=1.0,
                shape=orig_space.shape,
                dtype=np.float32
            )

            # Parchear reset para mantener la tupla (obs, info)
            original_reset = env.reset
            def reset(*args, **kwargs):
                # Eliminar parámetros no soportados
                kwargs.pop("seed", None)
                kwargs.pop("options", None)
                obs_info = original_reset(*args, **kwargs)
                # Asegurar formato (obs, info)
                if isinstance(obs_info, tuple) and len(obs_info) == 2:
                    return obs_info
                return obs_info, {}
            env.reset = reset

            return env

        return _init