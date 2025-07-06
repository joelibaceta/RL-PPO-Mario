import gymnasium as gym
import gymnasium_super_mario_bros  # registra SuperMarioBros-v3
from gymnasium_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from gymnasium.wrappers import GrayScaleObservation, ResizeObservation, FrameStack, RecordVideo
import numpy as np
from gymnasium import spaces
from gymnasium.core import ObservationWrapper


class TransposeImage(ObservationWrapper):
    """Pasa de HxWxC a CxHxW para las policies de SB3."""
    def __init__(self, env):
        super().__init__(env)
        h, w, c = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(c, h, w),
            dtype=env.observation_space.dtype
        )

    def observation(self, obs):
        return np.transpose(obs, (2, 0, 1))


class MarioEnvFactory:
    """
    Crea callables para SubprocVecEnv/DummyVecEnv.
    Usa Gymnasium + gymnasium-super-mario-bros + nes-py.
    """
    def __init__(self,
                 world="SuperMarioBros-v3",
                 actions=SIMPLE_MOVEMENT,
                 record=False,
                 record_path="data/videos",
                 render=False):
        self.world       = world
        self.actions     = actions
        self.record      = record
        self.record_path = record_path
        self.render      = render

    def make(self, rank=0):
        def _init():
            # 1) Crear el entorno
            env = gym.make(self.world, render_mode="human" if self.render else None)
            # 2) Limitar acciones
            env = JoypadSpace(env, self.actions)
            # 3) Grabar video si hace falta
            if self.record:
                env = RecordVideo(env, self.record_path, name_prefix=f"env_{rank}")
            # 4) Blanco y negro + redimensionar + stack de frames
            env = GrayScaleObservation(env, keep_dim=True)
            env = ResizeObservation(env, 84)
            env = FrameStack(env, num_stack=4)
            # 5) Transponer canales al frente
            env = TransposeImage(env)
            return env
        return _init