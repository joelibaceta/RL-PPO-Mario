import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT


class SeedResetWrapper(gym.Wrapper):
    """
    Wrapper que acepta seed= en reset(), resembrando el entorno.
    """
    def reset(self, seed=None, **kwargs):
        # Si se pasa seed, fijarla para reproducibilidad
        if seed is not None:
            try:
                self.env.seed(seed)
            except Exception:
                pass
        # Ignorar otros kwargs e invocar reset base
        return self.env.reset()
