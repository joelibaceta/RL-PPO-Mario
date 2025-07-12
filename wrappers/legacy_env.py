
import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from gym import Wrapper


class LegacyEnv(Wrapper):
    """
    Wrapper de compatibilidad que normaliza reset() y step() al API legacy de Gym v0.21.
    - reset(): retorna solo observation.
    - step(): retorna (obs, reward, done, info).
    """
    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        # result puede ser obs o (obs, info)
        if isinstance(result, tuple) and len(result) >= 1:
            obs = result[0]
        else:
            obs = result
        return obs

    def step(self, action):
        result = self.env.step(action)
        # Nueva API: (obs, reward, terminated, truncated, info)
        if isinstance(result, tuple) and len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = bool(terminated or truncated)
            return obs, reward, done, info
        # API legacy de 4 valores: (obs, reward, done, info)
        elif isinstance(result, tuple) and len(result) == 4:
            return result
        else:
            raise ValueError(f"Unexpected step() result: {result}")

