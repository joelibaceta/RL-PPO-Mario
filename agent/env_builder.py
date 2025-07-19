import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from gym.vector import SyncVectorEnv, AsyncVectorEnv
from gym.wrappers import FrameStack, GrayScaleObservation, ResizeObservation

from wrappers.frame_skip_wrapper import FrameSkipWrapper
from wrappers.frame_crop_wrapper import FrameCropWrapper
from wrappers.filter_color_wrapper import FilterColorsWrapper
from wrappers.life_reset_wrapper import LifeResetWrapper


def make_mario_env(env_id="SuperMarioBros-v0", seed=None, log_dir="logs_cleanrl", render_mode="rgb_array"):
    """
    Crea un entorno de Mario compatible con Gym v26+ y CleanRL.
    """
    env = gym.make(env_id, apply_api_compatibility=True, render_mode=render_mode)

    # ✅ Aplica JoypadSpace para trabajar con índices discretos
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    #env = LifeResetWrapper(env)  # Resetea al perder una vida
    env = FrameSkipWrapper(env, skip=4)
    env = FrameCropWrapper(env, hud_height=34, crop_size=200)
    env = ResizeObservation(env, (84, 84))
    #env = FilterColorsWrapper(
    #    env,
    #    color_filters=[((84, 120, 240), (110, 170, 255), (0, 0, 0))]  # Azul → Negro
    #)
    env = GrayScaleObservation(env, keep_dim=False)
    env = FrameStack(env, num_stack=4)

    return env


def make_vec_mario_env(num_envs=8, seed=0, async_mode=False):
    """
    Crea varios entornos de Mario en paralelo.
    """
    def make_env_fn(rank):
        def _init():
            env = make_mario_env(seed=seed + rank, log_dir=f"logs/env_{rank}")
            return env
        return _init

    env_fns = [make_env_fn(i) for i in range(num_envs)]
    return AsyncVectorEnv(env_fns) if async_mode else SyncVectorEnv(env_fns)