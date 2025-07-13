"""
Módulo para construir el entorno de SuperMarioBros compatible con Gymnasium v26+/CleanRL.
"""
import gym
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
from wrappers.monitor_wrapper import MonitorWrapper 
from gym.wrappers import FrameStack, TransformObservation, GrayScaleObservation


from wrappers.life_reset_wrapper import LifeResetWrapper
from wrappers.frame_skip_wrapper import FrameSkipWrapper
from wrappers.frame_crop_wrapper import FrameCropWrapper
from wrappers.filter_color_wrapper import FilterColorsWrapper
from wrappers.repaint_mario_wrapper import RepaintMarioWrapper

from wrappers.resize_observation_wrapper import ResizeObservation

import numpy as np

def make_mario_env(env_id: str = "SuperMarioBros-v0", seed: int = None, log_dir: str = "logs_cleanrl"):
    """
    Crea un entorno de Mario con:
      - Gymnasium v26+ API (reset→(obs,info), step→5-tupla)
      - JoypadSpace(SIMPLE_MOVEMENT) para acciones reducidas
      - Semilla reproducible
      - MonitorWrapper para métricas en TensorBoard

    Args:
        env_id: ID del entorno (e.g. 'SuperMarioBros-v0').
        seed: semilla RNG opcional.
        log_dir: carpeta para los logs de TensorBoard.
    Returns:
        Un gym.Env listo para entrenamiento.
    """

 
    # 1) gym.make con compatibilidad de API nueva
    env = gym.make(env_id, apply_api_compatibility=True, render_mode="rgb_array")

    # 2) restringir acciones a SIMPLE_MOVEMENT
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    
    env = FrameSkipWrapper(env, skip=4) 

    env = MonitorWrapper(env, log_dir=log_dir)

    #env = LifeResetWrapper(env)  

    env = FrameCropWrapper(env, hud_height=34, crop_size=160)
    env = ResizeObservation(env, (84, 84))

    #env = RepaintMarioWrapper(env)
 
    env = FilterColorsWrapper(
        env,
        color_filters=[
            ((80, 120, 240), (110, 170, 255), (0, 0, 0)),   # Azul → Negro 
        ]
    )

    env = ResizeObservation(env, target_shape=(240, 240))

    #env = GrayScaleObservation(env, keep_dim=True)



    env = FrameStack(env, num_stack=4)
    # 3) aplicar MonitorWrapper para métricas

    return env