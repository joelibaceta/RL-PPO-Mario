import gym
import numpy as np
from gym.spaces import Box

def dynamic_crop_nohud(obs: np.ndarray, x_pos: int, hud_height: int, crop_size: int, screen_width: int) -> np.ndarray:
    # 1) quita HUD
    obs_nohud = obs[hud_height:, :, :]               # (H', W, C)
    H_nohud, W, C = obs_nohud.shape

    # 2) calcula cámara tal como en el juego
    camera_left = min(max(0, x_pos - screen_width // 2), screen_width - crop_size)
    x_on_screen  = x_pos - camera_left
    left = int(x_on_screen - crop_size // 2)
    left = max(0, min(left, screen_width - crop_size))

    # 3) recorta la ventana de tamaño crop_size desde abajo
    top_crop = max(0, H_nohud - crop_size)
    return obs_nohud[top_crop : top_crop + crop_size,
                     left     : left + crop_size,
                     :]

class FrameCropWrapper(gym.Wrapper):
    """
    1) Elimina HUD (barra superior).
    2) Recorta dinámicamente un cuadrado crop_size×crop_size centrado en Mario.
    """
    def __init__(self, env, hud_height=34, crop_size=160):
        super().__init__(env)
        self.hud_height = hud_height
        self.crop_size  = crop_size

        # Detectamos canales
        shp = env.observation_space.shape
        C = shp[2] if len(shp) == 3 else 1
        # Definimos nuevo espacio de observación
        self.observation_space = Box(
            low=0, high=255,
            shape=(crop_size, crop_size, C),
            dtype=np.uint8
        )

    def reset(self):
        out = self.env.reset()
        if isinstance(out, tuple):
            obs, info = out
        else:
            obs, info = out, {}
        obs = self._crop(obs, info)
        return (obs, info) if isinstance(out, tuple) else obs

    def step(self, action):
        out = self.env.step(action)
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
            done = terminated or truncated
        else:
            obs, reward, done, info = out
        obs = self._crop(obs, info)
        if len(out) == 5:
            return obs, reward, terminated, truncated, info
        else:
            return obs, reward, done, info

    def _crop(self, obs, info):
        x_pos = info.get("x_pos", info.get("x_scroll", 0))
        screen_width = obs.shape[1]
        return dynamic_crop_nohud(
            obs, x_pos,
            hud_height=self.hud_height,
            crop_size=self.crop_size,
            screen_width=screen_width
        )