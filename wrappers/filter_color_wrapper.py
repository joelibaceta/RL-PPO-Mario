import gym
import numpy as np
import cv2

class FilterColorsWrapper(gym.ObservationWrapper):
    """
    Filtra y reemplaza rangos de colores específicos en la observación.
    Útil para eliminar cielos o HUDs de juegos retro.
    """
    def __init__(self, env, color_filters):
        """
        Args:
            color_filters: lista de tuplas (lower, upper, replacement)
                - lower: límite inferior RGB (tuple)
                - upper: límite superior RGB (tuple)
                - replacement: color RGB con el que se reemplazarán los píxeles (tuple)
        """
        super().__init__(env)
        self.color_filters = [
            (np.array(lower, dtype=np.uint8),
             np.array(upper, dtype=np.uint8),
             np.array(replacement, dtype=np.uint8))
            for (lower, upper, replacement) in color_filters
        ]
        self.observation_space = env.observation_space

    def observation(self, obs):
        for lower, upper, replacement in self.color_filters:
            mask = cv2.inRange(obs, lower, upper)
            obs[mask > 0] = replacement
        return obs