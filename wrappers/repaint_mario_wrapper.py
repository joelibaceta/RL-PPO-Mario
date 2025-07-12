import gym
import numpy as np
import cv2

class RepaintMarioWrapper(gym.ObservationWrapper):
    """
    Detecta a Mario por color y lo recolorea con un color artificial.
    """
    def __init__(self, env, repaint_color=(255, 0, 255)):
        super().__init__(env)
        self.repaint_color = repaint_color  # Magenta brillante

    def observation(self, obs):
        # Define rango de color para Mario (ajusta según necesidad)
        lower = np.array([150, 40, 0])  # Rango inferior (rojo/naranja)
        upper = np.array([255, 160, 120])  # Rango superior

        # Crear máscara para los píxeles de Mario
        mask = cv2.inRange(obs, lower, upper)

        # Recolorea solo las áreas de Mario
        obs[mask > 0] = self.repaint_color
        return obs